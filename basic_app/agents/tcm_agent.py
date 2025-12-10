from langchain_neo4j import Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from basic_app.utils.query_fix import fix_query
from .base_agent import BaseAgent
from dotenv import load_dotenv
load_dotenv()
import os

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class TcmAgent(BaseAgent):
    """
    中医知识图谱Agent
    """
    
    def __init__(self, llm, graph, retriever=None, **kwargs):
        super().__init__(llm, **kwargs)
        self.graph = graph
        self.setup_agent()
        if retriever != None:
            self.retriever = retriever



    def setup_agent(self):
        """
        设置agent的特定配置
        """
        CYPHER_GENERATION_TEMPLATE = """
        你是一个 Neo4j 专家，任务是将自然语言问题转换为 Cypher 查询。
        图数据库的 schema 如下：
        {schema}

        请严格遵守以下规则：
        1. **节点匹配必须使用 `id` 属性进行精确匹配**
        2. 不要使用 `CONTAINS`、`=~` 或其他模糊匹配。
        3. 只返回 Cypher 查询语句，不要解释，不要 markdown，不要反引号。
        4. 对于病症遵循以下规则：皮肤病-[辨证为]->证型-[主症包括]->症状
        5. 证型-[治法为]->方剂-[用于治疗]->皮肤病
        6. 查询语句长度不能超过100个字符
        问题：{question}
        你应该按照以下步骤：
        1. 按照皮肤病,证型,症状,方剂的分类从用户提问中提取出这些关键词
        2. 利用这些关键词来生成查询语句,要求尽可能简单并且严格遵循Cypher语法限制
        3. 确保查询语句简洁，不超过100字符
        4. 如果没有任何相关的内容，则生成一个只返回空值的查询语句
        """

        BASIC_PROMPT = """
        你是一个专业的评估专家，你对医生的问诊建议需要参考以下问诊流程：
        一、基础信息采集（简问）
        主诉：最困扰的皮肤问题是什么？（如“脸反复起红疹3个月”） 
        皮损特点：部位、形态（红/肿/干/流水/脱屑）、是否瘙痒/灼痛？ 
        诱因与变化：遇热/冷/情绪/饮食后是否加重？夜间是否更痒？ 
        全身伴随症状（关键！）： 
        怕冷 or 怕热？手脚凉 or 热？ 
        口干口苦？喜冷饮 or 热饮？ 
        大便干结 or 稀溏？小便黄 or 清？ 
        睡眠、情绪、女性月经情况？
        舌象（若可提供）：舌色（淡/红/紫）、苔（白/黄、厚/薄、腻/干）？
        其他如既往史、生活史等，仅在必要时追问。
        二、辨证分析框架（核心！）
        在获得上述信息后，必须按以下三层框架进行辨证：
        1. 八纲辨证（定总纲）
        表里：起病急、伴恶风 → 表证；久病、皮损深在 → 里证 
        寒热：皮疹色淡、喜暖、便溏、舌淡苔白 → 寒；皮疹红赤、灼热、喜冷、便结、舌红苔黄 → 热 
        虚实：病程久、乏力、皮损干燥脱屑 → 虚（血虚/阴虚）；起病急、红肿痒甚、苔厚腻 → 实（风/湿/热/毒）
        2. 病因病机辨证（定病性）
        结合皮损与全身症，判断主导病邪： 
        风：瘙痒剧烈、皮疹游走、起落不定（如荨麻疹） 
        湿：皮损糜烂、渗液、结痂、苔腻、大便黏滞 
        热 / 火 / 毒：红肿热痛、化脓、口干喜冷、尿黄便结 
        血热：皮疹鲜红、灼热、舌红绛（如急性湿疹、玫瑰糠疹） 
        血虚风燥：病久、皮损干燥、脱屑、夜间痒甚、面色无华 
        血瘀：皮损暗红、肥厚、结节、舌有瘀斑（如慢性湿疹、扁平苔藓） 
        肝郁脾虚：情绪诱发、腹胀便溏、月经不调
        3. 脏腑经络定位（定病位）
        肺：皮疹多在面部、上肢，伴鼻干、咽痒（肺主皮毛） 
        脾：渗液多、便溏、纳差、苔腻（脾主运化水湿） 
        肝：情绪相关、月经前加重、口苦（肝郁化火） 
        心：伴失眠、心烦、舌尖红（心主血，其华在面）
        """

        self.cypher_prompt = PromptTemplate(
            template=CYPHER_GENERATION_TEMPLATE,
            input_variables=["schema", "question"]
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
                    ("system", BASIC_PROMPT),
                    ("human", "参考资料：\n{context}\n\n用户问题：{question}\n\n前文问诊对话记录{history}")
                ])
        
    def context_retrieve(self,query):
        retrieved_docs = self.retriever.invoke(query)
        context = format_docs(retrieved_docs)
        return context
    
    def create_agent(self):
        """
        创建TCM agent
        """
        chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=self.llm,
            cypher_prompt=self.cypher_prompt,  
            verbose=True,
            allow_dangerous_requests=True,
            validate_cypher=True,
            fix_cypher=True, 
            max_fix_attempts=2,
            return_direct=True,
        )
        rag_chain = self.prompt_template | self.llm | StrOutputParser()
        return chain,rag_chain

    def query(self, query: str,history):
        """
        执行查询
        """
        agent,rag_agent = self.create_agent()
        response = agent.invoke({"query": query})
        try:
            graph_result = response.get('result', '') if isinstance(response, dict) else str(response)
        except:
            graph_result = []
        context = self.context_retrieve(query)
        
        response_rag = rag_agent.invoke({'query':query,
                                         'context':f"rag_result:{context}\n graph_result:{graph_result}",
                                         'question':query,
                                         'history':history})
        
        ret = {
            'graph':graph_result,
            'retrieved_docs':context,
            'result':response_rag
        }
        # print(response_rag)
        return ret


if __name__ == "__main__":
    graph = Neo4jGraph(database=os.environ["DB_NAME"])
    llm = ChatTongyi(
        model="qwen-max",        
        temperature=0,
    )
    embedding = DashScopeEmbeddings(model="text-embedding-v2")
    embedding2 = DashScopeEmbeddings(model="text-embedding-v3")
    vectorstore = Chroma(
        persist_directory="basic_app/chroma_db_embedding",
        embedding_function=embedding
    )

    med_vectorstore = Chroma(
        persist_directory="chroma_TCM_rag_db_qwen",
        embedding_function=embedding2,
        collection_name="medical_book_qwen"
    )
    med_retriever = med_vectorstore.as_retriever(search_kwargs={"k":4})
    query = input()
    fixed = fix_query(query,llm,vectorstore,10)
    print(fixed['query'])
    fixed_query = fixed['query']
    Agent = TcmAgent(llm=llm,graph=graph,retriever=med_retriever)
    ret = Agent.query(fixed_query)
    print(ret)


# query_fix = rag_chain.invoke("草还丹可以用来治疗什么")
# print(query_fix)
# response = chain.invoke({"query": query_fix})
# print(response)