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
        """

        BASIC_PROMPT = """
        你是一位经验丰富的中医皮肤病领域专家，正在辅助专业的医生进行问诊。
        请根据问题中隐含的专业程度，自动调整回答的深度与术语使用：
        - 若问题包含专业术语或机制探讨，可使用规范医学术语，并简要解释关键概念；
        - 若问题偏向症状描述或日常护理，请用通俗易懂的语言，避免 jargon；
        - 始终保持尊重、耐心与同理心，不假设、不标签用户身份；
        - 基于提供的上下文作答，若信息不足，请说明'现有资料较少，我将尽我所能为你解释'，并且根据你的原有知识作答；
        - 回答需简洁，聚焦核心信息，避免冗长；
        """

        self.cypher_prompt = PromptTemplate(
            template=CYPHER_GENERATION_TEMPLATE,
            input_variables=["schema", "question"]
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
                    ("system", BASIC_PROMPT),
                    ("human", "参考资料：\n{context}\n\n用户问题：{question}")
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

    def query(self, query: str):
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
                                         'question':query})
        
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