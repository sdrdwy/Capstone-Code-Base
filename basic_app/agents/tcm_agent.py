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


class TcmAgent(BaseAgent):
    """
    中医知识图谱Agent
    """
    
    def __init__(self, llm, graph, **kwargs):
        super().__init__(llm, **kwargs)
        self.graph = graph
        self.setup_agent()
        
    def setup_agent(self):
        """
        设置agent的特定配置
        """
        CYPHER_GENERATION_TEMPLATE = """
        你是一个 Neo4j 专家，任务是将自然语言问题转换为 Cypher 查询。
        图数据库的 schema 如下：
        {schema}

        请严格遵守以下规则：
        1. **节点匹配必须使用 `id` 属性进行精确匹配**，例如：`(n:皮肤病 {id: "扁平疣"})`
        2. 不要使用 `CONTAINS`、`=~` 或其他模糊匹配。
        3. 只返回 Cypher 查询语句，不要解释，不要 markdown，不要反引号。
        4. 对于病症遵循以下规则：皮肤病-[辨证为]->证型-[主症包括]->症状
        5. 证型-[治法为]->方剂-[用于治疗]->皮肤病
        问题：{question}
        你应该按照以下步骤：
        1. 按照皮肤病,证型,症状,方剂的分类从用户提问中提取出这些关键词
        2. 利用这些关键词来生成查询语句,要求尽可能简单并且严格遵循Cypher语法限制
        """

        self.cypher_prompt = PromptTemplate(
            template=CYPHER_GENERATION_TEMPLATE,
            input_variables=["schema", "question"]
        )
        
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
            # return_direct=True,
        )
        return chain

    def query(self, query: str):
        """
        执行查询
        """
        agent = self.create_agent()
        response = agent.invoke({"query": query})
        return response


def rag_query(graph, llm, query):
    """
    传统函数接口，为了向后兼容
    """
    agent = TcmAgent(llm, graph)
    return agent.query(query)


if __name__ == "__main__":
    graph = Neo4jGraph(database=os.environ["DB_NAME"])
    print(graph.schema)
    # exit(0)
    llm = ChatTongyi(
        model="qwen-max",        
        temperature=0,
        # max_tokens=2048,
    )
    embedding = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = Chroma(
        persist_directory="basic_app/chroma_db_embedding",
        embedding_function=embedding
    )
    query = input()
    fixed = fix_query(query,llm,vectorstore,10)
    print(fixed['query'])
    fixed_query = fixed['query']
    ret = rag_query(graph,llm,fixed_query)
    print(ret)


# query_fix = rag_chain.invoke("草还丹可以用来治疗什么")
# print(query_fix)
# response = chain.invoke({"query": query_fix})
# print(response)