from langchain_neo4j import Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from dotenv import load_dotenv
load_dotenv()
graph = Neo4jGraph(database='neo4j')
llm = ChatTongyi(
    model="qwen-flash",        
    temperature=0,
    # max_tokens=2048,
)
CYPHER_GENERATION_TEMPLATE = """
你是一个 Neo4j 专家，任务是将自然语言问题转换为 Cypher 查询。
图数据库的 schema 如下：
{schema}

请严格遵守以下规则：
1. **节点匹配必须使用 `id` 属性进行精确匹配**，例如：`(n:皮肤病 {{id: "扁平疣"}})`
2. 不要使用 `CONTAINS`、`=~` 或其他模糊匹配。
3. 只返回 Cypher 查询语句，不要解释，不要 markdown，不要反引号。
4. 对于病症遵循以下规则：皮肤病-[辨证为]->证型-[主症包括]->症状
5. 证型-[治法为]->方剂-[用于治疗]->皮肤病
问题：{question}
你应该按照以下步骤：
1. 按照皮肤病,证型,症状,方剂的分类从用户提问中提取出这些关键词
2. 利用这些关键词来生成查询语句，要求尽可能简单并且严格遵循Cypher语法限制
"""

cypher_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"]
)

chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    cypher_prompt=cypher_prompt,  
    verbose=True,
    allow_dangerous_requests=True,
    validate_cypher=True,
    fix_cypher=True, 
    max_fix_attempts=2,
)



prompt_query_fix = ChatPromptTemplate.from_template(
"""你是一个中医皮肤病领域专家，
你现在需要协助把用户提出的问题中用到的中医描述替换成关键词中的术语
你必须保证替换后的关键词和术语中的完全一致
你不能根据术语做过多的延伸解释
你必须保证输出的关键词数量和提取出来的关键词数量完全一致
你应该遵循以下步骤：
1. 按照方剂,皮肤病,证型,症状的分类提取出关键词,注意,其中有一些分类的关键词可能没有,如果没有,那就不需要提取那个分类的关键词
2. 必须把这些提取到的关键词替换成上下文中存在的文本
3. 按照类别输出这些关键词,并严格遵循提取出来的顺序,并使用 '|'分隔不同类别的关键词
4. 最后输出的格式为关键词: \n 替换关键词后的问题: 
参考术语：
{context}

问题：
{question}

回答："""
)


embedding = DashScopeEmbeddings(model="text-embedding-v2")
vectorstore = Chroma(
    persist_directory="basic app/chroma_db_embedding",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs  = {"k":10}
)

rag_chain = (
    {"context":retriever,"question":RunnablePassthrough()}
    | prompt_query_fix
    | llm
    | StrOutputParser()
)

# query_fix = rag_chain.invoke("草还丹可以用来治疗什么")
# print(query_fix)
# response = chain.invoke({"query": query_fix})
# print(response)

if __name__ == "__main__":
    while True:
        query_origin = input("your query:")
        if query_origin == "-1":
            break
        query_fix = rag_chain.invoke(query_origin)
        print(f"Fixed query:{query_fix}")
        response = chain.invoke({"query":query_fix})
        print(response["result"])