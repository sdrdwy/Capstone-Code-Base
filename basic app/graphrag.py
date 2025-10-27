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
graph = Neo4jGraph(database='neo4j-2025-10-27t07-22-12')
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
你现在需要协助把用户提出的问题中用到的可能的中医术语替换成上下文中查询到的术语最合适的，用于cypher的查询,
并把替换后的问题返回出来注意回答是修正之后的提问，为不是回答的答案。
注意：不能修改上下文中的内容，回答中替换后的文本一定是上下文中那块部分的原文
例如：提问是“阴除湿汤可以用来治疗什么” 你要返回：“阴除湿汤加减可以用来治疗什么”

上下文：
{context}

问题：
{question}

回答："""
)


embedding = DashScopeEmbeddings(model="text-embedding-v2")
vectorstore = Chroma(
    persist_directory="basic app/chroma_db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs  = {"k":29}
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