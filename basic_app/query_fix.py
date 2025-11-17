from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from dotenv import load_dotenv
load_dotenv()


def fix_query(query, llm, vectorestore,topk=10):
    prompt_query_fix = ChatPromptTemplate.from_template(
    """你是一个中医皮肤病领域专家，
    你现在需要协助把用户提出的问题中用到的中医描述替换成关键词中的术语
    你必须保证替换后的关键词和术语中的完全一致
    你不能根据术语做过多的延伸解释
    你必须保证输出的关键词数量和提取出来的关键词数量完全一致
    你应该遵循以下步骤：
    1. 按照方剂,皮肤病,证型,症状的分类提取出关键词,注意,其中有一些分类的关键词可能没有,如果没有,那就不需要提取那个分类的关键词
    2. 必须把这些提取到的关键词替换成上下文中存在的文本
    3. 按照类别输出这些关键词,并严格遵循提取出来的顺序
    4. 最后输出的格式为关键词: \n 替换关键词后的问题: 
    参考术语：
    {context}

    问题：
    {question}

    回答："""
    )
    retriever = vectorestore.as_retriever(
        search_type = "similarity",
        search_kwargs  = {"k":topk}
    )
    chain = (
        {"context":retriever,"question":RunnablePassthrough()}
        | prompt_query_fix
        | llm
        | StrOutputParser()
    )
    
    retrieved_docs = retriever.invoke(query)
    fixed_query = chain.invoke(query)
    related = []
    for doc in retrieved_docs:
        related.append(doc.page_content)
    return {
        "query": fixed_query,
        "topk": related
    }



if __name__ == "__main__":
    query = input()

    embedding = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = Chroma(
        persist_directory="basic_app/chroma_db_embedding",
        embedding_function=embedding
    )

    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs  = {"k":10}
    )
    llm = ChatTongyi(model="qwen-flash",temperature=0)
    query_fix = fix_query(query,llm,vectorstore,10)
    print(query_fix)