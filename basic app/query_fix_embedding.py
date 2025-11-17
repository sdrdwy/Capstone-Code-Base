
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
load_dotenv()
# 示例文本（多行）
with open("basic app/term.txt",'r') as f:
    text = f.read()

lines = text.strip().split('\n')

from langchain_core.documents import Document

documents = [
    Document(page_content=line.strip()) 
    for line in lines if line.strip()
]
embedding = DashScopeEmbeddings(model="text-embedding-v2")
# 3. 创建向量数据库（使用 Chroma）
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,  
    persist_directory="./chroma_db_embedding" 
)

print(f"已将 {len(documents)} 行文本存入向量库")