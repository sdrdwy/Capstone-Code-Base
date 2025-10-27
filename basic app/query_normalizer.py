import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
load_dotenv()
import tiktoken

def get_token_length(text:str) ->int:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    return len(tokens)
loader = TextLoader("basic app/term.txt",encoding="UTF-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2048,
    chunk_overlap = 50,
    length_function = get_token_length,
    # separators= ["----"]
)

splits = text_splitter.split_documents(docs)

splits = [doc for doc in splits if doc.page_content.strip()]


for i, split in enumerate(splits):  
    print(f"\n--- 片段 {i+1} ---")
    print(split.page_content[:10] + "...")

print(len(splits))

from langchain_chroma import Chroma

embedding = DashScopeEmbeddings(model="text-embedding-v2")
persist_directory = "./basic app/chroma_db"

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory  
)