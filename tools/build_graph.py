import os

from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_community.chat_models import ChatTongyi

import json

os.environ["DASHSCOPE_API_KEY"] = "sk-c763fc92bf8c46c7ae31639b05d89c96"

os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

graph = Neo4jGraph()
print("graph prepared")

llm = ChatTongyi(
    model="qwen-max",        
    temperature=0,
    # max_tokens=2048,
)
print("llm prepared")

disease = []
FILE_PATH = "disease.jsonl"
with open(FILE_PATH,"r") as file:
    content = file.readlines()
    s = ""
    for l in content:
        s += l
        if l == "}\n":
            disease.append(json.loads(s))
            s = ""
print("file loaded")

text = json.dumps(disease[1],ensure_ascii=False) # + json.dumps(disease[1],ensure_ascii=False)
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["皮肤病", "证型", "症状", "方剂"],
    allowed_relationships=[
        "辨证为",      # 皮肤病 → 证型
        "主症包括",    # 证型 → 症状
        "治法为",      # 证型 → 治法（可扩展）
        "方剂包含",    # 方剂 → 中药（若包含中药节点）
        "用于治疗"     # 方剂 → 皮肤病
    ],
    strict_mode=True,
    node_properties=["别名", "病因", "病机"],
    relationship_properties=["依据", "强度"],
    additional_instructions="""
    json格式中，name属性对应皮肤病和别名，症状在key_point中，用于治疗和方剂在solution里面
    皮肤病带括号的括号内是别名,各个节点的关系如下：皮肤病 → 证型，证型 → 症状，证型 → 治法（可扩展），方剂 → 中药（若包含中药节点），方剂 → 皮肤病
    """
)
documents = [Document(page_content=text)]
print("document prepared")
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print("transformed")
print(f"Nodes:{graph_documents[0].nodes}")
# print(f"Relationships:{graph_documents[0].relationships}")

graph.add_graph_documents(graph_documents)
print("node added")