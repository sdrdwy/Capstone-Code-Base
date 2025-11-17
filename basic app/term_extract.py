import os
from langchain_neo4j import Neo4jGraph

from dotenv import load_dotenv
load_dotenv()

graph = Neo4jGraph(database="neo4j-2025-10-27t07-22-12")
def write_chunk(name, fp):
    result = graph.query(f"MATCH (n:`{name}`) RETURN n.id as name;")
    fp.write(name+": ----\n")
    for r in result:
        fp.write(r['name']+"\n")




with open("basic app/term.txt","w",encoding="UTF-8") as f:
    write_chunk("方剂",f)
    write_chunk("证型",f)
    write_chunk("症状",f)
    
    result = graph.query("MATCH (n:`皮肤病`) RETURN n.id as name,n.别名 as other, n.alias as alias;")
    f.write("病症: ----\n")
    f.write("名称\t别名\t对照\n")
    for r in result:
        f.write(f"{r['name']}\t{r['alias']}\t{r['other']}\n")