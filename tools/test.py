from langchain_community.graphs import Neo4jGraph
import re
import os
# === 配置 Neo4j 连接 ===
os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

graph = Neo4jGraph()
# === 定义解析函数 ===
def parse_disease_name(name: str):
    """
    输入: "疔疮 (疔与痈病)" 或 "扁平疣"
    输出: (new_id, alias)
    """
    # 统一替换全角括号为半角
    name = name.replace("（", "(").replace("）", ")")
    # 匹配 "前缀 (后缀)" 模式
    match = re.search(r"^(.+?)\s*\((.+?)\)$", name.strip())
    if match:
        alias = match.group(1).strip()
        disease_id = match.group(2).strip()
        return disease_id, alias
    else:
        return name.strip(), "无"

# === 分批处理所有皮肤病节点 ===
batch_size = 50
skip = 0

while True:
    # 查询一批未处理的皮肤病节点（无 alias 属性 或 alias='无' 但含括号）
    query_fetch = """
    MATCH (d:皮肤病)
    WHERE d.alias IS NULL 
    OR (d.alias = $no_alias AND (d.id CONTAINS $paren1 OR d.id CONTAINS $paren2))
    RETURN d.id AS old_id, id(d) AS node_id
    SKIP $skip
    LIMIT $limit
    """

    batch = graph.query(query_fetch, params={
        "no_alias": "无",
        "paren1": "(",
        "paren2": "（",
        "skip": skip,
        "limit": batch_size
    })
    print(batch)
    if not batch:
        break

    # 构建更新语句
    updates = []
    for record in batch:
        old_id = record["old_id"]
        node_id = record["node_id"]
        new_id, alias = parse_disease_name(old_id)
        updates.append({'node_id':node_id, 'new_id':new_id, 'alias':alias})
    print(updates)
    # 批量更新（使用 id() 精准定位节点，避免唯一性冲突）
    update_query = """
    UNWIND $updates AS update
    MATCH (d) WHERE id(d) = update.node_id
    SET d.id = update.new_id, d.alias = update.alias
    """
    graph.query(update_query, params={"updates": updates})
    
    print(f"Processed {len(updates)} nodes (skip={skip})")
    skip += batch_size

print("✅ 所有皮肤病节点处理完成！")