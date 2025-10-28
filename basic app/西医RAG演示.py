# %%
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 加载已有向量库
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. 构建 retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. 初始化 LLM（Qwen via DashScope）
llm = ChatOpenAI(
    model="qwen-max",
    temperature=0.3,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 修正：参数名是 base_url
    api_key="sk-db85459561d04810ac504107dbd02936"
)

# 4. 构建 RAG 链（现代方式）
basic_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一位经验丰富的皮肤科临床辅助诊疗专家，正在辅助不同背景的用户理解皮肤病相关知识。"
     "用户可能是医学生、住院医师、资深中医、普通患者或医学爱好者。"
     "请根据问题中隐含的专业程度，自动调整回答的深度与术语使用："
     "\n- 若问题包含专业术语或机制探讨，可使用规范医学术语，并简要解释关键概念；"
     "\n- 若问题偏向症状描述或日常护理，请用通俗易懂的语言，避免 jargon；"
     "\n- 始终保持尊重、耐心与同理心，不假设、不标签用户身份；"
     "\n- 基于提供的上下文作答，若信息不足，请说明“现有资料较少，我将尽我所能为你解释”，并且根据你的原有知识作答；"
     "\n- 回答需简洁，聚焦核心信息，避免冗长；"
     "\n- 在回答末尾，用开放式提问引导用户深入探讨（如：‘你是否还想了解其鉴别诊断？’ 或 ‘需要我解释治疗方案的细节吗？’）"
    ),
    ("human", "参考资料：\n{context}\n\n用户问题：{question}")
])

deep_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你现在进入深度诊疗辅助模式。请严格遵循临床思维链（Chain of Thought）进行结构化分析，"
     "基于提供的参考资料，按以下逻辑顺序逐步推导并回答问题：\n"
     "1. **核心问题识别**：明确用户所问疾病的名称或核心症状。\n"
     "2. **病理机制/中医病机**：简述现代医学的病理生理基础或中医的病因病机（如风、湿、热、血瘀等）。\n"
     "3. **典型临床表现**：列出关键体征、症状特点及好发部位。\n"
     "4. **鉴别诊断要点**：指出需与哪些常见皮肤病区分，并说明关键鉴别特征。\n"
     "5. **诊疗建议**：\n"
     "   - 西医：一线治疗方案（如外用/系统药物）；\n"
     "   - 中医：辨证分型及对应治法方药（若资料支持）；\n"
     "   - 生活调护：日常注意事项或避免诱因。\n"
     "6. **知识边界说明**：若参考资料不足以覆盖上述任一环节，请明确指出“现有资料未提及XX部分”。\n\n"
     "要求：\n"
     "- 使用规范医学术语，但对关键概念（如“Th17通路”“血热证”）需简要解释；\n"
     "- 逻辑清晰，分点陈述，避免冗长段落；\n"
     "- 结尾提出一个值得深入探讨的临床问题（如机制、治疗难点或中西医结合切入点）。"
    ),
    ("human", "参考资料：\n{context}\n\n问题：{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

basic_chain = ( basic_prompt
    | llm
    | StrOutputParser()
)

deep_chain = (deep_prompt
    | llm
    | StrOutputParser()
)

# 5. 交互循环
print("✅ 知识库加载完成！请输入您的皮肤相关问题（输入 'quit' 或 'exit' 退出）：")

# 统一问答入口 
def ask_with_mode(question: str, use_deep: bool):
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)
    
    # 将 context 和 question 传给链
    input_data = {"context": context, "question": question}
    
    if use_deep:
        answer = deep_chain.invoke(input_data)
    else:
        answer = basic_chain.invoke(input_data)
    
    return {
        "result": answer,
        "source_documents": retrieved_docs
    }

# === 主交互循环 ===
while True:
    question = input("\n❓ 你的问题: ").strip()
    if question.lower() in ["quit", "exit", "退出"]:
        print("👋 再见！")
        break
    if not question:
        continue

    # 第一步：先问是否深度思考
    while True:
        deep_choice = input("🔍 是否进入深度思考模式？(y/n): ").strip().lower()
        if deep_choice in ['y', 'yes', '是']:
            use_deep = True
            break
        elif deep_choice in ['n', 'no', '否', '']:
            use_deep = False
            break
        else:
            print("请输入 y 或 n")

    try:
        print('正在思考中，请稍后...')
        result = ask_with_mode(question, use_deep=use_deep)

        # 根据模式显示不同标题
        mode_label = "🔬 深度解析" if use_deep else "💡 简明回答"
        print(f"\n{mode_label}：{result['result']}")
        
        # 显示参考来源（两种模式都保留，增强可信度）
        print("\n📚 部分参考片段：")
        for i, doc in enumerate(result["source_documents"][:2], 1):
            print(f"  [{i}] {doc.page_content[:200]}...")

    except Exception as e:
        print(f"❌ 出现错误: {e}")
# %%
