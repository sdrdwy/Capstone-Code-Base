from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

# 定义状态
class AgentState(BaseModel):
    user_input: str
    agent_mode: str  # "basic", "deep"
    context: str
    result: str
    source_documents: List[Any]
    use_deep: bool

# 提示模板
BASIC_PROMPT = """你是一位经验丰富的皮肤科临床辅助诊疗专家，正在辅助不同背景的用户理解皮肤病相关知识。
用户可能是医学生、住院医师、普通患者或医学爱好者。
请根据问题中隐含的专业程度，自动调整回答的深度与术语使用：
- 若问题包含专业术语或机制探讨，可使用规范医学术语，并简要解释关键概念；
- 若问题偏向症状描述或日常护理，请用通俗易懂的语言，避免 jargon；
- 始终保持尊重、耐心与同理心，不假设、不标签用户身份；
- 基于提供的上下文作答，若信息不足，请说明'现有资料较少，我将尽我所能为你解释'，并且根据你的原有知识作答；
- 回答需简洁，聚焦核心信息，避免冗长；
- 在回答末尾，用开放式提问引导用户深入探讨"""

DEEP_PROMPT = """你现在进入深度诊疗辅助模式。请按照用户要求作答，保证回答用户的所有问题
如果用户没有格式要求，就请严格遵循临床思维链（Chain of Thought）进行结构化分析，
基于提供的参考资料，按以下逻辑顺序逐步推导并回答问题：
1. **核心问题识别**：明确用户所问疾病的名称或核心症状。
2. **病理机制**：简述现代医学的病理生理基础。
3. **典型临床表现**：列出关键体征、症状特点及好发部位。
4. **鉴别诊断要点**：指出需与哪些常见皮肤病区分，并说明关键鉴别特征。
5. **诊疗建议**：
   - 一线治疗方案（如外用/系统药物）；
   - 生活调护：日常注意事项或避免诱因。
6. **知识边界说明**：若参考资料不足以覆盖上述任一环节，请明确指出'现有资料未提及XX部分'。

要求：
- 使用规范医学术语，但对关键概念（如'Th17通路'）需简要解释；
- 逻辑清晰，分点陈述，避免冗长段落；
- 结尾提出一个值得深入探讨的临床问题"""

# 工具 - 使用LLM判断是否需要深度解析
class DeepModeClassifier(BaseTool):
    name: str = Field(default="deep_mode_classifier")
    description: str = Field(default="使用LLM判断是否需要深度思考模式")
    llm: Any = Field(exclude=True)  # 声明 llm 为可输入字段，但不包含在序列化中

    def __init__(self, llm, **data):
        super().__init__(llm=llm, **data) # 调用父类初始化时传入 llm

    def _run(self, question: str) -> bool:
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "请判断用户的问题是否需要深度解析模式。"
             "如果问题涉及以下情况之一，请返回True：\n"
             "- 需要详细的病理机制解释\n"
             "- 需要鉴别诊断\n"
             "- 需要完整的诊疗方案\n"
             "- 需要深入的临床分析\n"
             "- 需要多个步骤的推理过程\n"
             "如果问题只是简单的信息查询或常识性问题，请返回False。"
             "只返回True或False，不要其他内容。"
            ),
            ("human", f"用户问题：{question}")
        ])
        
        classification_chain = classification_prompt | self.llm | StrOutputParser()
        response = classification_chain.invoke({})
        
        return response.strip().lower() in ['true', 'yes', '是', '需要']

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_medical_agent(llm, retriever):
    """创建医疗问答Agent"""
    
    def determine_agent_mode(state: AgentState) -> Dict[str, Any]:
        """确定代理模式 - 只判断是否需要深度解析"""
        deep_classifier = DeepModeClassifier(llm=llm)
        use_deep = deep_classifier._run(state.user_input)
        
        agent_mode = "deep" if use_deep else "basic"
        
        return {
            "agent_mode": agent_mode,
            "use_deep": use_deep
        }

    def retrieve_context(state: AgentState) -> Dict[str, Any]:
        """检索上下文"""
        retrieved_docs = retriever.invoke(state.user_input)
        context = format_docs(retrieved_docs)
        
        return {
            "context": context,
            "source_documents": retrieved_docs
        }

    def generate_response(state: AgentState) -> Dict[str, Any]:
        """根据模式生成响应"""
        input_data = {"context": state.context, "question": state.user_input}
        
        if state.agent_mode == "deep":
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", DEEP_PROMPT),
                ("human", "参考资料：\n{context}\n\n问题：{question}")
            ])
        else:  # basic
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", BASIC_PROMPT),
                ("human", "参考资料：\n{context}\n\n用户问题：{question}")
            ])
        
        chain = prompt_template | llm | StrOutputParser()
        result = chain.invoke(input_data)
        
        return {"result": result}

    # 构建LangGraph
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("determine_mode", determine_agent_mode)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    
    # 设置入口点
    workflow.set_entry_point("determine_mode")
    
    # 定义边
    workflow.add_edge("determine_mode", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # 编译图
    return workflow.compile()

def medical_qa_pipeline(llm_choice: str, vector_db_path: str, user_query: str, 
                       temperature: float = 0.3, k: int = 3) -> Dict[str, Any]:
    """
    医疗问答管道函数
    
    Args:
        llm_choice: LLM选择 ("qwen-flash", "qwen-max", 或其他支持的模型)
        vector_db_path: 向量库路径
        user_query: 用户查询
        temperature: LLM温度参数
        k: 检索文档数量
    
    Returns:
        Dict包含：
        - "answer": 模型回答
        - "retrieved_docs": 检索到的文档内容列表
        - "agent_mode": 使用的代理模式 ("basic" 或 "deep")
        - "use_deep": 是否使用深度模式
    """
    
    # 初始化LLM
    if llm_choice == "qwen-flash" or llm_choice.startswith("qwen"):
        llm = ChatTongyi(
            model=llm_choice,
            temperature=temperature,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
    else:
        # 其他LLM初始化逻辑可以在这里添加
        raise ValueError(f"Unsupported LLM choice: {llm_choice}")
    
    # 加载向量库
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # 创建Agent
    agent = create_medical_agent(llm=llm, retriever=retriever)
    
    # 执行Agent
    final_state = agent.invoke({
        "user_input": user_query,
        "agent_mode": "basic",
        "context": "",
        "result": "",
        "source_documents": [],
        "use_deep": False
    })
    
    # 提取检索文档的原文内容
    retrieved_docs_content = [doc.page_content for doc in final_state["source_documents"]]
    
    return {
        "answer": final_state["result"],
        "retrieved_docs": retrieved_docs_content,
        "agent_mode": final_state["agent_mode"],
        "use_deep": final_state["use_deep"]
    }

# 使用示例
if __name__ == "__main__":
    # 示例调用
    result = medical_qa_pipeline(
        llm_choice="qwen-flash",
        vector_db_path="./chroma_db_dash_w",
        user_query="湿疹用什么药？"
    )
    
    print(f"回答: {result['answer']}")
    print(f"检索文档数量: {len(result['retrieved_docs'])}")
    print(f"使用模式: {result['agent_mode']}")
    print(f"深度模式: {result['use_deep']}")
    
    for i, doc in enumerate(result['retrieved_docs']):
        print(f"文档 {i+1}: {doc[:200]}...")