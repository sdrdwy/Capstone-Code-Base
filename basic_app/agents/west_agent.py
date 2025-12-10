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
from .base_agent import BaseAgent

load_dotenv()

# 定义状态
class AgentState(BaseModel):
    user_input: str
    agent_mode: str  # "basic", "deep"
    context: str
    result: str
    source_documents: List[Any]
    use_deep: bool
    history: Any

# 提示模板
BASIC_PROMPT = """你是一个专业的评估专家，你对医生的问诊建议需要参考以下问诊流程：
1. **主诉**：请患者用一句话说明最主要问题（如“脸红痒3天”）。
2. **现病史**：围绕皮损的起病时间、部位、形态（红斑/水疱/鳞屑等）、演变、诱因、缓解因素、伴随症状（痒/痛/发热等）及既往治疗反应。
3. **既往史**：询问皮肤病史、系统性疾病（如糖尿病、自身免疫病）、药物过敏史。
4. **个人史**：职业、接触物（化学品/宠物/新护肤品）、生活习惯、旅行或性行为史（如相关）。
5. **家族史**：直系亲属有无类似皮肤病或遗传性皮肤疾病。
6. **系统回顾**：关注关节、口腔、眼睛、淋巴结等有无异常。
7. **皮肤检查引导**：请患者描述皮损外观、分布（对称？曝光区？）、是否扩散。
8. **初步判断与建议**：综合信息后给出可能诊断、是否需检查（如真菌镜检、活检）、治疗建议及随访计划。"""

DEEP_PROMPT = """你是一个专业的评估专家，你对医生的问诊建议需要参考以下问诊流程：
1. **主诉**：请患者用一句话说明最主要问题（如“脸红痒3天”）。
2. **现病史**：围绕皮损的起病时间、部位、形态（红斑/水疱/鳞屑等）、演变、诱因、缓解因素、伴随症状（痒/痛/发热等）及既往治疗反应。
3. **既往史**：询问皮肤病史、系统性疾病（如糖尿病、自身免疫病）、药物过敏史。
4. **个人史**：职业、接触物（化学品/宠物/新护肤品）、生活习惯、旅行或性行为史（如相关）。
5. **家族史**：直系亲属有无类似皮肤病或遗传性皮肤疾病。
6. **系统回顾**：关注关节、口腔、眼睛、淋巴结等有无异常。
7. **皮肤检查引导**：请患者描述皮损外观、分布（对称？曝光区？）、是否扩散。
8. **初步判断与建议**：综合信息后给出可能诊断、是否需检查（如真菌镜检、活检）、治疗建议及随访计划。"""

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

class WestAgent(BaseAgent):
    """
    西医知识检索Agent
    """
    
    def __init__(self, llm, retriever, **kwargs):
        super().__init__(llm, **kwargs)
        self.retriever = retriever
        self.setup_agent()
        
    def setup_agent(self):
        """
        设置agent的特定配置
        """
        # 提示模板已经定义在类外面，这里不需要重复定义
        pass

    def create_agent(self):
        """创建医疗问答Agent"""
        
        def determine_agent_mode(state: AgentState) -> Dict[str, Any]:
            """确定代理模式 - 只判断是否需要深度解析"""
            deep_classifier = DeepModeClassifier(llm=self.llm)
            use_deep = deep_classifier._run(state.user_input)
            
            agent_mode = "deep" if use_deep else "basic"
            
            return {
                "agent_mode": agent_mode,
                "use_deep": use_deep
            }

        def retrieve_context(state: AgentState) -> Dict[str, Any]:
            """检索上下文"""
            retrieved_docs = self.retriever.invoke(state.user_input)
            context = format_docs(retrieved_docs)
            
            return {
                "context": context,
                "source_documents": retrieved_docs
            }

        def generate_response(state: AgentState) -> Dict[str, Any]:
            """根据模式生成响应"""
            input_data = {"context": state.context, "question": state.user_input, "history":state.history}
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", DEEP_PROMPT),
                ("human", "参考资料：\n{context}\n\n问题：{question}\n\n前文问诊对话记录{history}")
            ])
            
            chain = prompt_template | self.llm | StrOutputParser()
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

    def query(self, user_query: str, history) -> Dict[str, Any]:
        """
        执行查询
        """
        agent = self.create_agent()
        final_state = agent.invoke({
            "user_input": user_query,
            "agent_mode": "basic",
            "context": "",
            "result": "",
            "source_documents": [],
            "use_deep": True,
            "history":history
        })
        
        # 提取检索文档的原文内容
        retrieved_docs_content = [doc.page_content for doc in final_state["source_documents"]]
        
        return {
            "result": final_state["result"],
            "retrieved_docs": retrieved_docs_content,
            "agent_mode": final_state["agent_mode"],
            "use_deep": final_state["use_deep"]
        }


def create_medical_agent(llm, retriever):
    """创建医疗问答Agent（传统函数接口，为了向后兼容）"""
    
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