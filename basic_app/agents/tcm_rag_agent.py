"""
中医RAG Agent - 专门处理中医知识的RAG检索增强生成
"""
from langchain_core.language_models import BaseLanguageModel
from .base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class TcmRagAgentState(BaseModel):
    """
    中医RAG Agent的状态
    """
    user_input: str
    agent_mode: str  # "basic", "deep"
    context: str
    result: str
    source_documents: List[Document]
    use_deep: bool


class TcmRagAgent(BaseAgent):
    """
    中医知识检索RAG Agent
    """
    
    def __init__(self, llm: BaseLanguageModel, retriever, **kwargs):
        super().__init__(llm, **kwargs)
        self.retriever = retriever
        self.setup_agent()
        
    def setup_agent(self):
        """
        设置agent的特定配置
        """
        # 中医基本模式提示
        self.basic_prompt = """你是一位经验丰富的中医专家，正在辅助用户理解中医相关知识。
用户可能是中医学生、住院医师、普通患者或中医爱好者。

请根据问题中隐含的专业程度，自动调整回答的深度与术语使用：
- 若问题包含中医专业术语或理论探讨，可使用规范中医术语，并简要解释关键概念；
- 若问题偏向症状描述或日常养生，请用通俗易懂的语言，避免 jargon；
- 始终保持尊重、耐心与同理心，不假设、不标签用户身份；
- 基于提供的上下文作答，若信息不足，请说明'现有资料较少，我将尽我所能为你解释'，并且根据你的原有知识作答；
- 回答需简洁，聚焦核心信息，避免冗长；
- 在回答末尾，用开放式提问引导用户深入探讨"""
        
        # 中医深度模式提示
        self.deep_prompt = """你现在进入深度中医诊疗辅助模式。请按照用户要求作答，保证回答用户的所有问题
如果用户没有格式要求，就请严格遵循中医临床思维链（Chain of Thought）进行结构化分析，
基于提供的参考资料，按以下逻辑顺序逐步推导并回答问题：

1. **核心问题识别**：明确用户所问的中医病症、证候或调理方法。
2. **中医病机**：简述中医理论中的病因病机基础。
3. **辨证分型**：列出相关的中医证候类型及其特点。
4. **治法方药**：
   - 推荐治疗方法（如中药方剂、针灸、推拿等）
   - 治疗原则和用药思路
5. **生活调护**：饮食宜忌、起居调养等中医养生建议。
6. **知识边界说明**：若参考资料不足以覆盖上述任一环节，请明确指出'现有资料未提及XX部分'。

要求：
- 使用规范中医术语，但对关键概念（如'肝郁脾虚'、'阴虚火旺'）需简要解释；
- 逻辑清晰，分点陈述，避免冗长段落；
- 结尾提出一个值得深入探讨的中医临床问题"""

    def create_agent(self):
        """
        创建中医RAG Agent
        """
        from langgraph.graph import StateGraph, END
        from langchain_core.tools import BaseTool
        from langchain_core.pydantic_v1 import Field
        
        # 工具 - 使用LLM判断是否需要深度解析
        class DeepModeClassifier(BaseTool):
            name: str = Field(default="tcm_deep_mode_classifier")
            description: str = Field(default="使用LLM判断中医问题是否需要深度思考模式")
            llm: Any = Field(exclude=True)

            def __init__(self, llm, **data):
                super().__init__(llm=llm, **data)

            def _run(self, question: str) -> bool:
                classification_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "请判断用户的问题是否需要中医深度解析模式。\n"
                     "如果问题涉及以下情况之一，请返回True：\n"
                     "- 需要详细的中医病机解释\n"
                     "- 需要辨证分型\n"
                     "- 需要完整的中医治疗方案\n"
                     "- 需要深入的中医临床分析\n"
                     "- 需要多个中医步骤的推理过程\n"
                     "如果问题只是简单的中医信息查询或常识性问题，请返回False。\n"
                     "只返回True或False，不要其他内容。"
                    ),
                    ("human", f"用户问题：{question}")
                ])
                
                classification_chain = classification_prompt | self.llm | StrOutputParser()
                response = classification_chain.invoke({})
                
                return response.strip().lower() in ['true', 'yes', '是', '需要']

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def determine_agent_mode(state: TcmRagAgentState) -> Dict[str, Any]:
            """确定代理模式 - 只判断是否需要深度解析"""
            deep_classifier = DeepModeClassifier(llm=self.llm)
            use_deep = deep_classifier._run(state.user_input)
            
            agent_mode = "deep" if use_deep else "basic"
            
            return {
                "agent_mode": agent_mode,
                "use_deep": use_deep
            }

        def retrieve_context(state: TcmRagAgentState) -> Dict[str, Any]:
            """检索上下文"""
            retrieved_docs = self.retriever.invoke(state.user_input)
            context = format_docs(retrieved_docs)
            
            return {
                "context": context,
                "source_documents": retrieved_docs
            }

        def generate_response(state: TcmRagAgentState) -> Dict[str, Any]:
            """根据模式生成响应，为每句话添加来源索引"""
            input_data = {"context": state.context, "question": state.user_input}
            
            if state.agent_mode == "deep":
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", self.deep_prompt),
                    ("human", "参考资料：\n{context}\n\n问题：{question}")
                ])
            else:  # basic
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", self.basic_prompt),
                    ("human", "参考资料：\n{context}\n\n用户问题：{question}")
                ])
            
            chain = prompt_template | self.llm | StrOutputParser()
            raw_result = chain.invoke(input_data)
            
            # 为每句话添加来源索引
            result_with_sources = self._add_source_indices(raw_result, state.source_documents)
            
            return {"result": result_with_sources}

        # 构建LangGraph
        workflow = StateGraph(TcmRagAgentState)
        
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

    def _add_source_indices(self, text: str, source_documents: List[Document]) -> str:
        """
        为生成的文本每句话添加来源索引
        """
        if not source_documents:
            return text
            
        # 将文本按句子分割
        sentences = []
        current_sentence = ""
        
        # 按标点符号分割句子
        for char in text:
            current_sentence += char
            if char in ['。', '！', '？', '\n']:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # 如果最后还有未处理的句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 为每个句子找到最相关的文档来源
        indexed_sentences = []
        for sentence in sentences:
            # 简单的关键词匹配来确定来源
            best_match_idx = 0
            max_matches = 0
            
            for idx, doc in enumerate(source_documents):
                # 计算句子中的关键词在文档中出现的次数
                doc_content = doc.page_content.lower()
                sentence_words = sentence.lower().split()
                
                matches = sum(1 for word in sentence_words if len(word) > 2 and word in doc_content)
                
                if matches > max_matches:
                    max_matches = matches
                    best_match_idx = idx
            
            # 添加来源索引标记
            indexed_sentence = f"{sentence}[来源: 文档{best_match_idx + 1}]"
            indexed_sentences.append(indexed_sentence)
        
        return " ".join(indexed_sentences)

    def query(self, user_query: str) -> Dict[str, Any]:
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