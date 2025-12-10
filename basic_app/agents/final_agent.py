from typing import Dict, Any, List
from langchain_core.language_models import BaseLanguageModel
from .base_agent import BaseAgent
from ..memory.conversation_memory import ConversationMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel


class DiagnosisState(BaseModel):
    """
    诊断状态，用于跟踪问诊过程
    """
    patient_input: str
    west_response: str
    tcm_response: str
    current_diagnosis: str
    conversation_history: List[str]
    is_ended: bool
    next_question: str


class FinalAgent(BaseAgent):
    """
    Final Agent负责整合西医和中医的分析结果，
    并进行多轮对话问诊
    """
    
    def __init__(self, llm: BaseLanguageModel, **kwargs):
        super().__init__(llm, **kwargs)
        self.memory = ConversationMemory()
        self.conversation_history = []
        self.setup_agent()
    
    def setup_agent(self):
        """
        设置final agent的提示模板和相关配置
        """
        self.diagnosis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个新手的中西医结合医生，正在有导师指导的情况下为患者进行问诊。
            你将结合西医和中医的分析结果，对患者进行多轮问诊。
            请遵循以下原则：
            1. 先根据现有信息分析可能的疾病方向
            2. 提出针对性的问题来进一步确认诊断
            3. 结合西医的病理机制分析和中医的辨证论治思想
            4. 保持专业、耐心、关怀的态度
            5. 每次只问1-2个关键问题，避免问题过多
            6. 当信息充分时，给出初步诊断和建议
            7. 当有足够把握判断是什么疾病的时候，给出问诊结果并最后输出“结束”
            8. 你是一个有足够自信的医生，为了保证对话轮次不要太长，请尽早给出病因诊断并输出"结束"
            """),
            ("human", """
            患者主诉：{patient_input}
            
            之前的对话历史：
            {conversation_history}
            
            导师的建议：
             {supervisor_advice}
            请根据以上信息，继续问诊或给出诊断建议：
            """)
        ])
        
        

    def create_agent(self):
        """
        创建agent的实现方法（这里只是满足BaseAgent的抽象方法要求）
        """
        return self.diagnosis_prompt | self.llm | StrOutputParser()
    
    def integrate_responses(self, patient_input: str,advice) -> str:
        """
        整合西医和中医的响应，形成初步的问诊策略
        """
        chain = self.diagnosis_prompt | self.llm | StrOutputParser()
        
        history_str = "\n".join(self.conversation_history)
        
        response = chain.invoke({
            "patient_input": patient_input,
            "conversation_history": history_str,
            "supervisor_advice":advice
        })
        
        return response
    
    def process_input(self, patient_input: str,advice) -> Dict[str, Any]:
        """
        处理输入并返回问诊响应
        """
        # 整合两个外部传入的分析结果
        response = self.integrate_responses(patient_input,advice)
        
        # 更新对话历史
        self.conversation_history.append(f"患者: {patient_input}")
        self.conversation_history.append(f"医生: {response}")

        return {
            "response": response,
            "conversation_history": self.conversation_history.copy(),
            "is_ended": self._check_end_condition(response)
        }
    
    def _check_end_condition(self, response: str) -> bool:
        """
        检查是否满足结束条件
        """
        end_keywords = ["结束", "完成", "建议就医", "需要面诊", "诊断完成", "请到医院", "需要检查"]
        return any(keyword in response for keyword in end_keywords)
    
    
    def reset_conversation(self):
        """
        重置对话历史
        """
        self.conversation_history = []