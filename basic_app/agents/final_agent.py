from typing import Dict, Any, List
from langchain_core.language_models import BaseLanguageModel
from .base_agent import BaseAgent
from .west_agent import WestAgent
from .tcm_agent import TcmAgent
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
    Final Agent负责模拟医生角色，
    仅由supervisor_agent调用，不直接调用其他agent
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
            你是一个资深的中西医结合医生，正在为患者进行问诊。
            请遵循以下原则：
            1. 先根据患者主诉分析可能的疾病方向
            2. 提出针对性的问题来进一步确认诊断
            3. 结合西医的病理机制分析和中医的辨证论治思想
            4. 保持专业、耐心、关怀的态度
            5. 每次只问1-2个关键问题，避免问题过多
            6. 当信息充分时，给出初步诊断和建议
            """),
            ("human", """
            患者主诉：{patient_input}
            
            supervisor的建议（如果有的话）：{supervisor_advice}
            
            之前的对话历史：
            {conversation_history}
            
            请根据以上信息，继续问诊或给出诊断建议：
            """)
        ])
        


    def create_agent(self):
        """
        创建agent的实现方法（这里只是满足BaseAgent的抽象方法要求）
        """
        return self.diagnosis_prompt | self.llm | StrOutputParser()
    
    def integrate_responses(self, patient_input: str, supervisor_advice: str = None) -> str:
        """
        根据supervisor的建议，形成问诊策略
        """
        chain = self.diagnosis_prompt | self.llm | StrOutputParser()
        
        history_str = "\n".join(self.conversation_history)
        
        response = chain.invoke({
            "patient_input": patient_input,
            "supervisor_advice": supervisor_advice if supervisor_advice else "无建议",
            "conversation_history": history_str
        })
        
        return response
    
    def process_input(self, patient_input: str, supervisor_advice: str = None) -> Dict[str, Any]:
        """
        处理输入并返回问诊响应
        """
        # 根据supervisor的建议生成响应
        response = self.integrate_responses(patient_input, supervisor_advice)
        
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