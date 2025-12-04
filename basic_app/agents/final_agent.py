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
    Final Agent负责整合西医和中医的分析结果，
    并进行多轮对话问诊
    """
    
    def __init__(self, llm: BaseLanguageModel, west_agent: WestAgent, tcm_agent: TcmAgent, **kwargs):
        super().__init__(llm, **kwargs)
        self.west_agent = west_agent
        self.tcm_agent = tcm_agent
        self.memory = ConversationMemory()
        self.conversation_history = []
        self.setup_agent()
    
    def should_call_west_agent(self, patient_input: str, conversation_history: List[str] = None) -> bool:
        """
        决定是否调用西医agent
        """
        if conversation_history is None:
            conversation_history = self.conversation_history
            
        history_str = "\n".join(conversation_history)
        
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个决策助手，负责判断当前问题是否需要调用西医知识库。
            返回True表示需要调用西医agent，False表示不需要。
            如果问题是关于症状、疾病、治疗方案等，通常需要调用西医agent。
            如果问题主要涉及中医理论、中药、经络等，可能不需要调用西医agent。
            只返回True或False，不要其他内容。"""),
            ("human", f"""患者输入：{patient_input}

对话历史：
{history_str}

是否需要调用西医agent？""")
        ])
        
        chain = decision_prompt | self.llm | StrOutputParser()
        response = chain.invoke({})
        
        return response.strip().lower() in ['true', 'yes', '是', '需要']
    
    def should_call_tcm_agent(self, patient_input: str, conversation_history: List[str] = None) -> bool:
        """
        决定是否调用中医agent
        """
        if conversation_history is None:
            conversation_history = self.conversation_history
            
        history_str = "\n".join(conversation_history)
        
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个决策助手，负责判断当前问题是否需要调用中医知识库。
            返回True表示需要调用中医agent，False表示不需要。
            如果问题涉及中医理论、中药、辨证论治、体质等，通常需要调用中医agent。
            如果问题纯粹是西医范畴，可能不需要调用中医agent。
            只返回True或False，不要其他内容。"""),
            ("human", f"""患者输入：{patient_input}

对话历史：
{history_str}

是否需要调用中医agent？""")
        ])
        
        chain = decision_prompt | self.llm | StrOutputParser()
        response = chain.invoke({})
        
        return response.strip().lower() in ['true', 'yes', '是', '需要']
    
    def setup_agent(self):
        """
        设置final agent的提示模板和相关配置
        """
        self.diagnosis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个资深的中西医结合医生，正在为患者进行问诊。
            你将结合西医和中医的分析结果，对患者进行多轮问诊。
            请遵循以下原则：
            1. 先根据现有信息分析可能的疾病方向
            2. 提出针对性的问题来进一步确认诊断
            3. 结合西医的病理机制分析和中医的辨证论治思想
            4. 保持专业、耐心、关怀的态度
            5. 每次只问1-2个关键问题，避免问题过多
            6. 当信息充分时，给出初步诊断和建议
            """),
            ("human", """
            患者主诉：{patient_input}
            
            西医分析：{west_response}
            中医分析：{tcm_response}
            
            之前的对话历史：
            {conversation_history}
            
            请根据以上信息，继续问诊或给出诊断建议：
            """)
        ])
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个中西医结合专家，负责分析整个问诊过程。
            请从以下角度分析：
            1. 问诊思路是否清晰合理
            2. 问诊流程是否完整
            3. 是否有重要信息遗漏
            4. 给出改进建议
            5. 对整个问诊过程进行打分（1-10分）
            """),
            ("human", """
            完整问诊对话记录：
            {full_conversation}
            
            请给出分析和建议：
            """)
        ])

    def create_agent(self):
        """
        创建agent的实现方法（这里只是满足BaseAgent的抽象方法要求）
        """
        return self.diagnosis_prompt | self.llm | StrOutputParser()
    
    def integrate_responses(self, patient_input: str, west_response: str, tcm_response: str) -> str:
        """
        整合西医和中医的响应，形成初步的问诊策略
        """
        chain = self.diagnosis_prompt | self.llm | StrOutputParser()
        
        history_str = "\n".join(self.conversation_history)
        
        response = chain.invoke({
            "patient_input": patient_input,
            "west_response": west_response,
            "tcm_response": tcm_response,
            "conversation_history": history_str
        })
        
        return response
    
    def process_input(self, patient_input: str, west_response: str, tcm_response: str) -> Dict[str, Any]:
        """
        处理输入并返回问诊响应
        """
        # 整合两个agent的响应
        response = self.integrate_responses(patient_input, west_response, tcm_response)
        
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
    
    def analyze_diagnosis_process(self) -> str:
        """
        分析整个诊断过程并给出评价
        """
        full_conversation = "\n".join(self.conversation_history)
        
        chain = self.analysis_prompt | self.llm | StrOutputParser()
        analysis = chain.invoke({
            "full_conversation": full_conversation
        })
        
        return analysis
    
    def reset_conversation(self):
        """
        重置对话历史
        """
        self.conversation_history = []