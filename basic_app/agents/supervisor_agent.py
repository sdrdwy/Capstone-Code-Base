from typing import Dict, Any
from langchain_core.language_models import BaseLanguageModel
from .base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent负责监督整个问诊过程，
    检测final_agent和用户的对话，并作为专家给出建议
    """
    
    def __init__(self, llm, **kwargs):
        super().__init__(llm, **kwargs)
        self.setup_agent()
        self.supervision_count = 0  # 记录监督次数
        
    def setup_agent(self):
        """
        设置supervisor agent的提示模板和相关配置
        """
        self.supervision_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个资深的中西医结合专家，负责监督和指导问诊过程。
            你的职责是：
            1. 分析当前的问诊对话是否合理
            2. 识别可能的诊断疏漏或错误
            3. 在必要时提供专业建议
            4. 评估问诊质量并决定是否需要干预
            """),
            ("human", """
            当前问诊对话记录：
            {conversation_history}
            
            请评估当前问诊过程并决定是否需要给出建议。
            如果需要建议，请提供具体的专业建议。
            如果不需要建议，请只返回"无建议"。
            
            注意：只有在发现明显问题或重要信息遗漏时才提供建议，
            避免过度干预正常的问诊流程。
            """)
        ])
    
    def create_agent(self):
        """
        创建supervisor agent
        """
        return self.supervision_prompt | self.llm | StrOutputParser()
    
    def evaluate_conversation(self, conversation_history: str) -> Dict[str, Any]:
        """
        评估对话并决定是否提供建议
        """
        agent = self.create_agent()
        advice = agent.invoke({
            "conversation_history": conversation_history
        })
        
        # 如果没有建议，返回空值
        if advice.strip() == "无建议":
            return {
                "should_advise": False,
                "advice": None,
                "evaluation": "问诊过程正常，无需干预"
            }
        
        # 检查是否应该提供建议（基于内容分析）
        should_advise = self._should_advise(advice, conversation_history)
        
        return {
            "should_advise": should_advise,
            "advice": advice if should_advise else None,
            "evaluation": advice if not should_advise else "问诊过程正常，无需干预"
        }
    
    def _should_advise(self, advice: str, conversation_history: str) -> bool:
        """
        决定是否应该提供建议
        """
        # 分析建议内容和对话历史，决定是否需要干预
        # 如果建议包含重要医疗提醒或关键信息，则提供建议
        important_keywords = [
            "注意", "警告", "重要", "关键", "必须", "应该", "需要", 
            "检查", "诊断", "鉴别", "严重", "危险", "紧急", "立即"
        ]
        
        advice_lower = advice.lower()
        contains_important = any(keyword in advice_lower for keyword in important_keywords)
        
        # 如果已经进行了多次监督，可以减少干预频率
        self.supervision_count += 1
        if self.supervision_count > 5 and not contains_important:
            # 如果监督次数过多且没有重要信息，降低干预概率
            return False
        
        return contains_important
    
    def provide_guidance(self, current_diagnosis: str, symptoms: str, conversation_history: str) -> str:
        """
        提供具体的指导建议
        """
        guidance_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个中西医结合专家，正在为问诊过程提供专业指导。
            请基于当前的诊断和症状信息，提供专业的建议。
            """),
            ("human", f"""
            当前诊断：{current_diagnosis}
            症状描述：{symptoms}
            问诊历史：{conversation_history}
            
            请提供专业的指导建议，包括：
            1. 问诊方向的建议
            2. 可能的鉴别诊断
            3. 需要注意的关键点
            4. 进一步检查的建议
            """)
        ])
        
        chain = guidance_prompt | self.llm | StrOutputParser()
        return chain.invoke({})
    
    def decide_agent_usage(self, patient_input: str, conversation_history: str) -> Dict[str, bool]:
        """
        决定是否需要调用west_agent或tcm_agent
        """
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个智能决策助手，负责决定在当前问诊情况下是否需要调用西医知识库(west_agent)或中医知识库(tcm_agent)。
            根据患者输入和对话历史，判断最合适的知识库组合。
            """),
            ("human", """
            患者输入：{patient_input}
            对话历史：{conversation_history}
            
            请决定是否需要调用：
            1. 西医知识库 (west_agent) - 对应 should_call_west
            2. 中医知识库 (tcm_agent) - 对应 should_call_tcm
            
            请以JSON格式返回，包含以下字段：
            {
                "should_call_west": true/false,
                "should_call_tcm": true/false
            }
            
            判断依据：
            - 如果问题涉及现代医学、病理机制、西药、实验室检查等，应调用西医知识库
            - 如果问题涉及中医理论、证型、中药、经络等，应调用中医知识库
            - 如果问题比较综合或需要中西医结合，可以同时调用两个知识库
            - 如果问题很基础或已有足够信息，可以不调用任何知识库
            """)
        ])
        
        chain = decision_prompt | self.llm | StrOutputParser()
        decision_result = chain.invoke({
            "patient_input": patient_input,
            "conversation_history": conversation_history
        })
        
        # 解析决策结果，这里简单处理，实际中可能需要更复杂的JSON解析
        should_call_west = "should_call_west\": true" in decision_result.lower() or "\"west\": true" in decision_result.lower()
        should_call_tcm = "should_call_tcm\": true" in decision_result.lower() or "\"tcm\": true" in decision_result.lower()
        
        return {
            "should_call_west": should_call_west,
            "should_call_tcm": should_call_tcm
        }
    
    def generate_summary(self, conversation_history: str) -> str:
        """
        在程序中断时自动生成问诊过程总结
        """
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个中西医结合专家，需要对当前的问诊过程进行总结。
            请提供以下内容：
            1. 问诊过程概述
            2. 已收集的主要症状和信息
            3. 初步的诊断方向或考虑
            4. 尚未明确的问题或需要进一步询问的内容
            5. 后续建议
            """),
            ("human", """
            问诊对话记录：
            {conversation_history}
            
            请生成问诊过程总结：
            """)
        ])
        
        chain = summary_prompt | self.llm | StrOutputParser()
        return chain.invoke({"conversation_history": conversation_history})