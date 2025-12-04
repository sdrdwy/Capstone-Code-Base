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
    
    def evaluate_conversation(self, conversation_history: str, west_agent=None, tcm_agent=None, patient_input: str = "") -> Dict[str, Any]:
        """
        评估对话并决定是否提供建议
        可以根据需要决定是否调用west_agent或tcm_agent
        """
        # 首先评估是否需要调用额外的agent
        if west_agent and tcm_agent and patient_input:
            call_west, call_tcm = self._should_call_agents(patient_input, conversation_history)
            
            additional_info = ""
            if call_west or call_tcm:
                additional_info = "\n\n基于当前对话，建议补充查询："
                if call_west:
                    additional_info += "\n- 西医知识库查询"
                if call_tcm:
                    additional_info += "\n- 中医知识库查询"
        else:
            additional_info = ""
        
        agent = self.create_agent()
        advice = agent.invoke({
            "conversation_history": conversation_history + additional_info
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
    
    def _should_call_agents(self, patient_input: str, conversation_history: str) -> tuple[bool, bool]:
        """
        决定supervisor是否应该调用west_agent或tcm_agent
        返回 (call_west: bool, call_tcm: bool)
        """
        decision_prompt = f"""
        作为中西医结合专家，请判断当前问诊过程中是否需要补充西医或中医知识查询。
        
        患者输入：{patient_input}
        对话历史：{conversation_history}
        
        请返回一个JSON格式的结果，包含以下字段：
        {{
          "call_west": true/false,
          "call_tcm": true/false
        }}
        
        如果当前对话缺少西医相关知识，返回call_west为true。
        如果当前对话缺少中医相关知识，返回call_tcm为true。
        """
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import json
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", decision_prompt)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({})
        
        try:
            result = json.loads(response)
            return result.get("call_west", False), result.get("call_tcm", False)
        except:
            return False, False
    
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