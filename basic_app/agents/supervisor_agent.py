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
        self.advice_memory = []

        
    def setup_agent(self):
        """
        设置supervisor agent的提示模板和相关配置
        """
        self.supervision_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一个资深的中西医结合专家，你在和一个中医和西医专家一起监督和指导问诊过程。
            你的职责是：
            1. 结合中医和西医专家的问诊结果，分析当前的问诊对话记录中的医生提问是否合理，
            不需要质疑中医和西医专家的回答
            2. 识别可能的诊断疏漏或错误
            3. 在必要时提供专业建议
            4. 评估问诊质量并决定是否需要干预
            5. 注意中西医专家的建议被监督者是看不到的，需要你根据知识总结并转述
            """),
            ("human", """
            当前问诊对话记录：
            {conversation_history}
            
            当前中医专家给出的参考:
             {tcm_response}
            当前西医专家给出的参考:
             {west_response}
            请评估当前问诊过程并决定是否需要给出建议。
            如果需要建议，请提供具体的专业建议。
            如果不需要建议，请只返回"无建议"。
            
            注意：只有在发现明显问题或重要信息遗漏时才提供建议，
            避免过度干预正常的问诊流程。
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
            5. 总结并给出医生的问诊思路,画一个简单的带箭头的图
            6. 从以下几个角度对于问诊的流程和结果的各个方面打分，相对严格,满分5分：
                问诊思路
                问诊流程
                信息收集
                细节询问
                中医结合
                总体评分
            """),
            ("human", """
            完整问诊对话记录：
            {full_conversation}
            
            请给出分析和建议：
            """)
        ])
    
    
    def create_agent(self):
        """
        创建supervisor agent
        """
        return self.supervision_prompt | self.llm | StrOutputParser()
    
    def evaluate_conversation(self, conversation_history: str,tcm_response: str, west_response: str) -> Dict[str, Any]:
        """
        评估对话并决定是否提供建议
        """
        print(type(conversation_history))
        agent = self.create_agent()
        advice = agent.invoke({
            "conversation_history": conversation_history,
            "tcm_response":tcm_response,
            "west_response":west_response

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
    
    def analyze_diagnosis_process(self, conversation_history) -> str:
        """
        分析整个诊断过程并给出评价
        """
        full_conversation = "\n".join(conversation_history)
        
        chain = self.analysis_prompt | self.llm | StrOutputParser()
        analysis = chain.invoke({
            "full_conversation": full_conversation
        })
        
        return analysis