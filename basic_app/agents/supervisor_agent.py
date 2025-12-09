from typing import Dict, Any
from langchain_core.language_models import BaseLanguageModel
from .base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio


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
    
    def should_call_west_agent(self, conversation_history: str) -> bool:
        """
        决定是否调用西医agent
        """
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个决策助手，负责判断当前问诊过程是否需要调用西医知识库来辅助诊断。
            返回True表示需要调用西医agent，False表示不需要。
            如果当前症状或问题可能需要西医诊断或治疗建议，返回True。
            只返回True或False，不要其他内容。"""),
            ("human", f"""当前问诊对话记录：
{conversation_history}

是否需要调用西医agent？""")
        ])
        
        chain = decision_prompt | self.llm | StrOutputParser()
        response = chain.invoke({})
        
        return response.strip().lower() in ['true', 'yes', '是', '需要']
    
    def should_call_tcm_agent(self, conversation_history: str) -> bool:
        """
        决定是否调用中医agent
        """
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个决策助手，负责判断当前问诊过程是否需要调用中医知识库来辅助诊断。
            返回True表示需要调用中医agent，False表示不需要。
            如果当前症状或问题可能需要中医辨证论治或中药建议，返回True。
            只返回True或False，不要其他内容。"""),
            ("human", f"""当前问诊对话记录：
{conversation_history}

是否需要调用中医agent？""")
        ])
        
        chain = decision_prompt | self.llm | StrOutputParser()
        response = chain.invoke({})
        
        return response.strip().lower() in ['true', 'yes', '是', '需要']
    
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

    def call_west_agent(self, query: str, graph_results: str = "") -> str:
        """
        调用西医agent获取诊断建议
        """
        if hasattr(self, 'west_agent') and self.west_agent:
            try:
                # 调用西医agent
                west_result = self.west_agent.query(query)
                return west_result.get('answer', '无结果')
            except Exception as e:
                print(f"调用西医agent出错: {str(e)}")
                return "西医agent暂时无法提供结果"
        else:
            return "西医agent未初始化"

    def call_tcm_agent(self, query: str, graph_results: str = "") -> str:
        """
        调用中医agent获取诊断建议
        """
        if hasattr(self, 'tcm_agent') and self.tcm_agent:
            try:
                # 调用中医agent
                tcm_result = self.tcm_agent.query(query)
                return tcm_result.get('result', '无结果')
            except Exception as e:
                print(f"调用中医agent出错: {str(e)}")
                return "中医agent暂时无法提供结果"
        else:
            return "中医agent未初始化"

    def call_tcm_rag_agent(self, query: str, rag_context: str = "") -> str:
        """
        调用中医RAG agent获取诊断建议
        """
        if hasattr(self, 'tcm_rag_agent') and self.tcm_rag_agent:
            try:
                # 调用中医RAG agent
                tcm_rag_result = self.tcm_rag_agent.query(query)
                return tcm_rag_result.get('answer', '无结果')
            except Exception as e:
                print(f"调用中医RAG agent出错: {str(e)}")
                return "中医RAG agent暂时无法提供结果"
        else:
            return "中医RAG agent未初始化"

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