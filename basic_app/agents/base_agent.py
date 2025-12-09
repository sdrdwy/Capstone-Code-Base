from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel


class BaseAgent(ABC):
    """
    Agent的基础类，提供通用的功能和接口
    """
    
    def __init__(self, llm: BaseLanguageModel, **kwargs):
        self.llm = llm
        self.tools = kwargs.get('tools', [])
        self.prompt = kwargs.get('prompt', None)
        self.parser = kwargs.get('parser', None)
        
    @abstractmethod
    def create_agent(self) -> Runnable:
        """
        创建agent的抽象方法，子类必须实现
        """
        pass
    
    def add_tool(self, tool: BaseTool):
        """
        添加工具到agent
        """
        self.tools.append(tool)
        
    def set_prompt(self, prompt: ChatPromptTemplate):
        """
        设置提示模板
        """
        self.prompt = prompt
        
    def set_parser(self, parser: BaseOutputParser):
        """
        设置输出解析器
        """
        self.parser = parser
        
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行agent的标准方法
        """
        agent = self.create_agent()
        return agent.invoke(inputs)
        
    def stream(self, inputs: Dict[str, Any]):
        """
        流式运行agent
        """
        agent = self.create_agent()
        return agent.stream(inputs)