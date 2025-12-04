from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any


class ConversationMemory:
    """
    用于记忆短期对话内容的内存管理器
    使用langgraph中的MemorySaver实现
    """
    
    def __init__(self):
        self.memory = MemorySaver()
        
    def get_config(self, thread_id: str = "default"):
        """
        获取配置，用于langgraph的可记忆化执行
        """
        return {"configurable": {"thread_id": thread_id}}
        
    def save_state(self, thread_id: str, state: Dict[str, Any]):
        """
        保存指定线程的状态
        """
        config = self.get_config(thread_id)
        # 由于MemorySaver主要用于检查点，我们直接返回配置供使用
        return config
        
    def load_state(self, thread_id: str, checkpoint_id: str = None):
        """
        加载指定线程的状态
        """
        config = self.get_config(thread_id)
        return config
    
    def clear_memory(self, thread_id: str):
        """
        清除指定线程的记忆
        """
        # 在实际应用中，这会清除特定线程的检查点数据
        pass
        
    @property
    def memory_saver(self):
        """
        返回MemorySaver实例供langgraph使用
        """
        return self.memory