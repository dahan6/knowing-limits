"""LLM Provider 抽象接口"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """LLM 后端抽象接口，定义所有模式必须实现的方法。"""

    # ------------------------------------------------------------------
    # 基础能力（所有模式必须支持）
    # ------------------------------------------------------------------
    @abstractmethod
    def chat(self, messages: List[Dict], temperature: float = 0.6, max_tokens: int = 1024) -> str:
        """基础对话生成"""
        pass

    @abstractmethod
    def rag_answer(self, query: str, chunks: List[Dict]) -> str:
        """基于 KB 的 RAG 生成"""
        pass

    @abstractmethod
    def direct_answer(self, query: str, note: str = "") -> str:
        """直接回答"""
        pass

    @abstractmethod
    def rag_reasoning_answer(self, query: str, chunks: List[Dict]) -> str:
        """基于 KB 的逻辑推理生成"""
        pass

    @abstractmethod
    def fusion_answer(
        self,
        query: str,
        kb_chunks: List[Dict],
        kb_score: float,
        llm_direct_answer: str,
        llm_known_proba: float,
        case: str = "",
    ) -> str:
        """融合 KB 和 LLM 证据的鲁棒生成"""
        pass

    @abstractmethod
    def moe_reasoning(self, query: str, n_paths: int = 3) -> Dict:
        """MoE 多路推理"""
        pass

    @abstractmethod
    def verify_conclusion(self, query: str, conclusion: str, web_results: List[Dict]) -> Dict:
        """验证结论合理性"""
        pass

    @abstractmethod
    def unknown_response(self, query: str) -> str:
        """未知问题拒答"""
        pass

    @abstractmethod
    def learning_log(self, query: str, source: str, content: str, log_path: str = "learning_log.jsonl") -> None:
        """记录学习样本"""
        pass

    # ------------------------------------------------------------------
    # 可选能力（本地模式支持，远程模式降级）
    # ------------------------------------------------------------------
    @abstractmethod
    def supports_implicit_memory(self) -> bool:
        """是否支持隐式记忆（LoRA 微调）"""
        pass

    @abstractmethod
    def load_adapter(self, adapter_path: str) -> bool:
        """挂载 LoRA adapter（本地模式）"""
        pass

    @abstractmethod
    def unload_adapter(self) -> bool:
        """卸载 LoRA adapter"""
        pass

    @abstractmethod
    def train_adapter(self, samples: List[Dict], adapter_name: str, **kwargs) -> str:
        """训练新的 LoRA adapter（本地模式）"""
        pass
