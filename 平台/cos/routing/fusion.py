"""融合生成器：同时考虑 KB 和 LLM 证据的鲁棒生成"""
from typing import Dict, List

from ..providers.base import LLMProvider


class FusionGenerator:
    """融合 KB 检索结果和 LLM 内部知识的生成器。"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def answer(
        self,
        query: str,
        kb_chunks: List[Dict],
        kb_score: float,
        llm_direct_answer: str,
        llm_known_proba: float,
        case: str = "",
    ) -> str:
        """
        融合生成：根据证据强度给出有效回答。
        即使路由分类有偏差，也能根据证据强度给出有效回答。
        """
        return self.llm.fusion_answer(
            query=query,
            kb_chunks=kb_chunks,
            kb_score=kb_score,
            llm_direct_answer=llm_direct_answer,
            llm_known_proba=llm_known_proba,
            case=case,
        )
