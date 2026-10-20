"""四象限路由器：KB Score × LLM Known Prob → A/B/C/D"""
from ..config import COSConfig


class QuadrantRouter:
    """根据 KB 分数和 LLM 知识概率进行四象限分类。"""

    def __init__(self, config: COSConfig = None):
        self.config = config or COSConfig.from_env()

    def classify(self, kb_score: float, llm_known_proba: float) -> str:
        """
        四象限分类：
        A. KB高 + LLM高 → "A"
        B. KB高 + LLM低 → "B"
        C. KB低 + LLM高 → "C"
        D. KB低 + LLM低 → "D"
        """
        kb_high = kb_score >= self.config.kb_hit_high
        llm_high = llm_known_proba >= self.config.llm_known_high

        if kb_high and llm_high:
            return "A"
        if kb_high and not llm_high:
            return "B"
        if not kb_high and llm_high:
            return "C"
        return "D"

    def is_borderline(self, kb_score: float, llm_known_proba: float) -> bool:
        """判断是否处于边界区"""
        kb_borderline = abs(kb_score - self.config.kb_hit_high) < self.config.routing_borderline_kb
        llm_borderline = abs(llm_known_proba - self.config.llm_known_high) < self.config.routing_borderline_llm
        return kb_borderline or llm_borderline

    def need_fusion(self, kb_score: float, llm_known_proba: float, case: str) -> bool:
        """
        判断是否需要融合生成。
        触发条件：
        1. 分数落在阈值边界区
        2. KB 有中等命中但路由走了 C/D
        3. LLM 有一定自信但路由走了 B/D
        """
        if self.is_borderline(kb_score, llm_known_proba):
            return True

        kb_useful = kb_score >= self.config.kb_hit_low
        llm_useful = llm_known_proba >= self.config.llm_known_low

        if kb_useful and case in ("C", "D"):
            return True
        if llm_useful and case in ("B", "D"):
            return True
        return False
