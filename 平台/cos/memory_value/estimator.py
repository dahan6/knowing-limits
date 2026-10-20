"""Memory value estimator: decide what to remember and how important it is."""
import re
from typing import Dict, List, Optional

import jieba.posseg as pseg

from ..config import COSConfig as config


class MemoryValueEstimator:
    """Estimate confidence, weight and topics for a memory snippet."""

    # Source type → base confidence & weight
    SOURCE_PROFILES = {
        "user_added": {"confidence": 0.95, "weight": 1.2},
        "llm_moe_verified": {"confidence": 0.90, "weight": 1.0},
        "file": {"confidence": 0.85, "weight": 1.0},
        "web_search": {"confidence": 0.60, "weight": 0.6},
        "unknown": {"confidence": 0.50, "weight": 0.5},
    }

    @classmethod
    def estimate(
        cls,
        content: str,
        query: str = "",
        source_type: str = "unknown",
        verified: bool = False,
        verification_score: float = 0.0,
    ) -> Dict:
        """
        Estimate memory metadata.
        Returns dict with: confidence, weight, topics, verified, source_type.
        """
        profile = cls.SOURCE_PROFILES.get(source_type, cls.SOURCE_PROFILES["unknown"])
        confidence = profile["confidence"]
        weight = profile["weight"]

        if verified:
            confidence = min(1.0, confidence + 0.05)
            weight = min(2.0, weight + 0.1)

        # Web search with strong verification can be upgraded
        if source_type == "web_search" and verification_score > 0.7:
            confidence = max(confidence, 0.75)
            weight = max(weight, 0.8)

        # Penalize very short or vague content
        content = content.strip()
        if len(content) < 30:
            confidence *= 0.9
            weight *= 0.9

        topics = cls.extract_topics(content) or cls.extract_topics(query)

        return {
            "source_type": source_type,
            "verified": bool(verified),
            "confidence": round(confidence, 3),
            "weight": round(weight, 3),
            "topics": topics,
        }

    @staticmethod
    def extract_topics(text: str, top_n: int = 5) -> List[str]:
        """Extract noun topics from text using jieba."""
        if not text:
            return []
        try:
            nouns = {}
            for word, flag in pseg.cut(text):
                w = word.strip()
                if len(w) < 2:
                    continue
                if flag.startswith("n"):
                    nouns[w] = nouns.get(w, 0) + 1
            sorted_nouns = sorted(nouns.items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in sorted_nouns[:top_n]]
        except Exception:
            return []

    @classmethod
    def should_remember(
        cls,
        content: str,
        query: str = "",
        source_type: str = "unknown",
        verified: bool = False,
    ) -> bool:
        """Quick filter: decide if a snippet is worth storing."""
        if not content or len(content.strip()) < 10:
            return False

        # Do not store pure chitchat or self-referential noise
        content_lower = content.lower()
        noise_phrases = [
            "我不知道", "无法回答", "不能确定", "没有相关信息",
            "i don't know", "i cannot", "not sure", "无法确定",
        ]
        if any(p in content_lower for p in noise_phrases):
            return False

        # High-confidence sources always remember
        profile = cls.SOURCE_PROFILES.get(source_type, cls.SOURCE_PROFILES["unknown"])
        if verified and profile["confidence"] >= 0.8:
            return True

        # Unverified web search needs some substance
        if source_type == "web_search" and len(content.strip()) < 30:
            return False

        return True

    @staticmethod
    def merge_topics(existing: List[str], new_topics: List[str]) -> List[str]:
        """Merge topic lists, keeping order and uniqueness."""
        seen = set()
        merged = []
        for t in list(existing) + list(new_topics):
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
        return merged[:10]
