"""LLM 知识检测封装 + 开放问题启发式判断"""
import sys
from pathlib import Path
from typing import Dict

import config

# 把检测器实验脚本所在目录加入路径
_EXPERIMENTS_DIR = Path.home() / "ai-training" / "glm-4.5-air" / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from predict_llm_detector import predict as _predict


class LLMClassifier:
    def __init__(self, model_path: Path = config.LLM_DETECTOR_MODEL):
        self.model_path = model_path

    def detect(self, query: str) -> Dict:
        """调用 Hybrid + Expanded 检测器，返回 known/not_known 结果。"""
        try:
            result = _predict(query, model_path=str(self.model_path))
            return result
        except Exception as e:
            # Fallback: if detector fails (e.g., embeddings endpoint error),
            # return a neutral result so the controller can still route.
            print(f"[WARN] LLM 知识检测器调用失败: {e}; 使用中性回退")
            return {
                "query": query,
                "label": "unknown",
                "confidence": 0.5,
                "proba": {"not_known": 0.5, "known": 0.5},
                "boost": 0.0,
                "short_answers": [],
                "expanded_answers": [],
                "features": {},
            }

    @staticmethod
    def is_open_question(query: str) -> bool:
        """基于关键词判断是否为开放/主观/建议性问题。"""
        q = query.lower()
        return any(k in q for k in config.OPEN_KEYWORDS)
