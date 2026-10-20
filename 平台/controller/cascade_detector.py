"""级联知识检测器：多次采样一致性为主，原 SVM 分类器为兜底。

学术参照：SelfCheckGPT / Semantic Entropy / EigenScore
核心思想：模型如果真正知道某个事实，多次采样的答案应该高度一致；
如果是幻觉，采样会发散。用一致性代替单次分类，更稳定。
"""
import statistics
import time
from typing import Dict, List

import jieba

import config
from classifier import LLMClassifier


def _tokenize(text: str) -> set:
    """简单分词：jieba + 英文数字。"""
    text = text.lower()
    tokens = set(jieba.lcut(text))
    tokens = {t.strip() for t in tokens if len(t.strip()) > 0}
    return tokens


def _answer_similarity(a: str, b: str) -> float:
    """计算两个答案的 Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) if union else 0.0


def _mean_pairwise_similarity(answers: List[str]) -> float:
    """计算答案集合的平均两两相似度。"""
    if len(answers) < 2:
        return 0.0
    sims = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            sims.append(_answer_similarity(answers[i], answers[j]))
    return statistics.mean(sims) if sims else 0.0


class CascadeDetector:
    """三级级联检测器。"""

    def __init__(self):
        self.fallback_detector = LLMClassifier()

    def _is_open_question(self, query: str) -> bool:
        """判断是否为开放/主观问题。"""
        q = query.lower()
        return any(k in q for k in config.OPEN_KEYWORDS)

    def _multi_sample_consistency(self, query: str, llm_client, n_samples: int = 3) -> Dict:
        """
        多次采样并计算一致性。
        返回 known_prob、一致性分数、采样答案列表。
        """
        answers = []
        temps = [0.3, 0.5, 0.7]
        for i in range(n_samples):
            temp = temps[i % len(temps)]
            try:
                messages = [
                    {"role": "system", "content": "请用一句话直接回答这个问题，不要解释。"},
                    {"role": "user", "content": query},
                ]
                text = llm_client._chat(messages, temperature=temp, max_tokens=128)
                text = text.strip()
                if text:
                    answers.append(text)
            except Exception:
                continue
            time.sleep(0.05)

        if not answers:
            return {"known_prob": 0.5, "consistency": 0.0, "answers": []}

        consistency = _mean_pairwise_similarity(answers)

        # 映射一致性到 known 概率
        # 高一致性（>0.6）→ 高 known；低一致性（<0.2）→ 低 known
        if consistency >= 0.6:
            known_prob = 0.8 + (consistency - 0.6) * 0.5
        elif consistency >= 0.4:
            known_prob = 0.5 + (consistency - 0.4) * 1.5
        else:
            known_prob = 0.2 + consistency * 1.5

        return {
            "known_prob": max(0.0, min(1.0, known_prob)),
            "consistency": consistency,
            "answers": answers,
        }

    def detect(self, query: str, llm_client=None) -> Dict:
        """
        级联检测入口。
        返回与 LLMClassifier.detect 兼容的格式。
        """
        # 第一级：开放问题快速判定
        if self._is_open_question(query):
            return {
                "query": query,
                "label": "not_known",
                "confidence": 0.8,
                "proba": {"not_known": 0.8, "known": 0.2},
                "boost": 0.0,
                "short_answers": [],
                "expanded_answers": [],
                "features": {"cascade_level": 1, "reason": "open_question"},
            }

        # 第二级：多次采样一致性
        if llm_client is not None:
            try:
                multi = self._multi_sample_consistency(query, llm_client, n_samples=3)
                known_prob = multi["known_prob"]
                label = "known" if known_prob >= 0.5 else "not_known"
                return {
                    "query": query,
                    "label": label,
                    "confidence": known_prob if label == "known" else 1 - known_prob,
                    "proba": {"not_known": 1 - known_prob, "known": known_prob},
                    "boost": 0.0,
                    "short_answers": multi["answers"],
                    "expanded_answers": [],
                    "features": {
                        "cascade_level": 2,
                        "consistency": multi["consistency"],
                        "n_samples": len(multi["answers"]),
                    },
                }
            except Exception as e:
                print(f"[WARN] 多次采样一致性检测失败: {e}; 回退到 SVM 检测器")

        # 第三级：原 SVM 检测器兜底
        return self.fallback_detector.detect(query)


# 单例
_cascade_detector = None


def get_cascade_detector() -> CascadeDetector:
    global _cascade_detector
    if _cascade_detector is None:
        _cascade_detector = CascadeDetector()
    return _cascade_detector
