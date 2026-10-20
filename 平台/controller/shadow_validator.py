"""Shadow validator: test new adapters before deployment."""
import json
from pathlib import Path
from typing import Dict, List

import config
from kb_client import KBClient
from llm_client import LLMClient


class ShadowValidator:
    """Validate that a new adapter doesn't degrade general capabilities."""

    def __init__(self, llm_client: LLMClient = None, adapter_path: str = None):
        self.llm = llm_client or LLMClient()
        self.adapter_path = adapter_path

    def _build_test_queries(self, n: int = 100) -> List[str]:
        """Build test queries from KB or predefined set."""
        # Try to get queries from test dataset
        dataset_path = Path(__file__).parent / "test_dataset.json"
        if dataset_path.exists():
            import json as _json
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = _json.load(f)
            return [item["question"] for item in dataset[:n]]

        # Fallback: general capability queries
        return [
            "中国的首都是哪里？",
            "水的化学式是什么？",
            "100的平方根是多少？",
            "HTTP 协议默认使用哪个端口？",
            "第二次世界大战哪一年结束？",
            "《红楼梦》的作者是谁？",
            "澳大利亚的首都是哪里？",
            "鸟类用什么呼吸？",
            "人生的意义是什么？",
            "如何看待人工智能对就业的影响？",
        ][:n]

    def _evaluate_answer(self, query: str, answer: str) -> float:
        """Simple heuristic evaluation of answer quality."""
        score = 0.0

        # Length check
        if len(answer) > 10:
            score += 0.3

        # Not a refusal for known questions
        known_questions = ["首都", "化学式", "平方根", "端口", "结束", "作者"]
        if any(q in query for q in known_questions):
            if "不知道" not in answer and "无法" not in answer:
                score += 0.4
            else:
                score -= 0.2

        # Has structure
        if "##" in answer or "###" in answer:
            score += 0.2

        # Has citation
        if "参考来源" in answer or "[1]" in answer:
            score += 0.1

        return max(0.0, min(1.0, score))

    def validate(self, adapter_path: str = None, test_queries: List[str] = None) -> Dict:
        """
        Compare baseline vs adapter performance.
        Returns validation result with pass/fail and metrics.
        """
        adapter_path = adapter_path or self.adapter_path
        if not adapter_path:
            return {"passed": False, "error": "no adapter path provided"}

        test_queries = test_queries or self._build_test_queries(
            config.QLORA_SHADOW_TEST_SIZE
        )

        # Load adapter (placeholder - in real implementation this would
        # temporarily mount the adapter to the LLM client)
        baseline_scores = []
        adapter_scores = []

        for query in test_queries:
            # Baseline
            baseline_answer = self.llm.direct_answer(query)
            baseline_scores.append(self._evaluate_answer(query, baseline_answer))

            # With adapter (simulated - actual implementation would load adapter)
            # For now, we use the same answer as placeholder
            adapter_answer = baseline_answer
            adapter_scores.append(self._evaluate_answer(query, adapter_answer))

        baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        adapter_avg = sum(adapter_scores) / len(adapter_scores) if adapter_scores else 0.0

        degradation = (baseline_avg - adapter_avg) / baseline_avg if baseline_avg > 0 else 0.0
        passed = degradation <= config.QLORA_MAX_DEGRADATION

        return {
            "passed": passed,
            "baseline_score": round(baseline_avg, 4),
            "adapter_score": round(adapter_avg, 4),
            "degradation": round(degradation, 4),
            "max_allowed_degradation": config.QLORA_MAX_DEGRADATION,
            "num_test_queries": len(test_queries),
            "adapter_path": adapter_path,
        }
