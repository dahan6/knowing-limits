"""Training sample manager: load, clean, deduplicate, cluster and export samples."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import jieba
from datasketch import MinHash, MinHashLSH

import config


class TrainingSampleManager:
    """Manage training samples from learning_log.jsonl."""

    def __init__(self, log_path: str = None):
        self.log_path = Path(log_path or "learning_log.jsonl")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_samples(self, source: str = None, min_quality: float = 0.0) -> List[Dict]:
        """Load samples from learning_log.jsonl, optionally filtered by source."""
        if not self.log_path.exists():
            return []

        samples = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if source and entry.get("source") != source:
                        continue
                    samples.append(entry)
                except json.JSONDecodeError:
                    continue

        if min_quality > 0:
            samples = [s for s in samples if self.quality_score(s) >= min_quality]

        return samples

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean_sample(self, sample: Dict) -> Optional[Dict]:
        """Clean a single sample. Return None if it fails quality checks."""
        content = sample.get("content", "").strip()
        query = sample.get("query", "").strip()

        # 1. Length filtering
        if len(content) < 20:
            return None
        if len(content) > 5000:
            content = content[:5000]

        # 2. Quality filtering
        if content.startswith("[调用大模型失败"):
            return None
        if "我不知道" in content and len(content) < 50:
            return None
        # Require citation structure for reasoning samples
        if sample.get("source") in ("case_B_shadow", "llm_moe_verified"):
            if "参考来源" not in content and "##" not in content:
                return None

        # 3. Format standardization
        return {
            "instruction": query,
            "input": "",
            "output": content,
            "source": sample.get("source", "unknown"),
            "source_type": sample.get("source_type", "unknown"),
            "timestamp": sample.get("timestamp", ""),
            "topics": sample.get("topics", []),
        }

    def clean_samples(self, samples: List[Dict]) -> List[Dict]:
        """Clean a batch of samples."""
        cleaned = []
        for s in samples:
            c = self.clean_sample(s)
            if c is not None:
                cleaned.append(c)
        return cleaned

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    @staticmethod
    def _exact_dedup(samples: List[Dict]) -> List[Dict]:
        """Remove exact duplicates based on instruction+output hash."""
        seen = set()
        unique = []
        for s in samples:
            key = hashlib.md5(
                (s["instruction"] + s["output"]).encode("utf-8")
            ).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    @staticmethod
    def _near_dedup(samples: List[Dict], threshold: float = 0.9) -> List[Dict]:
        """Remove near-duplicates using MinHash LSH."""
        if not samples:
            return []

        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        unique = []
        for i, s in enumerate(samples):
            m = MinHash()
            text = s["instruction"] + " " + s["output"]
            for word in jieba.lcut(text):
                m.update(word.encode("utf-8"))
            result = lsh.query(m)
            if not result:
                lsh.insert(f"sample_{i}", m)
                unique.append(s)
        return unique

    @staticmethod
    def _diversity_check(samples: List[Dict], max_per_topic: int = 50) -> List[Dict]:
        """Ensure topic diversity, keep at most max_per_topic per topic."""
        topic_groups = defaultdict(list)
        for s in samples:
            topics = s.get("topics", ["general"])
            topic = topics[0] if topics else "general"
            topic_groups[topic].append(s)

        diverse = []
        for topic, group in topic_groups.items():
            group.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            diverse.extend(group[:max_per_topic])
        return diverse

    def deduplicate_samples(self, samples: List[Dict]) -> List[Dict]:
        """Three-level deduplication: exact → near → diversity."""
        samples = self._exact_dedup(samples)
        samples = self._near_dedup(samples, threshold=0.9)
        samples = self._diversity_check(samples, max_per_topic=50)
        return samples

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------
    @staticmethod
    def quality_score(sample: Dict) -> float:
        """Score sample quality 0-1."""
        score = 0.0
        content = sample.get("output", sample.get("content", ""))

        # 1. Structure completeness (30%)
        if "## 推理过程" in content or "### 分析" in content:
            score += 0.15
        if "参考来源" in content:
            score += 0.15

        # 2. Information density (30%)
        words = jieba.lcut(content)
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += 0.3 * min(unique_ratio * 2, 1.0)

        # 3. Source trust (40%)
        source_type = sample.get("source_type", "unknown")
        trust_map = {
            "user_added": 1.0,
            "file": 0.9,
            "llm_moe_verified": 0.8,
            "case_B_shadow": 0.8,
            "web_search": 0.5,
            "unknown": 0.3,
        }
        score += 0.4 * trust_map.get(source_type, 0.3)

        return round(score, 3)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def cluster_by_topic(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Cluster samples by topic."""
        clusters = defaultdict(list)
        for s in samples:
            topics = s.get("topics", [])
            if not topics:
                # Fallback: extract from source or instruction
                topics = [s.get("source", "general")]
            topic = topics[0] if topics else "general"
            clusters[topic].append(s)
        return dict(clusters)

    # ------------------------------------------------------------------
    # Trigger condition
    # ------------------------------------------------------------------
    def check_trigger_condition(
        self,
        cluster: List[Dict],
        min_samples: int = None,
        min_repeat_query: int = 3,
    ) -> bool:
        """Check if a cluster meets training trigger conditions."""
        min_samples = min_samples or config.QLORA_MIN_SAMPLES

        # Condition 1: enough samples
        if len(cluster) >= min_samples:
            return True

        # Condition 2: same query repeated enough times
        query_counts = defaultdict(int)
        for s in cluster:
            query_counts[s["instruction"]] += 1
        if max(query_counts.values(), default=0) >= min_repeat_query:
            return True

        return False

    # ------------------------------------------------------------------
    # Export for training
    # ------------------------------------------------------------------
    def export_for_training(self, cluster: List[Dict]) -> List[Dict]:
        """Export samples in QLoRA training format (cleaned + deduplicated)."""
        cleaned = self.clean_samples(cluster)
        deduped = self.deduplicate_samples(cleaned)
        return deduped

    def get_training_clusters(
        self,
        source: str = None,
        min_quality: float = 0.5,
    ) -> Dict[str, List[Dict]]:
        """Load, clean, deduplicate, cluster, and filter by trigger condition."""
        samples = self.load_samples(source=source, min_quality=min_quality)
        clusters = self.cluster_by_topic(samples)
        ready = {}
        for topic, cluster in clusters.items():
            if self.check_trigger_condition(cluster):
                ready[topic] = self.export_for_training(cluster)
        return ready

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_sample(self, query: str, content: str, source: str = "unknown"):
        """Append a new sample to learning_log.jsonl."""
        entry = {
            "timestamp": self._now(),
            "query": query,
            "source": source,
            "content": content,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
