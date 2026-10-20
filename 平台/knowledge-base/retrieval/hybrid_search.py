"""Hybrid search: dense + BM25 + RRF + reranker."""
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import config
from .keyword_index import KeywordIndex
from .reranker import Reranker


def _score_to_rank_list(results: List[Dict], score_key: str = "score") -> List[Tuple[str, float]]:
    """Convert a list of result dicts into (id, rank_score) tuples."""
    return [(r["id"], float(r[score_key])) for r in results]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: float = 60.0,
) -> Dict[str, float]:
    """
    Fuse multiple ranked lists using RRF.
    Each list should be ordered from best to worst.
    Returns a mapping id -> fused score.
    """
    fused: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (doc_id, _) in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridSearcher:
    """Combine Chroma vector search, BM25 keyword search and ONNX reranker."""

    def __init__(self, kb_manager):
        self.kb_manager = kb_manager
        self._reranker: Optional[Reranker] = None
        self._keyword_indexes: Dict[str, KeywordIndex] = {}

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def _get_keyword_index(self, slug: str) -> KeywordIndex:
        if slug not in self._keyword_indexes:
            idx = KeywordIndex(slug)
            if not idx.load():
                self.rebuild_keyword_index(slug)
            self._keyword_indexes[slug] = idx
        return self._keyword_indexes[slug]

    def rebuild_keyword_index(self, slug: str) -> None:
        """Rebuild BM25 index from the current Chroma collection."""
        records = self.kb_manager.get_all_records(slug)
        idx = KeywordIndex(slug)
        idx.build(records)
        self._keyword_indexes[slug] = idx

    def add_to_keyword_index(self, slug: str, records: List[Dict]) -> None:
        """Incrementally add records to the BM25 index without full rebuild from DB."""
        idx = self._get_keyword_index(slug)
        idx.add_records(records)
        self._keyword_indexes[slug] = idx

    def update_keyword_index(self, slug: str, records: List[Dict]) -> None:
        """Incrementally update records in the BM25 index."""
        idx = self._get_keyword_index(slug)
        idx.update_records(records)
        self._keyword_indexes[slug] = idx

    def delete_from_keyword_index(self, slug: str, ids: List[str]) -> None:
        """Incrementally delete records from the BM25 index."""
        idx = self._get_keyword_index(slug)
        idx.delete_records(ids)
        self._keyword_indexes[slug] = idx

    def vector_search(self, slug: str, query: str, top_n: int = 30) -> List[Dict]:
        return self.kb_manager.vector_search(slug, query, top_n=top_n)

    def keyword_search(self, slug: str, query: str, top_n: int = 30) -> List[Dict]:
        idx = self._get_keyword_index(slug)
        return idx.search(query, top_n=top_n)

    @staticmethod
    def _compute_memory_score(metadata: Dict) -> float:
        """Compute a multiplicative boost based on memory metadata."""
        weight = float(metadata.get("weight", config.MEMORY_DEFAULT_WEIGHT))
        confidence = float(metadata.get("confidence", 0.5))
        access_count = int(metadata.get("access_count", 0))
        tier = metadata.get("tier", "warm")

        # Time decay based on last access
        last_accessed = metadata.get("last_accessed")
        time_decay = 1.0
        if last_accessed:
            try:
                last = datetime.fromisoformat(last_accessed)
                days = (datetime.now(timezone.utc) - last).days
                time_decay = math.exp(-config.MEMORY_TIME_DECAY_LAMBDA * max(days, 0))
            except Exception:
                pass

        # Frequency boost
        freq_boost = 1.0 + config.MEMORY_ACCESS_WEIGHT * math.log1p(access_count)

        # Confidence boost
        conf_boost = 1.0 + config.MEMORY_CONFIDENCE_WEIGHT * confidence

        # Tier factor
        tier_factor = {"hot": 1.2, "warm": 1.0, "cold": 0.7}.get(tier, 1.0)

        return weight * time_decay * freq_boost * conf_boost * tier_factor

    def search(
        self,
        slug: str,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        rerank_candidates: int = 60,
        apply_memory_scoring: bool = True,
    ) -> List[Dict]:
        """
        Run hybrid search for a single query and return top_k chunks.
        Each chunk has keys: id, content, source, score, metadata.
        """
        vector_results = self.vector_search(slug, query, top_n=config.VECTOR_CANDIDATES)

        if not use_hybrid:
            fused = _score_to_rank_list(vector_results)
        else:
            keyword_results = self.keyword_search(slug, query, top_n=config.KEYWORD_CANDIDATES)
            fused = reciprocal_rank_fusion(
                [_score_to_rank_list(vector_results), _score_to_rank_list(keyword_results)]
            )

        # Sort by fused score
        sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        needed_ids = [doc_id for doc_id, _ in sorted_ids[: max(top_k, rerank_candidates)]]

        # Fetch full records for candidate ids
        records = self.kb_manager.get_records_by_ids(slug, needed_ids)
        record_map = {r["id"]: r for r in records}

        candidates = []
        for doc_id, base_score in sorted_ids[:rerank_candidates]:
            rec = record_map.get(doc_id)
            if rec:
                candidates.append((doc_id, rec["text"], rec.get("metadata", {}), base_score))

        if use_rerank and candidates:
            try:
                reranker = self._get_reranker()
                rerank_inputs = [(pid, text, meta) for pid, text, meta, _ in candidates]
                ranked = reranker.rerank(query, rerank_inputs)
                results = []
                for pid, text, meta, rerank_score in ranked:
                    memory_boost = self._compute_memory_score(meta) if apply_memory_scoring else 1.0
                    final_score = rerank_score * memory_boost
                    results.append({
                        "id": pid,
                        "content": text,
                        "source": meta.get("source", "未知来源"),
                        "score": final_score,
                        "raw_score": rerank_score,
                        "metadata": meta,
                    })
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
            except Exception:
                # Fall back to fused scores if reranker fails
                pass

        # No rerank: use fused score order with memory scoring
        final = []
        for doc_id, score in sorted_ids[:top_k]:
            rec = record_map.get(doc_id)
            if rec:
                meta = rec.get("metadata", {})
                memory_boost = self._compute_memory_score(meta) if apply_memory_scoring else 1.0
                final.append(
                    {
                        "id": doc_id,
                        "content": rec["text"],
                        "source": meta.get("source", "未知来源"),
                        "score": score * memory_boost,
                        "raw_score": score,
                        "metadata": meta,
                    }
                )
        final.sort(key=lambda x: x["score"], reverse=True)
        return final
