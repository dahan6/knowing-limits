"""Context post-processing: deduplication, compression and citation formatting."""
from typing import Dict, List, Tuple

import config


def _token_count(text: str) -> int:
    """Fast token count approximation: ~4 chars per token for CJK/English mix."""
    return max(1, len(text) // 4)


def _jaccard_similarity(a: str, b: str) -> float:
    """Character-level Jaccard similarity for deduplication."""
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def deduplicate(chunks: List[Dict], threshold: float = 0.85) -> List[Dict]:
    """Remove near-duplicate chunks, keeping the one with higher score."""
    result: List[Dict] = []
    for chunk in chunks:
        is_dup = False
        for existing in result:
            sim = _jaccard_similarity(chunk["content"], existing["content"])
            if sim >= threshold:
                is_dup = True
                if chunk["score"] > existing["score"]:
                    existing.update(chunk)
                break
        if not is_dup:
            result.append(chunk.copy())
    return result


def compress(chunks: List[Dict], token_limit: int = None) -> List[Dict]:
    """Drop lowest-scoring chunks until the total token count is within limit."""
    if token_limit is None:
        token_limit = config.CONTEXT_TOKEN_LIMIT
    if not chunks:
        return chunks

    # Already sorted by score descending; keep dropping the last one
    total = sum(_token_count(c["content"]) for c in chunks)
    kept = list(chunks)
    while total > token_limit and len(kept) > 1:
        removed = kept.pop()
        total -= _token_count(removed["content"])
    return kept


def format_context(chunks: List[Dict]) -> Tuple[str, List[str]]:
    """
    Format chunks into a context string with numbered citations.
    Returns (context_text, sources) where sources[i] is the source of citation [i+1].
    """
    sources = []
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "未知来源")
        sources.append(source)
        lines.append(f"[{i}] 来源：{source}\n{chunk['content']}")
    return "\n\n".join(lines), sources
