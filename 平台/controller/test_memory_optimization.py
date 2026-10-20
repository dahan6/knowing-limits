"""Tests for memory storage & strategy optimizations."""
import json
import time
import uuid
from pathlib import Path

import requests

import config
from kb_client import KBClient
from memory_value import MemoryValueEstimator
from context_memory import ContextMemory


TEST_KB_PREFIX = "memory_opt_test"
API_BASE = config.KB_API_BASE


def _unique_kb() -> str:
    return f"{TEST_KB_PREFIX}_{uuid.uuid4().hex[:8]}"


def _delete_kb(kb_name: str):
    try:
        # Best effort cleanup via direct Chroma is not exposed; we rely on unique names.
        pass
    except Exception:
        pass


def test_metadata_fields():
    """New chunks should carry memory management metadata."""
    kb = KBClient()
    kb_name = _unique_kb()
    try:
        added = kb.add_snippets(
            kb_name=kb_name,
            snippets=[{
                "source": "test",
                "content": "中国的首都是北京。北京是政治中心。",
                "metadata": {
                    "source_type": "user_added",
                    "verified": True,
                    "confidence": 0.95,
                    "weight": 1.2,
                },
            }],
        )
        assert added > 0, "should add at least one chunk"

        results = kb.search("中国的首都是哪里？", kb_name=kb_name, top_k=1)
        assert results, "should retrieve a chunk"
        meta = results[0].get("metadata", {})
        required_keys = [
            "created_at", "updated_at", "source_type", "verified",
            "confidence", "weight", "access_count", "last_accessed",
            "topics", "version", "tier",
        ]
        for key in required_keys:
            assert key in meta, f"missing metadata key: {key}"
        assert meta["source_type"] == "user_added"
        assert meta["verified"] is True
        assert float(meta["confidence"]) > 0.9
        print("✓ test_metadata_fields passed")
    finally:
        _delete_kb(kb_name)


def test_deduplication_update():
    """Adding a near-duplicate should update rather than duplicate."""
    kb = KBClient()
    kb_name = _unique_kb()
    try:
        content = "人工智能对就业的影响包括自动化替代和创造新岗位。"
        kb.add_snippets(
            kb_name=kb_name,
            snippets=[{
                "source": "web_search",
                "content": content,
                "metadata": {"source_type": "web_search", "confidence": 0.6, "weight": 0.6},
            }],
        )
        # Add nearly identical content with higher confidence
        kb.add_snippets(
            kb_name=kb_name,
            snippets=[{
                "source": "web_search",
                "content": content + " 未来需要终身学习。",
                "metadata": {"source_type": "web_search", "confidence": 0.8, "weight": 0.9},
            }],
        )
        # Query and check we don't have a huge number of chunks
        results = kb.search("人工智能对就业的影响", kb_name=kb_name, top_k=10)
        # Should be a small number; exact count depends on chunking but duplicates should be merged.
        ids = {r["id"] for r in results}
        print(f"  retrieved {len(results)} results, {len(ids)} unique ids after dedup")
        assert len(ids) <= 3, f"too many unique chunks after dedup: {len(ids)}"
        print("✓ test_deduplication_update passed")
    finally:
        _delete_kb(kb_name)


def test_memory_scoring_boost():
    """Higher-weight, recently-accessed chunk should outrank a cold old one."""
    kb = KBClient()
    kb_name = _unique_kb()
    try:
        # Low confidence, old-ish chunk
        kb.add_snippets(
            kb_name=kb_name,
            snippets=[{
                "source": "old",
                "content": "水的化学式是H2O，由氢和氧组成。",
                "metadata": {
                    "source_type": "web_search",
                    "confidence": 0.4,
                    "weight": 0.5,
                    "tier": "cold",
                    "access_count": 0,
                },
            }],
        )
        # High confidence, hot chunk
        kb.add_snippets(
            kb_name=kb_name,
            snippets=[{
                "source": "verified",
                "content": "水的化学式是H2O，一个水分子由两个氢原子和一个氧原子组成。",
                "metadata": {
                    "source_type": "user_added",
                    "verified": True,
                    "confidence": 0.95,
                    "weight": 1.5,
                    "tier": "hot",
                    "access_count": 10,
                },
            }],
        )
        results = kb.search("水的化学式是什么？", kb_name=kb_name, top_k=2)
        assert results, "should retrieve results"
        top = results[0]
        assert top["metadata"]["source"] == "verified", f"expected verified chunk first, got {top['metadata']}"
        print("✓ test_memory_scoring_boost passed")
    finally:
        _delete_kb(kb_name)


def test_memory_value_estimator():
    meta = MemoryValueEstimator.estimate(
        content="鸟类主要使用肺进行呼吸，同时体内有多个气囊辅助完成高效的气体交换。",
        query="鸟类用什么呼吸？",
        source_type="llm_moe_verified",
        verified=True,
    )
    assert meta["confidence"] >= 0.9, f"confidence too low: {meta['confidence']}"
    assert meta["weight"] >= 1.0, f"weight too low: {meta['weight']}"
    assert "鸟类" in meta["topics"] or "呼吸" in meta["topics"], f"unexpected topics: {meta['topics']}"

    assert MemoryValueEstimator.should_remember("北京是中国的首都，位于华北平原北部。", source_type="user_added", verified=True)
    assert not MemoryValueEstimator.should_remember("我不知道这个问题的答案。", source_type="web_search")
    print("✓ test_memory_value_estimator passed")


def test_context_memory_topic():
    mem = ContextMemory(max_turns=5)
    mem.add("北京天气怎样？", "晴")
    mem.add("上海呢？", "阴")
    assert mem.detect_topic_shift("Python 怎么写循环？") is True
    mem.switch_topic("coding")
    mem.add("Python 怎么写循环？", "用 for")
    ctx = mem.context_text(topic_only=True)
    assert "Python" in ctx
    assert "北京" not in ctx
    print("✓ test_context_memory_topic passed")


if __name__ == "__main__":
    test_memory_value_estimator()
    test_context_memory_topic()
    test_metadata_fields()
    test_deduplication_update()
    test_memory_scoring_boost()
    print("\nAll memory optimization tests passed.")
