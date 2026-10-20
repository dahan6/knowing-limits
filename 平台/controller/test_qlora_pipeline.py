"""Tests for QLoRA implicit memory pipeline."""
import json
import tempfile
from pathlib import Path

import pytest

from training_sample_manager import TrainingSampleManager
from adapter_registry import AdapterRegistry
from shadow_validator import ShadowValidator


def test_sample_loading_and_cleaning():
    """Test loading and cleaning samples from learning_log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "learning_log.jsonl"
        lines = [
            json.dumps({"timestamp": "2026-01-01T00:00:00", "query": "测试问题", "source": "case_B_shadow", "content": "## 推理过程\n这是一个测试回答。\n\n参考来源：[1] 测试"}, ensure_ascii=False),
            json.dumps({"timestamp": "2026-01-01T00:01:00", "query": "太短", "source": "case_B_shadow", "content": "短"}, ensure_ascii=False),
            json.dumps({"timestamp": "2026-01-01T00:02:00", "query": "重复问题", "source": "case_B_shadow", "content": "## 推理过程\n重复的回答。\n\n参考来源：[1] 测试"}, ensure_ascii=False),
        ]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        mgr = TrainingSampleManager(log_path=str(log_path))
        samples = mgr.load_samples()
        assert len(samples) == 3

        cleaned = mgr.clean_samples(samples)
        # The "太短" sample should be filtered out
        assert len(cleaned) == 2


def test_deduplication():
    """Test exact and near deduplication."""
    samples = [
        {"instruction": "问题A", "output": "回答A", "topics": ["topic1"]},
        {"instruction": "问题A", "output": "回答A", "topics": ["topic1"]},  # exact dup
        {"instruction": "问题B", "output": "回答B", "topics": ["topic1"]},
        {"instruction": "问题C", "output": "回答C", "topics": ["topic2"]},
    ]

    mgr = TrainingSampleManager()
    deduped = mgr.deduplicate_samples(samples)
    # Should have removed the exact duplicate
    assert len(deduped) <= 3


def test_quality_score():
    """Test quality scoring."""
    mgr = TrainingSampleManager()

    good_sample = {
        "output": "## 推理过程\n这是一个详细的推理过程，包含多个步骤。\n\n参考来源：[1] 测试来源",
        "source_type": "llm_moe_verified",
    }
    poor_sample = {
        "output": "不知道",
        "source_type": "unknown",
    }

    assert mgr.quality_score(good_sample) > 0.5
    assert mgr.quality_score(poor_sample) < 0.5


def test_trigger_condition():
    """Test training trigger condition."""
    mgr = TrainingSampleManager()

    # Enough samples
    cluster = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(60)]
    assert mgr.check_trigger_condition(cluster, min_samples=50)

    # Repeated query
    cluster = [{"instruction": "same_q", "output": f"a{i}"} for i in range(5)]
    assert mgr.check_trigger_condition(cluster, min_samples=50, min_repeat_query=3)

    # Not enough
    cluster = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(10)]
    assert not mgr.check_trigger_condition(cluster, min_samples=50)


def test_adapter_registry():
    """Test adapter registration and compatibility checking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.json"
        registry = AdapterRegistry(registry_path=str(registry_path))

        registry.register("test_adapter", {
            "adapter_name": "test_adapter",
            "base_model": "glm-4.5-air",
            "rank": 8,
            "alpha": 16,
            "num_samples": 100,
        })

        assert registry.get_adapter("test_adapter") is not None
        assert len(registry.list_adapters()) == 1

        # Test conflict matrix
        registry.add_conflict("adapter_a", "adapter_b", "风格冲突")
        assert not registry.validate_compatibility(["adapter_a", "adapter_b"])
        assert registry.validate_compatibility(["adapter_a", "adapter_c"])


def test_shadow_validator():
    """Test shadow validation logic."""
    validator = ShadowValidator()
    result = validator.validate(adapter_path="/fake/path")
    # Should return a result dict with passed/error
    assert "passed" in result or "error" in result


def test_controller_imports():
    """Test that all controller modules import correctly."""
    from controller import Controller
    from training_sample_manager import TrainingSampleManager
    from qlora_trainer import QLoRATrainer
    from adapter_converter import AdapterConverter
    from adapter_registry import get_registry

    # Just check they can be instantiated without errors
    c = Controller()
    assert c.adapter_registry is not None
    assert c.training_manager is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
