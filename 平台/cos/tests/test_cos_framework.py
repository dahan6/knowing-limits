"""COS Framework 测试"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cos import COSConfig, LocalConfig, RemoteConfig
from cos.providers import LocalLLMProvider, RemoteLLMProvider, LLMProviderFactory
from cos.controller import Controller


def test_config_creation():
    """测试配置创建"""
    config = COSConfig(mode="local")
    assert config.mode == "local"
    assert config.local is not None
    assert config.remote is not None


def test_config_from_env(monkeypatch):
    """测试从环境变量创建配置"""
    monkeypatch.setenv("COS_MODE", "remote")
    monkeypatch.setenv("REMOTE_MODEL", "gpt-4")
    config = COSConfig.from_env()
    assert config.mode == "remote"
    assert config.remote.model == "gpt-4"


def test_local_provider_creation():
    """测试本地 Provider 创建"""
    provider = LLMProviderFactory.create_local()
    assert isinstance(provider, LocalLLMProvider)
    assert provider.supports_implicit_memory() is True


def test_remote_provider_creation():
    """测试远程 Provider 创建"""
    provider = LLMProviderFactory.create_remote()
    assert isinstance(provider, RemoteLLMProvider)
    assert provider.supports_implicit_memory() is False


def test_controller_creation():
    """测试 Controller 创建"""
    provider = LLMProviderFactory.create_local()
    controller = Controller(llm_provider=provider)
    assert controller.llm is not None
    assert controller.kb is not None
    assert controller.web is not None


def test_controller_remote_mode():
    """测试远程模式 Controller"""
    provider = LLMProviderFactory.create_remote()
    controller = Controller(llm_provider=provider)
    assert controller.adapter_registry is None
    assert controller.training_manager is None


def test_provider_factory():
    """测试 Provider 工厂"""
    config = COSConfig(mode="local")
    provider = LLMProviderFactory.create(config)
    assert isinstance(provider, LocalLLMProvider)

    config = COSConfig(mode="remote")
    provider = LLMProviderFactory.create(config)
    assert isinstance(provider, RemoteLLMProvider)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
