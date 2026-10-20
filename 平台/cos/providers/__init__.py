"""LLM Provider 工厂与接口"""
from .base import LLMProvider
from .local import LocalLLMProvider
from .remote import RemoteLLMProvider
from ..config import COSConfig, LocalConfig, RemoteConfig


class LLMProviderFactory:
    """根据配置创建对应的 LLM Provider。"""

    @staticmethod
    def create(config: COSConfig = None) -> LLMProvider:
        config = config or COSConfig.from_env()
        if config.mode == "local":
            return LocalLLMProvider(config.local)
        elif config.mode == "remote":
            return RemoteLLMProvider(config.remote)
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

    @staticmethod
    def create_local(config: LocalConfig = None) -> LocalLLMProvider:
        """创建本地 Provider"""
        return LocalLLMProvider(config or LocalConfig.from_env())

    @staticmethod
    def create_remote(config: RemoteConfig = None) -> RemoteLLMProvider:
        """创建远程 Provider"""
        return RemoteLLMProvider(config or RemoteConfig.from_env())


__all__ = [
    "LLMProvider",
    "LocalLLMProvider",
    "RemoteLLMProvider",
    "LLMProviderFactory",
]
