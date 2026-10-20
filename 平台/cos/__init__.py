"""COS Framework: 深盆科技认知操作系统框架"""

from .config import COSConfig, LocalConfig, RemoteConfig
from .controller import Controller
from .providers import LLMProvider, LocalLLMProvider, RemoteLLMProvider, LLMProviderFactory

__version__ = "0.1.0"
__all__ = [
    "COSConfig",
    "LocalConfig",
    "RemoteConfig",
    "Controller",
    "LLMProvider",
    "LocalLLMProvider",
    "RemoteLLMProvider",
    "LLMProviderFactory",
]
