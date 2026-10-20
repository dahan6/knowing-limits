"""LLM adapter supporting multiple providers (local GLM + external Kimi/OpenAI)."""
from typing import Any, Dict, List, Optional

from openai import OpenAI

import config


class LLMProvider:
    """A single OpenAI-compatible LLM provider."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        messages: list,
        stream: bool = True,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> Any:
        # kimi-for-coding only accepts temperature=1
        if self.model == "kimi-for-coding":
            temperature = 1.0
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )


class LLMAdapter:
    """Switch between local and external LLM providers at runtime."""

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.add_provider(
            "local",
            config.LLM_BASE_URL,
            config.LLM_API_KEY,
            config.LLM_MODEL,
        )
        if config.EXTERNAL_LLM_BASE_URL:
            self.add_provider(
                "external",
                config.EXTERNAL_LLM_BASE_URL,
                config.EXTERNAL_LLM_API_KEY,
                config.EXTERNAL_LLM_MODEL,
            )
        self.current = config.LLM_PROVIDER if config.LLM_PROVIDER in self.providers else "local"

    def add_provider(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        self.providers[name] = LLMProvider(name, base_url, api_key, model)

    def set_current(self, name: str) -> bool:
        if name in self.providers:
            self.current = name
            return True
        return False

    def get_current_info(self) -> Dict[str, str]:
        provider = self.providers.get(self.current)
        if not provider:
            return {}
        return {
            "provider": provider.name,
            "base_url": provider.base_url,
            "model": provider.model,
        }

    def list_providers(self) -> List[str]:
        return list(self.providers.keys())

    def chat(
        self,
        messages: list,
        stream: bool = True,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> Any:
        if self.current not in self.providers:
            raise RuntimeError(f"当前模型提供者 {self.current} 未配置")
        return self.providers[self.current].chat(
            messages, stream=stream, temperature=temperature, max_tokens=max_tokens
        )


adapter = LLMAdapter()
