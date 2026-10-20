"""Simple LLM client wrapper with provider switching support."""
from typing import List, Tuple

import config
from context_processor import format_context
from llm_adapter import adapter


def chat(
    messages: list,
    temperature: float = config.LLM_TEMPERATURE,
    max_tokens: int = config.LLM_MAX_TOKENS,
    stream: bool = True,
):
    """Call the current LLM provider. Returns a streaming iterator if stream=True."""
    return adapter.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=stream)


def switch_provider(name: str) -> bool:
    """Switch the active LLM provider at runtime."""
    return adapter.set_current(name)


def current_provider_info() -> dict:
    """Return information about the current LLM provider."""
    return adapter.get_current_info()


def list_providers() -> List[str]:
    """List available LLM provider names."""
    return adapter.list_providers()


def build_rag_messages(
    query: str,
    context_chunks: List[dict],
    system_prompt: str = "",
) -> list:
    """Build messages with numbered citations and source tracking."""
    if not system_prompt:
        system_prompt = (
            "你是一个基于本地知识库的 AI 助手。请严格根据下面提供的参考资料回答问题。"
            "如果资料中没有相关信息，请明确说明。回答时请使用 [1]、[2] 等编号引用对应资料，"
            "并在末尾列出参考来源。"
        )

    context_text, sources = format_context(context_chunks)

    user_content = (
        f"参考资料：\n{context_text}\n\n"
        f"问题：{query}\n\n"
        "请根据参考资料回答，使用 [编号] 格式标注来源，并在回答最后列出参考来源。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def format_sources(context_chunks: List[dict]) -> str:
    """Format a source list for display in the UI, grouping repeated sources."""
    if not context_chunks:
        return ""
    from collections import OrderedDict

    source_to_indices: OrderedDict[str, List[int]] = OrderedDict()
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "未知来源")
        source_to_indices.setdefault(source, []).append(i)

    lines = ["参考来源："]
    for source, indices in source_to_indices.items():
        numbers = "".join(f"[{i}]" for i in indices)
        lines.append(f"{numbers} {source}")
    return "\n".join(lines)
