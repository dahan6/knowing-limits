"""OpenAI-compatible API server for the local RAG knowledge base.

Cherry Studio / other OpenAI clients can connect to this endpoint and get
answers augmented by the knowledge base.
"""
import json
import time
import uuid
from typing import AsyncIterator, Dict, List, Optional

import config
from knowledge_base import KnowledgeBaseManager
from llm import chat, format_sources
from query_understanding import is_chitchat

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any

app = FastAPI(title="本地 RAG API", version="1.0.0")
kb_manager = KnowledgeBaseManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=config.LLM_TEMPERATURE)
    max_tokens: Optional[int] = Field(default=config.LLM_MAX_TOKENS)
    stream: bool = False
    top_k: int = 5
    use_hybrid: bool = True
    use_rerank: bool = True
    use_query_expansion: bool = False


class SearchRequest(BaseModel):
    kb_name: str = ""
    query: str
    top_k: int = 5
    use_hybrid: bool = True
    use_rerank: bool = True
    use_query_expansion: bool = False


class AddSnippetsRequest(BaseModel):
    kb_name: str = ""
    snippets: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_size: int = 500
    chunk_overlap: int = 50


@app.get("/v1/models")
def list_models():
    kbs = kb_manager.list_kbs()
    data = []
    for kb in kbs:
        data.append(
            {
                "id": kb,
                "object": "model",
                "owned_by": "local-rag",
            }
        )
    if not data:
        data.append(
            {
                "id": "rag-default",
                "object": "model",
                "owned_by": "local-rag",
            }
        )
    return {"object": "list", "data": data}


def _resolve_kb_name(model: str) -> str:
    """Use the requested model name as KB name if it exists, else fall back."""
    available = kb_manager.list_kbs()
    if model in available:
        return model
    default = getattr(config, "API_DEFAULT_KB", "")
    if default and default in available:
        return default
    return available[0] if available else ""


def _build_messages(
    request_messages: List[ChatMessage],
    kb_name: str,
    top_k: int,
    use_hybrid: bool,
    use_rerank: bool,
    use_query_expansion: bool,
) -> tuple[List[dict], str, List[dict]]:
    """Build OpenAI messages with RAG context. Returns (messages, sources_text, chunks)."""
    messages = [m.model_dump() for m in request_messages]

    # Extract last user message for retrieval
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break

    if not last_user:
        return messages, "", []

    # Extract existing system prompt or use default
    system_prompt = ""
    system_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            system_prompt = m.get("content", "")
            system_idx = i
            break

    if not kb_name:
        # No KB available
        return messages, "", []

    retrieved_chunks = []
    if not is_chitchat(last_user):
        retrieved_chunks = kb_manager.search(
            kb_name=kb_name,
            query=last_user,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_query_expansion=use_query_expansion,
        )

    if retrieved_chunks:
        from context_processor import format_context

        context_text, _ = format_context(retrieved_chunks)
        rag_system = (
            "你是一个基于本地知识库的 AI 助手。请严格根据下面提供的参考资料回答问题。"
            "如果资料中没有相关信息，请明确说明。"
        )
        if system_prompt:
            rag_system += f"\n\n用户额外设定：{system_prompt}"
        rag_system += (
            f"\n\n参考资料：\n{context_text}\n\n"
            "回答时请使用 [1]、[2] 等编号引用对应资料，并在回答最后列出参考来源。"
        )
        if system_idx >= 0:
            messages[system_idx]["content"] = rag_system
        else:
            messages.insert(0, {"role": "system", "content": rag_system})
        sources_text = format_sources(retrieved_chunks)
    else:
        # No retrieved context: just keep original system prompt
        sources_text = ""

    return messages, sources_text, retrieved_chunks


def _make_chunk(content: str, model: str, finish_reason: Optional[str] = None) -> str:
    data = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_response(
    messages: List[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    sources_text: str,
) -> AsyncIterator[str]:
    try:
        stream = chat(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
    except Exception as e:
        yield _make_chunk(f"调用大模型失败: {e}", model, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield _make_chunk(delta, model)

    yield _make_chunk("", model, finish_reason="stop")
    yield "data: [DONE]\n\n"


def _resolve_kb_name_str(name: str) -> str:
    """Resolve a human-readable KB name; fall back to first available."""
    available = kb_manager.list_kbs()
    if name and name in available:
        return name
    default = getattr(config, "API_DEFAULT_KB", "")
    if default and default in available:
        return default
    return available[0] if available else ""


@app.post("/v1/search")
def search_knowledge_base(req: SearchRequest):
    """暴露原始检索结果，供 Controller 做路由决策。"""
    available = kb_manager.list_kbs()
    kb_name = req.kb_name
    if not kb_name or kb_name not in available:
        kb_name = _resolve_kb_name(req.model if hasattr(req, "model") else "")
    if not kb_name:
        raise HTTPException(status_code=503, detail="没有可用的知识库")

    try:
        chunks = kb_manager.search(
            kb_name=kb_name,
            query=req.query,
            top_k=req.top_k,
            use_hybrid=req.use_hybrid,
            use_rerank=req.use_rerank,
            use_query_expansion=req.use_query_expansion,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")

    return {
        "query": req.query,
        "kb_name": kb_name,
        "results": chunks,
    }


@app.post("/v1/add_snippets")
def add_snippets(req: AddSnippetsRequest):
    """接收文本片段并写入指定知识库，供 Controller 验证后回写结论。"""
    kb_name = req.kb_name.strip() if req.kb_name else ""
    if not kb_name:
        kb_name = _resolve_kb_name_str("")
    if not kb_name:
        raise HTTPException(status_code=503, detail="没有可用的知识库")

    if not req.snippets:
        raise HTTPException(status_code=400, detail="snippets 不能为空")

    try:
        total = kb_manager.add_texts(
            kb_name=kb_name,
            texts=req.snippets,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"写入知识库失败: {e}")

    return {
        "kb_name": kb_name,
        "added_chunks": total,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    kb_name = _resolve_kb_name(req.model)
    if not kb_name:
        raise HTTPException(status_code=503, detail="没有可用的知识库")

    messages, sources_text, retrieved_chunks = _build_messages(
        req.messages,
        kb_name,
        req.top_k,
        req.use_hybrid,
        req.use_rerank,
        req.use_query_expansion,
    )

    if req.stream:
        return StreamingResponse(
            _stream_response(messages, req.model, req.temperature, req.max_tokens, sources_text),
            media_type="text/event-stream",
        )

    try:
        resp = chat(messages, temperature=req.temperature, max_tokens=req.max_tokens, stream=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"调用大模型失败: {e}")

    answer = resp.choices[0].message.content or ""

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "sources": [c.get("source", "") for c in retrieved_chunks],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = getattr(config, "API_HOST", "0.0.0.0")
    port = getattr(config, "API_PORT", 7861)
    uvicorn.run(app, host=host, port=port)
