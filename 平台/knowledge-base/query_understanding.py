"""Query understanding: intent detection and query rewriting/expansion."""
import re
from typing import List

import llm


_CHITCHAT_PATTERNS = re.compile(
    r"^(\s*(你好|您好|嗨|hello|hi|hey|在吗|在？|请问|谢谢|再见|拜拜|好的|ok|好的，|嗯|哦|啊)\s*[！。，.!?]?\s*)+$",
    re.IGNORECASE | re.UNICODE,
)


def is_chitchat(query: str) -> bool:
    """Simple heuristic to detect pure greetings/thanks/farewells."""
    return bool(_CHITCHAT_PATTERNS.match(query.strip()))


def should_use_kb(query: str) -> bool:
    return not is_chitchat(query)


def rewrite_queries(query: str, n: int = 2) -> List[str]:
    """
    Ask the local LLM to produce alternative phrasings of the query.
    Returns [original, ...variants].
    """
    if not query or not query.strip():
        return []
    query = query.strip()

    prompt = (
        f"你是一个检索系统的查询改写助手。请针对下面的用户问题，生成 {n} 个语义等价但表述不同的查询变体，"
        "用于从文档中检索相关信息。只输出改写后的查询，每行一个，不要编号、不要解释。\n\n"
        f"原问题：{query}\n"
        "改写："
    )
    messages = [
        {"role": "system", "content": "你只输出查询改写结果，每行一个。"},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = llm.chat(messages, stream=False)
        text = resp.choices[0].message.content or ""
        variants = [line.strip() for line in text.splitlines() if line.strip()]
        # Deduplicate while preserving order
        seen = {query}
        result = [query]
        for v in variants:
            if v not in seen and len(v) > 3:
                seen.add(v)
                result.append(v)
            if len(result) >= n + 1:
                break
        return result
    except Exception:
        return [query]
