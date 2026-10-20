"""短期记忆（上下文）模块，支持主题化与会话切分。"""
from collections import deque
from typing import List, Dict, Optional

import config


class ContextMemory:
    """维护最近 N 轮对话，支持按主题隔离和切换检测。"""

    def __init__(self, max_turns: int = None, session_id: str = "default"):
        self.max_turns = max_turns or getattr(config, "MAX_CONTEXT_TURNS", 5)
        self.session_id = session_id
        self.turns: deque = deque(maxlen=self.max_turns)
        self.current_topic: str = "general"
        self.topic_history: List[str] = []

    def add(self, query: str, answer: str, source: str = ""):
        self.turns.append({
            "query": query,
            "answer": answer,
            "source": source,
            "topic": self.current_topic,
        })

    def context_text(self, topic_only: bool = True) -> str:
        """返回可供 LLM 使用的上下文文本。默认只返回当前主题内的对话。"""
        if not self.turns:
            return ""
        lines = []
        for i, turn in enumerate(self.turns, start=1):
            if topic_only and turn.get("topic") != self.current_topic:
                continue
            lines.append(f"第{i}轮：")
            lines.append(f"  用户：{turn['query']}")
            lines.append(f"  助手：{turn['answer']}")
        return "\n".join(lines)

    def recent_queries(self, n: int = 3) -> List[str]:
        """返回最近 n 个用户问题，用于上下文感知的扩写。"""
        return [t["query"] for t in list(self.turns)[-n:] if t.get("topic") == self.current_topic]

    def clear(self):
        self.turns.clear()
        self.current_topic = "general"

    def switch_topic(self, new_topic: str):
        """切换到新主题，可选把旧主题摘要等后续再处理。"""
        if new_topic != self.current_topic:
            self.topic_history.append(self.current_topic)
            self.current_topic = new_topic

    def detect_topic_shift(self, query: str, llm_client=None) -> bool:
        """
        判断当前 query 是否偏离当前主题。
        简单实现：基于关键词重叠；如提供 llm_client，可启用 LLM 判断。
        """
        if not self.turns:
            return False

        # Heuristic: keyword overlap with recent queries in current topic
        recent = " ".join(self.recent_queries(n=2))
        if not recent:
            return False

        recent_words = set(recent.lower().split())
        query_words = set(query.lower().split())
        if not recent_words or not query_words:
            return False

        overlap = len(recent_words & query_words)
        similarity = overlap / max(len(query_words), 1)

        # If similarity is low, likely topic shift
        if similarity < config.MEMORY_TOPIC_SHIFT_THRESHOLD:
            # Optional LLM confirmation
            if llm_client:
                try:
                    prompt = (
                        f"当前主题：{self.current_topic}\n"
                        f"近期问题：{recent}\n"
                        f"当前问题：{query}\n"
                        f"当前问题是否切换到新主题？只回答'是'或'否'。"
                    )
                    ans = llm_client.direct_answer(prompt).strip().lower()
                    return ans.startswith("是") or ans.startswith("yes")
                except Exception:
                    pass
            return True
        return False

    def contextualize(self, query: str, llm_client) -> str:
        """结合上下文，对当前 query 做指代消解和意图补全。"""
        context = self.context_text()
        if not context:
            return query

        messages = [
            {"role": "system", "content": "你负责把用户当前问题结合上下文改写成一个完整、独立的查询，便于检索和回答。不要解释，只输出改写后的问题。"},
            {"role": "user", "content": f"上下文：\n{context}\n\n当前问题：{query}\n\n请改写为完整独立的问题："},
        ]
        try:
            rewritten = llm_client._chat(messages, temperature=0.1, max_tokens=256)
            rewritten = rewritten.strip()
            return rewritten if rewritten else query
        except Exception:
            return query
