"""查询扩写模块"""
import json
import re
from typing import List

from ..config import COSConfig as config
from ..providers.base import LLMProvider


class QueryExpander:
    """用 LLM 生成同义/补全查询，提升 KB 检索召回。"""

    def __init__(self, llm_client: LLMProvider = None):
        self.llm = llm_client

    def expand(self, query: str, n: int = 2, context: str = "") -> List[str]:
        """返回扩写后的查询列表（包含原查询）。"""
        if not query or not query.strip():
            return []

        ctx_part = f"\n\n当前对话上下文：\n{context}\n" if context else ""
        prompt = (
            f"请为下面的问题生成 {n} 个语义等价的查询变体，用于在知识库中检索相关信息。"
            f"只返回列表，不要解释。{ctx_part}\n\n原问题：{query}\n\n查询变体："
        )
        messages = [
            {"role": "system", "content": "你是一个查询扩写助手，只输出 JSON 数组格式的查询列表。"},
            {"role": "user", "content": prompt},
        ]
        try:
            if self.llm is None:
                return [query.strip()]
            text = self.llm.chat(messages, temperature=0.3, max_tokens=256)
            variants = self._parse(text)
        except Exception as e:
            print(f"[WARN] 查询扩写失败: {e}")
            variants = []

        # 去重并保证原查询在第一位
        seen = {query.strip()}
        result = [query.strip()]
        for v in variants:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                result.append(v)
        return result

    @staticmethod
    def _parse(text: str) -> List[str]:
        """从 LLM 输出中解析查询列表，支持 JSON 数组或编号列表。"""
        text = text.strip()
        if not text:
            return []

        # 尝试 JSON 数组
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass

        # 尝试 1. / - / * 开头的列表
        lines = [re.sub(r"^\s*[-\*\d]+[\.、\)]?\s*", "", line).strip()
                 for line in text.splitlines()]
        return [line for line in lines if line]
