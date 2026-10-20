"""本地大模型客户端"""
from typing import List, Dict
import requests

import config


class LLMClient:
    def __init__(
        self,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
        api_key: str = config.LLM_API_KEY,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _chat(self, messages: List[Dict], temperature: float = config.LLM_TEMPERATURE, max_tokens: int = config.LLM_MAX_TOKENS) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[调用大模型失败: {e}]"

    def rag_answer(self, query: str, chunks: List[Dict]) -> str:
        context = "\n\n".join(
            f"[{i+1}] {c.get('content', '')}" for i, c in enumerate(chunks)
        )
        sources = "\n".join(
            f"[{i+1}] {c.get('source', '未知来源')}" for i, c in enumerate(chunks)
        )
        messages = [
            {"role": "system", "content": (
                "你是基于本地知识库的 AI 助手。请严格根据提供的参考资料回答问题。"
                "如果资料中没有相关信息，请明确说明。"
                "回答时使用 [1]、[2] 等编号引用对应资料，并在回答最后列出参考来源。"
            )},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}\n\n请根据参考资料回答。"},
        ]
        answer = self._chat(messages)
        if sources and "参考来源" not in answer:
            answer += f"\n\n参考来源：\n{sources}"
        return answer

    def direct_answer(self, query: str, note: str = "") -> str:
        sys_msg = "你是知识丰富的 AI 助手，请直接回答用户问题。"
        if note:
            sys_msg += f"\n\n注意：{note}"
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": query},
        ]
        return self._chat(messages)

    def open_discussion(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个善于讨论的 AI 助手。用户的问题偏主观、开放或建议性，请给出多角度、审慎的回答，并说明这是观点而非唯一答案。"},
            {"role": "user", "content": query},
        ]
        return self._chat(messages)

    def web_answer(self, query: str, web_results: List[Dict]) -> str:
        context = "\n\n".join(
            f"[{i+1}] {r.get('title', '')}\n{r.get('snippet', '')}" for i, r in enumerate(web_results)
        )
        sources = "\n".join(
            f"[{i+1}] {r.get('title', '')}: {r.get('link', '')}" for i, r in enumerate(web_results)
        )
        messages = [
            {"role": "system", "content": (
                "你基于联网搜索到的摘要回答问题。请客观概括，不要编造。"
                "使用 [1]、[2] 等编号引用搜索结果，并在回答最后列出参考链接。"
            )},
            {"role": "user", "content": f"搜索摘要：\n{context}\n\n问题：{query}\n\n请根据搜索摘要回答。"},
        ]
        answer = self._chat(messages)
        if sources and "参考链接" not in answer:
            answer += f"\n\n参考链接：\n{sources}"
        return answer

    def unknown_response(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个诚实的 AI 助手。对于你没有可靠信息的问题，请明确说明无法回答，不要编造。"},
            {"role": "user", "content": query},
        ]
        return self._chat(messages)

    def rag_reasoning_answer(self, query: str, chunks: List[Dict]) -> str:
        """情况 B：KB 命中但 LLM 不自信，强制基于 KB 资料做逻辑推理。"""
        context = "\n\n".join(
            f"[{i+1}] {c.get('content', '')}" for i, c in enumerate(chunks)
        )
        sources = "\n".join(
            f"[{i+1}] {c.get('source', '未知来源')}" for i, c in enumerate(chunks)
        )
        messages = [
            {"role": "system", "content": (
                "你是一个严谨的推理助手。请严格根据提供的参考资料，先分析资料中的逻辑关系，"
                "再一步步推导出问题的答案。不要引入资料外的知识。"
                "使用 [1]、[2] 等编号引用资料，并列出参考来源。"
            )},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}\n\n请先分析资料，再给出推理过程和结论。"},
        ]
        answer = self._chat(messages)
        if sources and "参考来源" not in answer:
            answer += f"\n\n参考来源：\n{sources}"
        return answer

    def fusion_answer(
        self,
        query: str,
        kb_chunks: List[Dict],
        kb_score: float,
        llm_direct_answer: str,
        llm_known_proba: float,
        case: str = "",
    ) -> str:
        """
        鲁棒生成：同时参考 KB 检索结果和 LLM 内部知识。
        即使路由分类有偏差，也能根据证据强度给出有效回答。
        """
        context = "\n\n".join(
            f"[{i+1}] {c.get('content', '')}" for i, c in enumerate(kb_chunks)
        ) if kb_chunks else "（无相关 KB 资料）"
        sources = "\n".join(
            f"[{i+1}] {c.get('source', '未知来源')}" for i, c in enumerate(kb_chunks)
        ) if kb_chunks else ""

        messages = [
            {"role": "system", "content": (
                "你是一个谨慎的 AI 助手。请根据以下两类证据回答用户问题：\n"
                "1. KB 检索结果（长期显式记忆）\n"
                "2. LLM 内部知识判断（长期隐式记忆）\n\n"
                "请遵循以下规则：\n"
                "- 如果 KB 证据充分（分数高），优先基于 KB 资料回答，并引用来源。\n"
                "- 如果 LLM 内部知识置信度高且 KB 证据不足，可以使用 LLM 知识回答。\n"
                "- 如果两类证据冲突，请指出分歧并选择更可靠的来源。\n"
                "- 如果都不足以回答，请明确说明无法确定，不要编造。\n"
                "使用 [1]、[2] 等编号引用 KB 资料。"
            )},
            {"role": "user", "content": (
                f"问题：{query}\n\n"
                f"当前路由分类：{case or '未知'}\n"
                f"KB 检索结果（最高分 {kb_score:.3f}）：\n{context}\n\n"
                f"LLM 内部知识回答（置信度 {llm_known_proba:.3f}）：\n{llm_direct_answer}\n\n"
                f"请综合以上信息，给出最终回答。"
            )},
        ]
        answer = self._chat(messages)
        if sources and "参考来源" not in answer:
            answer += f"\n\n参考来源：\n{sources}"
        return answer

    def moe_reasoning(self, query: str, n_paths: int = 3) -> Dict:
        """情况 C：KB 未命中但 LLM 自信，走多路 MoE 推理。"""
        paths = []
        temps = [0.0, 0.5, 0.8]
        prompts = [
            "请直接、简洁地回答这个问题。",
            "请从常识和背景知识角度详细分析这个问题。",
            "请从不同角度思考这个问题，给出最合理的答案并说明理由。",
        ]
        for i in range(min(n_paths, len(temps))):
            messages = [
                {"role": "system", "content": prompts[i]},
                {"role": "user", "content": query},
            ]
            text = self._chat(messages, temperature=temps[i], max_tokens=512)
            paths.append({"temperature": temps[i], "prompt": prompts[i], "answer": text})

        # 用 LLM 做汇总与一致性判断
        combined = "\n\n".join(
            f"路径 {i+1}：{p['answer']}" for i, p in enumerate(paths)
        )
        summary_messages = [
            {"role": "system", "content": (
                "你是一位多路径答案的整合者。请综合下面不同路径的回答，"
                "给出最合理的最终结论，并指出是否一致、是否有不确定之处。"
            )},
            {"role": "user", "content": f"问题：{query}\n\n{combined}\n\n请给出最终结论。"},
        ]
        final_answer = self._chat(summary_messages, temperature=0.3, max_tokens=1024)
        return {
            "paths": paths,
            "final_answer": final_answer,
        }

    def verify_conclusion(self, query: str, conclusion: str, web_results: List[Dict]) -> Dict:
        """用联网搜索摘要验证 MoE 结论的合理性。"""
        if not web_results:
            return {"verified": False, "reason": "无联网搜索结果，无法验证"}

        context = "\n\n".join(
            f"[{i+1}] {r.get('title', '')}\n{r.get('snippet', '')}" for i, r in enumerate(web_results)
        )
        messages = [
            {"role": "system", "content": (
                "你是一位事实核查助手。请判断下面的结论是否与提供的搜索摘要一致。"
                "如果不一致或摘要不足以支持，请说明。只回答 JSON：{\"verified\": true/false, \"reason\": \"...\"}"
            )},
            {"role": "user", "content": (
                f"问题：{query}\n\n结论：{conclusion}\n\n搜索摘要：\n{context}\n\n请判断结论是否合理。"
            )},
        ]
        try:
            text = self._chat(messages, temperature=0.0, max_tokens=256)
            # 简单解析
            if '"verified": true' in text or "'verified': true" in text:
                verified = True
            elif '"verified": false' in text or "'verified': false" in text:
                verified = False
            else:
                verified = False
            return {"verified": verified, "reason": text}
        except Exception as e:
            return {"verified": False, "reason": f"验证调用失败: {e}"}

    def learning_log(self, query: str, source: str, content: str, log_path: str = "learning_log.jsonl") -> None:
        """记录需要后续学习的样本（情况 B 影子验证 / D 补充 KB）。"""
        import json
        from datetime import datetime
        from pathlib import Path

        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "source": source,
            "content": content,
        }
        path = Path(log_path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
