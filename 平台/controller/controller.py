"""Controller V0.1 核心状态机

架构（按之前讨论确定）：
1. 查询扩写
2. KB 检索 与 LLM 知识检测 并行执行
3. 根据 KB 分数 与 LLM known 概率 的四种组合分类：
   A. KB高 + LLM高  → 直接 RAG 生成
   B. KB高 + LLM低  → 依赖 KB 生成 + 逻辑推理思考
   C. KB低 + LLM高  → MoE 多路推理，验证合理后写入 KB
   D. KB低 + LLM低  → 拒绝回答，补充 KB 后学习
"""
from typing import Dict, List

import config
from kb_client import KBClient
from web_search import WebSearchClient
from llm_client import LLMClient
from classifier import LLMClassifier
from cascade_detector import get_cascade_detector
from query_expander import QueryExpander
from context_memory import ContextMemory
from memory_value import MemoryValueEstimator
from adapter_registry import get_registry
from training_sample_manager import TrainingSampleManager


class Controller:
    def __init__(
        self,
        kb_client: KBClient = None,
        web_client: WebSearchClient = None,
        llm_client: LLMClient = None,
        classifier: LLMClassifier = None,
        expander: QueryExpander = None,
        memory: ContextMemory = None,
        adapter_registry=None,
        training_manager=None,
    ):
        self.kb = kb_client or KBClient()
        self.web = web_client or WebSearchClient()
        self.llm = llm_client or LLMClient()
        self.classifier = classifier or LLMClassifier()
        self.expander = expander or QueryExpander(self.llm)
        self.memory = memory or ContextMemory(max_turns=getattr(config, "MAX_CONTEXT_TURNS", 5))
        self.adapter_registry = adapter_registry or get_registry()
        self.training_manager = training_manager or TrainingSampleManager()

    @staticmethod
    def _aggregate_kb_results(per_query_results: List[List[dict]]) -> List[dict]:
        """合并多个扩写查询的检索结果，按 id 去重并保留最高 score。"""
        best = {}
        for results in per_query_results:
            for r in results:
                rid = r.get("id")
                if rid is None:
                    continue
                score = float(r.get("score", 0))
                if rid not in best or score > best[rid].get("score", 0):
                    best[rid] = r
        return sorted(best.values(), key=lambda x: x.get("score", 0), reverse=True)

    def _kb_search_expanded(self, expanded_queries: List[str]) -> List[dict]:
        """对原查询及扩写查询分别检索，合并后返回排序结果。"""
        if not expanded_queries:
            expanded_queries = [""]
        per_query_results = [self.kb.search(q, top_k=config.KB_TOP_K) for q in expanded_queries]
        return self._aggregate_kb_results(per_query_results)

    @staticmethod
    def _classify(kb_score: float, llm_known_proba: float) -> str:
        """
        四种情况：
        A. KB高 + LLM高 → "A"
        B. KB高 + LLM低 → "B"
        C. KB低 + LLM高 → "C"
        D. KB低 + LLM低 → "D"
        """
        kb_high = kb_score >= config.KB_HIT_HIGH
        llm_high = llm_known_proba >= config.LLM_KNOWN_HIGH

        if kb_high and llm_high:
            return "A"
        if kb_high and not llm_high:
            return "B"
        if not kb_high and llm_high:
            return "C"
        return "D"

    @staticmethod
    def _need_fusion(kb_score: float, llm_known_proba: float, case: str) -> bool:
        """
        判断是否需要走多证据融合生成。
        触发条件：
        1. 分数落在阈值边界区（路由可能不准）。
        2. KB 其实有中等以上命中，但路由却走了 C/D（可能漏用 KB）。
        3. LLM 其实有一定自信，但路由却走了 B/D（可能漏用 LLM 知识）。
        其它清晰分类则走原分支，节省一次 LLM 调用。
        """
        kb_borderline = abs(kb_score - config.KB_HIT_HIGH) < config.ROUTING_BORDERLINE_KB
        llm_borderline = abs(llm_known_proba - config.LLM_KNOWN_HIGH) < config.ROUTING_BORDERLINE_LLM
        if kb_borderline or llm_borderline:
            return True

        kb_useful = kb_score >= config.KB_HIT_LOW
        llm_useful = llm_known_proba >= config.LLM_KNOWN_LOW

        if kb_useful and case in ("C", "D"):
            return True
        if llm_useful and case in ("B", "D"):
            return True
        return False

    def _resolve_target_kb(self, topics: List[str] = None) -> str:
        """根据隔离模式选择目标 KB 名称。"""
        mode = config.KB_ISOLATION_MODE
        if mode == "single":
            return self.kb.default_kb
        if mode == "by_topic" and topics:
            # Use the first topic as KB suffix; fall back to default
            primary = topics[0].lower().replace(" ", "_")[:30]
            kb_name = f"{self.kb.default_kb}_{primary}" if primary else self.kb.default_kb
            self.kb.ensure_kb(kb_name)
            return kb_name
        if mode == "by_user":
            # Reserved for future multi-user support
            return self.kb.default_kb
        return self.kb.default_kb

    @staticmethod
    def _source_trust_rank(source_type: str) -> int:
        """Higher number = more trustworthy."""
        return {
            "user_added": 4,
            "file": 3,
            "llm_moe_verified": 2,
            "web_search": 1,
            "unknown": 0,
        }.get(source_type, 0)

    def _detect_conflicts(self, kb_chunks: List[dict]) -> List[List[dict]]:
        """
        Detect groups of chunks that appear to contradict each other.
        Simple heuristic: if top chunks have very different source_type/confidence
        and low mutual similarity, treat as potential conflict.
        Returns groups of conflicting chunks (each group is a list).
        """
        if len(kb_chunks) < 2:
            return []

        conflicts = []
        for i in range(len(kb_chunks)):
            group = [kb_chunks[i]]
            meta_i = kb_chunks[i].get("metadata", {})
            type_i = meta_i.get("source_type", "unknown")
            conf_i = float(meta_i.get("confidence", 0.5))
            for j in range(i + 1, len(kb_chunks)):
                meta_j = kb_chunks[j].get("metadata", {})
                type_j = meta_j.get("source_type", "unknown")
                conf_j = float(meta_j.get("confidence", 0.5))
                # Conflict signal: one is verified/user_added and the other is unverified web/moe
                if abs(conf_i - conf_j) > 0.3 and self._source_trust_rank(type_i) != self._source_trust_rank(type_j):
                    if self._source_trust_rank(type_i) >= 2 or self._source_trust_rank(type_j) >= 2:
                        group.append(kb_chunks[j])
            if len(group) > 1:
                conflicts.append(group)
        return conflicts

    def _arbitrate_conflict(self, query: str, conflict_group: List[dict]) -> dict:
        """Pick the most trustworthy chunk from a conflict group."""
        best = max(
            conflict_group,
            key=lambda c: (
                self._source_trust_rank(c.get("metadata", {}).get("source_type", "unknown")),
                float(c.get("metadata", {}).get("confidence", 0.5)),
                c.get("metadata", {}).get("verified", False),
            ),
        )
        return best

    def _select_adapter(self, query: str, case: str) -> str:
        """根据查询类型和路由情况选择要挂载的 adapter。"""
        # 简单的 adapter 选择逻辑：按 case 和查询关键词
        adapters = self.adapter_registry.get_active_adapters()
        if not adapters:
            return ""

        query_lower = query.lower()
        # 示例规则：代码相关查询使用 code_style adapter
        if any(k in query_lower for k in ["代码", "code", "编程", "python", "函数"]):
            for name in adapters:
                if "code" in name.lower():
                    return name
        # 开放讨论使用 open_discussion adapter
        if case == "D" and any(k in query_lower for k in ["意义", "看法", "评价", "如何", "怎样"]):
            for name in adapters:
                if "discussion" in name.lower() or "open" in name.lower():
                    return name
        return ""

    def _maybe_collect_training_sample(self, query: str, result: Dict):
        """判断本轮对话是否值得加入训练样本池。"""
        # 收集条件：
        # 1. case B/C/D 的回答（需要推理/学习的情况）
        # 2. 用户明确纠正（TODO: 需要用户反馈接口）
        # 3. 系统检测到幻觉或错误（TODO: 需要自动检测）
        case = result.get("case")
        if case in ("B", "C", "D"):
            answer = result.get("answer", "")
            if answer and not answer.startswith("[调用大模型失败"):
                source_map = {
                    "B": "case_B_shadow",
                    "C": "llm_moe_verified",
                    "D": "web_search",
                }
                self.training_manager.log_sample(
                    query=query,
                    content=answer,
                    source=source_map.get(case, "unknown"),
                )

    def _finish(self, query: str, result: Dict) -> Dict:
        """把本轮结果记入短期记忆，然后返回。"""
        self.memory.add(query, result.get("answer", ""), result.get("source", ""))
        self._maybe_collect_training_sample(query, result)
        return result

    def process(self, query: str) -> Dict:
        result = {
            "query": query,
            "contextualized_query": query,
            "context": self.memory.context_text(),
            "expanded_queries": [],
            "case": None,
            "source": None,
            "kb_score": 0.0,
            "kb_chunks": [],
            "llm_detection": None,
            "web_results": [],
            "moe_paths": [],
            "verification": None,
            "kb_added_chunks": 0,
            "answer": "",
            "confidence": 0.0,
            "needs_fact_check": False,
        }

        # 1. 上下文思考：把当前 query 结合上下文改写成独立查询
        query_ctx = self.memory.contextualize(query, self.llm)
        result["contextualized_query"] = query_ctx

        # 2. 查询扩写（基于上下文理解后的 query）
        context_text = self.memory.context_text()
        expanded_queries = self.expander.expand(query_ctx, n=config.QUERY_EXPANSION_N, context=context_text)
        result["expanded_queries"] = expanded_queries

        # 3. 并行：KB 检索 + LLM 知识检测
        kb_chunks = self._kb_search_expanded(expanded_queries)
        det = self.classifier.detect(query_ctx)
        result["llm_detection"] = det

        # Use raw_score for routing decision so memory boost doesn't inflate weak hits into A/B.
        kb_score = float(kb_chunks[0].get("raw_score", kb_chunks[0].get("score", 0))) if kb_chunks else 0.0
        llm_known_proba = float(det["proba"].get("known", 0))

        result["kb_score"] = kb_score
        result["kb_chunks"] = kb_chunks[: config.KB_TOP_K]

        # 3. 分类
        case = self._classify(kb_score, llm_known_proba)
        result["case"] = case

        # 计算路由置信度与边界区：即使路由不准确，也通过融合生成保证答案有效性
        kb_borderline = abs(kb_score - config.KB_HIT_HIGH) < config.ROUTING_BORDERLINE_KB
        llm_borderline = abs(llm_known_proba - config.LLM_KNOWN_HIGH) < config.ROUTING_BORDERLINE_LLM
        is_borderline = kb_borderline or llm_borderline
        result["needs_fact_check"] = is_borderline

        # 冲突检测：若 KB 命中内部存在高可信 vs 低可信的矛盾，进行仲裁
        conflicts = self._detect_conflicts(result["kb_chunks"])
        if conflicts:
            # For now, keep the most trustworthy chunk in each group and drop the rest
            trusted = {self._arbitrate_conflict(query, g)["id"] for g in conflicts}
            result["kb_chunks"] = [c for c in result["kb_chunks"] if c["id"] in trusted or not any(c["id"] in [x["id"] for x in g] for g in conflicts)]
            result["conflicts"] = [[c["id"] for c in g] for g in conflicts]

        # 只在必要时预取 LLM 内部知识回答，作为融合生成输入
        need_fusion = self._need_fusion(kb_score, llm_known_proba, case)
        llm_direct_answer = ""
        if need_fusion:
            llm_direct_answer = self.llm.direct_answer(query_ctx)

        # 隐式记忆：根据查询类型选择 adapter
        adapter_name = self._select_adapter(query, case)
        if adapter_name:
            result["adapter_used"] = adapter_name
            # TODO: 通过 LLM 客户端挂载 adapter
            # 当前 llama.cpp 需要重启加载，V0.1 先记录使用情况

        # ---------------------------------------------------------------
        # 情况 A：KB 高 + LLM 高 → 直接 RAG 生成
        # ---------------------------------------------------------------
        if case == "A":
            result["source"] = "kb"
            result["confidence"] = (kb_score + llm_known_proba) / 2
            if need_fusion:
                result["answer"] = self.llm.fusion_answer(
                    query, result["kb_chunks"], kb_score, llm_direct_answer, llm_known_proba, case="A"
                )
            else:
                result["answer"] = self.llm.rag_answer(query, result["kb_chunks"])
            return self._finish(query, result)

        # ---------------------------------------------------------------
        # 情况 B：KB 高 + LLM 低 → 依赖 KB 生成 + 逻辑推理思考
        # ---------------------------------------------------------------
        if case == "B":
            result["source"] = "kb_reasoned"
            result["confidence"] = kb_score
            if need_fusion:
                result["answer"] = self.llm.fusion_answer(
                    query, result["kb_chunks"], kb_score, llm_direct_answer, llm_known_proba, case="B"
                )
            else:
                result["answer"] = self.llm.rag_reasoning_answer(query, result["kb_chunks"])
            # 影子验证：记录样本，供后续 frozen LoRA 微调
            self.llm.learning_log(
                query=query,
                source="case_B_shadow",
                content=result["answer"],
            )
            return self._finish(query, result)

        # ---------------------------------------------------------------
        # 情况 C：KB 低 + LLM 高 → MoE 多路推理
        # ---------------------------------------------------------------
        if case == "C":
            moe = self.llm.moe_reasoning(query, n_paths=3)
            result["moe_paths"] = moe["paths"]
            conclusion = moe["final_answer"]

            # 联网搜索验证
            web_results = []
            if config.ENABLE_WEB_SEARCH:
                web_results = self.web.search(query, top_k=config.WEB_SEARCH_TOP_K)
                result["web_results"] = web_results

            verification = self.llm.verify_conclusion(query, conclusion, web_results)
            result["verification"] = verification

            if verification.get("verified") and config.ENABLE_MEMORY_VALUE_ESTIMATION:
                # 评估记忆价值后再回写 KB
                if MemoryValueEstimator.should_remember(conclusion, query, source_type="llm_moe_verified", verified=True):
                    meta = MemoryValueEstimator.estimate(
                        content=conclusion,
                        query=query,
                        source_type="llm_moe_verified",
                        verified=True,
                    )
                    target_kb = self._resolve_target_kb(meta.get("topics", []))
                    added = self.kb.add_snippets(
                        kb_name=target_kb,
                        snippets=[{
                            "source": "llm_moe_verified",
                            "content": f"问题：{query}\n结论：{conclusion}",
                            "metadata": meta,
                        }],
                    )
                    result["kb_added_chunks"] = added
                    result["target_kb"] = target_kb

            result["source"] = "llm_moe"
            result["needs_fact_check"] = True
            result["confidence"] = llm_known_proba
            # 即使走 MoE，也把 KB 检索结果（可能为空/弱相关）和 LLM 内部回答纳入最终融合，
            # 防止 KB 其实命中了少量关键信息却被忽略。
            result["answer"] = self.llm.fusion_answer(
                query,
                result["kb_chunks"],
                kb_score,
                conclusion,
                llm_known_proba,
                case="C",
            )
            return self._finish(query, result)

        # ---------------------------------------------------------------
        # 情况 D：KB 低 + LLM 低 → 拒绝回答，补充 KB 后学习
        # ---------------------------------------------------------------
        # 先尝试联网搜索作为补充材料
        web_results = []
        if config.ENABLE_WEB_SEARCH:
            web_results = self.web.search(query, top_k=config.WEB_SEARCH_TOP_K)
            result["web_results"] = web_results

        if web_results and config.ENABLE_MEMORY_VALUE_ESTIMATION:
            # 评估并过滤搜索摘要，只写回有价值的记忆
            valuable_snippets = []
            for r in web_results:
                snippet = r.get("snippet", "").strip()
                if not snippet:
                    continue
                if MemoryValueEstimator.should_remember(snippet, query, source_type="web_search"):
                    meta = MemoryValueEstimator.estimate(
                        content=snippet,
                        query=query,
                        source_type="web_search",
                    )
                    valuable_snippets.append({
                        "source": r.get("link", "web_search"),
                        "content": snippet,
                        "metadata": meta,
                    })
            if valuable_snippets:
                target_kb = self._resolve_target_kb([])
                added = self.kb.add_snippets(kb_name=target_kb, snippets=valuable_snippets)
                result["kb_added_chunks"] = added
                result["target_kb"] = target_kb

        result["source"] = "unknown"
        result["confidence"] = 0.0
        # D 类：明确拒绝；若 LLM 其实有一定自信（路由可能不准），则融合生成而非简单拒绝
        if need_fusion:
            if not llm_direct_answer:
                llm_direct_answer = self.llm.direct_answer(query_ctx)
            result["answer"] = self.llm.fusion_answer(
                query,
                result["kb_chunks"],
                kb_score,
                llm_direct_answer,
                llm_known_proba,
                case="D",
            )
        else:
            result["answer"] = self.llm.unknown_response(query)
        # 记录学习样本
        self.llm.learning_log(
            query=query,
            source="case_D_unknown",
            content="",
        )
        return self._finish(query, result)
