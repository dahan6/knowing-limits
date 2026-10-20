"""COS Framework 核心控制器：后端无关的四象限路由状态机"""
from typing import Dict, List, Optional

from .config import COSConfig
from .providers.base import LLMProvider
from .providers import LLMProviderFactory
from .memory.context_memory import ContextMemory
from .memory.kb_client import KBClient
from .routing.classifier import LLMClassifier
from .routing.router import QuadrantRouter
from .routing.fusion import FusionGenerator
from .memory_value.estimator import MemoryValueEstimator
from .utils.web_search import WebSearchClient
from .utils.query_expander import QueryExpander


class Controller:
    """COS 控制器：输入 → 查询扩写 → 并行检索 → 四象限路由 → 分支生成 → 记忆固化"""

    def __init__(
        self,
        llm_provider: LLMProvider = None,
        config: COSConfig = None,
        kb_client: KBClient = None,
        web_client: WebSearchClient = None,
        classifier: LLMClassifier = None,
        expander: QueryExpander = None,
        memory: ContextMemory = None,
    ):
        self.config = config or COSConfig.from_env()
        self.llm = llm_provider or LLMProviderFactory.create(self.config)
        self.kb = kb_client or KBClient(self.config.kb_api_base)
        self.web = web_client or WebSearchClient(
            enabled=self.config.enable_web_search,
            region=self.config.web_search_region,
        )
        self.classifier = classifier or LLMClassifier()
        self.expander = expander or QueryExpander(self.llm)
        self.memory = memory or ContextMemory(max_turns=self.config.max_context_turns)
        self.router = QuadrantRouter(self.config)
        self.fusion = FusionGenerator(self.llm)

        # 隐式记忆组件（仅本地模式可用）
        if self.llm.supports_implicit_memory():
            from .adapters.registry import AdapterRegistry
            from .memory.training_manager import TrainingSampleManager
            self.adapter_registry = AdapterRegistry()
            self.training_manager = TrainingSampleManager()
        else:
            self.adapter_registry = None
            self.training_manager = None

    # ------------------------------------------------------------------
    # 核心流程
    # ------------------------------------------------------------------
    def process(self, query: str) -> Dict:
        """处理用户查询，返回完整结果"""
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
            "adapter_used": None,
        }

        # 1. 上下文思考
        query_ctx = self.memory.contextualize(query, self.llm)
        result["contextualized_query"] = query_ctx

        # 2. 查询扩写
        context_text = self.memory.context_text()
        expanded_queries = self.expander.expand(
            query_ctx, n=2, context=context_text
        )
        result["expanded_queries"] = expanded_queries

        # 3. 并行检索
        kb_chunks = self._kb_search_expanded(expanded_queries)
        det = self.classifier.detect(query_ctx)
        result["llm_detection"] = det

        kb_score = float(kb_chunks[0].get("raw_score", kb_chunks[0].get("score", 0))) if kb_chunks else 0.0
        llm_known_proba = float(det["proba"].get("known", 0))

        result["kb_score"] = kb_score
        result["kb_chunks"] = kb_chunks[: self.config.kb_top_k]

        # 4. 四象限路由
        case = self.router.classify(kb_score, llm_known_proba)
        result["case"] = case

        # 边界区检测
        is_borderline = self.router.is_borderline(kb_score, llm_known_proba)
        result["needs_fact_check"] = is_borderline

        # 冲突检测
        conflicts = self._detect_conflicts(result["kb_chunks"])
        if conflicts:
            trusted = {self._arbitrate_conflict(query, g)["id"] for g in conflicts}
            result["kb_chunks"] = [
                c for c in result["kb_chunks"]
                if c["id"] in trusted or not any(c["id"] in [x["id"] for x in g] for g in conflicts)
            ]
            result["conflicts"] = [[c["id"] for c in g] for g in conflicts]

        # 融合判断
        need_fusion = self.router.need_fusion(kb_score, llm_known_proba, case)
        llm_direct_answer = ""
        if need_fusion:
            llm_direct_answer = self.llm.direct_answer(query_ctx)

        # 隐式记忆：adapter 选择（仅本地模式）
        adapter_name = self._select_adapter(query, case)
        if adapter_name and self.llm.supports_implicit_memory():
            result["adapter_used"] = adapter_name
            self.llm.load_adapter(adapter_name)

        # 5. 分支生成
        if case == "A":
            result = self._case_a(query, result, need_fusion, llm_direct_answer)
        elif case == "B":
            result = self._case_b(query, result, need_fusion, llm_direct_answer)
        elif case == "C":
            result = self._case_c(query, result, need_fusion, llm_direct_answer)
        else:  # D
            result = self._case_d(query, result, need_fusion, llm_direct_answer)

        # 6. 记忆固化
        self._finish(query, result)
        return result

    # ------------------------------------------------------------------
    # 分支处理
    # ------------------------------------------------------------------
    def _case_a(self, query: str, result: Dict, need_fusion: bool, llm_direct_answer: str) -> Dict:
        """Case A: KB 高 + LLM 高 → 直接 RAG"""
        result["source"] = "kb"
        result["confidence"] = (result["kb_score"] + result["llm_detection"]["proba"]["known"]) / 2
        if need_fusion:
            result["answer"] = self.fusion.answer(
                query, result["kb_chunks"], result["kb_score"],
                llm_direct_answer, result["llm_detection"]["proba"]["known"], case="A"
            )
        else:
            result["answer"] = self.llm.rag_answer(query, result["kb_chunks"])
        return result

    def _case_b(self, query: str, result: Dict, need_fusion: bool, llm_direct_answer: str) -> Dict:
        """Case B: KB 高 + LLM 低 → KB 逻辑推理"""
        result["source"] = "kb_reasoned"
        result["confidence"] = result["kb_score"]
        if need_fusion:
            result["answer"] = self.fusion.answer(
                query, result["kb_chunks"], result["kb_score"],
                llm_direct_answer, result["llm_detection"]["proba"]["known"], case="B"
            )
        else:
            result["answer"] = self.llm.rag_reasoning_answer(query, result["kb_chunks"])
        # 影子验证样本
        if self.training_manager:
            self.training_manager.log_sample(
                query=query,
                content=result["answer"],
                source="case_B_shadow",
            )
        return result

    def _case_c(self, query: str, result: Dict, need_fusion: bool, llm_direct_answer: str) -> Dict:
        """Case C: KB 低 + LLM 高 → MoE 多路推理"""
        moe = self.llm.moe_reasoning(query, n_paths=3)
        result["moe_paths"] = moe["paths"]
        conclusion = moe["final_answer"]

        # 联网验证
        web_results = []
        if self.config.enable_web_search:
            web_results = self.web.search(query, top_k=self.config.web_search_top_k)
            result["web_results"] = web_results

        verification = self.llm.verify_conclusion(query, conclusion, web_results)
        result["verification"] = verification

        # 验证通过写回 KB
        if verification.get("verified") and self.config.enable_memory_value_estimation:
            if MemoryValueEstimator.should_remember(conclusion, query, source_type="llm_moe_verified", verified=True):
                meta = MemoryValueEstimator.estimate(
                    content=conclusion, query=query,
                    source_type="llm_moe_verified", verified=True,
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
        result["confidence"] = result["llm_detection"]["proba"]["known"]
        result["answer"] = self.fusion.answer(
            query, result["kb_chunks"], result["kb_score"],
            conclusion, result["llm_detection"]["proba"]["known"], case="C"
        )
        return result

    def _case_d(self, query: str, result: Dict, need_fusion: bool, llm_direct_answer: str) -> Dict:
        """Case D: KB 低 + LLM 低 → 拒绝/学习"""
        web_results = []
        if self.config.enable_web_search:
            web_results = self.web.search(query, top_k=self.config.web_search_top_k)
            result["web_results"] = web_results

        # 网查学习写回 KB
        if web_results and self.config.enable_memory_value_estimation:
            valuable_snippets = []
            for r in web_results:
                snippet = r.get("snippet", "").strip()
                if not snippet:
                    continue
                if MemoryValueEstimator.should_remember(snippet, query, source_type="web_search"):
                    meta = MemoryValueEstimator.estimate(
                        content=snippet, query=query, source_type="web_search",
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
        if need_fusion:
            if not llm_direct_answer:
                llm_direct_answer = self.llm.direct_answer(query)
            result["answer"] = self.fusion.answer(
                query, result["kb_chunks"], result["kb_score"],
                llm_direct_answer, result["llm_detection"]["proba"]["known"], case="D"
            )
        else:
            result["answer"] = self.llm.unknown_response(query)

        # 记录学习样本
        if self.training_manager:
            self.training_manager.log_sample(
                query=query, content="", source="case_D_unknown",
            )
        return result

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _kb_search_expanded(self, expanded_queries: List[str]) -> List[dict]:
        """KB 检索并合并结果"""
        if not expanded_queries:
            expanded_queries = [""]
        per_query_results = [
            self.kb.search(q, top_k=self.config.kb_top_k)
            for q in expanded_queries
        ]
        return self._aggregate_kb_results(per_query_results)

    @staticmethod
    def _aggregate_kb_results(per_query_results: List[List[dict]]) -> List[dict]:
        """合并多个扩写查询的检索结果"""
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

    @staticmethod
    def _source_trust_rank(source_type: str) -> int:
        """来源信任等级"""
        return {
            "user_added": 4,
            "file": 3,
            "llm_moe_verified": 2,
            "web_search": 1,
            "unknown": 0,
        }.get(source_type, 0)

    def _detect_conflicts(self, kb_chunks: List[dict]) -> List[List[dict]]:
        """检测 KB 冲突"""
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
                if abs(conf_i - conf_j) > 0.3 and self._source_trust_rank(type_i) != self._source_trust_rank(type_j):
                    if self._source_trust_rank(type_i) >= 2 or self._source_trust_rank(type_j) >= 2:
                        group.append(kb_chunks[j])
            if len(group) > 1:
                conflicts.append(group)
        return conflicts

    def _arbitrate_conflict(self, query: str, conflict_group: List[dict]) -> dict:
        """仲裁冲突"""
        return max(
            conflict_group,
            key=lambda c: (
                self._source_trust_rank(c.get("metadata", {}).get("source_type", "unknown")),
                float(c.get("metadata", {}).get("confidence", 0.5)),
                c.get("metadata", {}).get("verified", False),
            ),
        )

    def _resolve_target_kb(self, topics: List[str] = None) -> str:
        """选择目标 KB"""
        return self.config.default_kb_name

    def _select_adapter(self, query: str, case: str) -> str:
        """选择 adapter（仅本地模式）"""
        if not self.adapter_registry:
            return ""
        adapters = self.adapter_registry.get_active_adapters()
        query_lower = query.lower()
        if any(k in query_lower for k in ["代码", "code", "编程", "python", "函数"]):
            for name in adapters:
                if "code" in name.lower():
                    return name
        if case == "D" and any(k in query_lower for k in ["意义", "看法", "评价", "如何", "怎样"]):
            for name in adapters:
                if "discussion" in name.lower() or "open" in name.lower():
                    return name
        return ""

    def _finish(self, query: str, result: Dict) -> Dict:
        """记忆固化"""
        self.memory.add(query, result.get("answer", ""), result.get("source", ""))
        return result
