#!/usr/bin/env python3
"""
三阶段评估：冷启动路由测试 → 学习效能测试 → 热启动路由测试。
用于在 KB 自学习场景下公平评估 Controller。
"""
import json
import time
import uuid
from pathlib import Path

from controller import Controller
from kb_client import KBClient

DATASET = Path(__file__).parent / "test_dataset.json"
REPORT_DIR = Path(__file__).parent / "three_phase_reports"
SEED_KB_PREFIX = "controller_eval_kb"


def _fresh_kb_name() -> str:
    return f"{SEED_KB_PREFIX}_{uuid.uuid4().hex[:8]}"


def _build_seed_kb(kb_name: str) -> None:
    """使用 build_test_kb.py 的内容构建种子 KB。"""
    from build_test_kb import MANUAL_CONTENT
    from web_search import WebSearchClient
    from llm_client import LLMClient

    kb = KBClient(default_kb=kb_name)
    web = WebSearchClient()
    llm = LLMClient()

    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_added = 0
    for item in dataset:
        if not item.get("needs_kb"):
            continue
        q = item["question"]
        topic = item.get("kb_topic", "")
        snippets = []
        # 优先使用手动内容，保证测试可复现；否则才用网络搜索
        if topic in MANUAL_CONTENT and q in MANUAL_CONTENT[topic]:
            snippets = MANUAL_CONTENT[topic][q]
            # 给手动种子内容打上高信任标记，防止被学习阶段覆盖
            snippets = [{
                **s,
                "metadata": {
                    "source_type": "user_added",
                    "verified": True,
                    "confidence": 0.95,
                    "weight": 1.5,
                },
            } for s in snippets]
        elif "general" in MANUAL_CONTENT and q in MANUAL_CONTENT["general"]:
            snippets = MANUAL_CONTENT["general"][q]
            snippets = [{
                **s,
                "metadata": {
                    "source_type": "user_added",
                    "verified": True,
                    "confidence": 0.95,
                    "weight": 1.5,
                },
            } for s in snippets]
        else:
            results = web.search(q, top_k=3)
            if results:
                summary = "\n".join([r.get("snippet", "") for r in results if r.get("snippet")])
                if summary:
                    snippets = [{"source": f"web_search_{item['id']}", "content": summary}]
        if snippets:
            added = kb.add_snippets(snippets=snippets, kb_name=kb_name)
            total_added += added
    print(f"种子 KB '{kb_name}' 构建完成，共 {total_added} chunks")


def _run_phase(kb_name: str, phase_name: str, results_out: Path) -> dict:
    """在指定 KB 上跑一轮测试，返回 summary。"""
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    case_correct = 0
    source_correct = 0
    learned_chunks = 0

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected_case = item["expected_case"]
        expected_source = item["expected_source"]

        ctrl = Controller(kb_client=KBClient(default_kb=kb_name))
        start = time.time()
        res = ctrl.process(question)
        elapsed = time.time() - start

        pred_case = res.get("case")
        pred_source = res.get("source")
        kb_added = res.get("kb_added_chunks", 0)
        learned_chunks += kb_added

        case_ok = pred_case == expected_case
        source_ok = pred_source == expected_source
        if case_ok:
            case_correct += 1
        if source_ok:
            source_correct += 1

        record = {
            "id": qid,
            "question": question,
            "expected_case": expected_case,
            "predicted_case": pred_case,
            "case_match": case_ok,
            "expected_source": expected_source,
            "predicted_source": pred_source,
            "source_match": source_ok,
            "kb_score": res.get("kb_score"),
            "llm_known_proba": res.get("llm_detection", {}).get("proba", {}).get("known"),
            "kb_added_chunks": kb_added,
            "needs_kb": item.get("needs_kb"),
            "elapsed": round(elapsed, 2),
            "answer": res.get("answer", "")[:300],
        }
        results.append(record)
        print(f"[{phase_name}] {qid}: case={pred_case}({'✓' if case_ok else '✗'}) "
              f"source={pred_source}({'✓' if source_ok else '✗'}) "
              f"kb_added={kb_added} time={elapsed:.1f}s", flush=True)

    total = len(dataset)
    summary = {
        "phase": phase_name,
        "kb_name": kb_name,
        "total": total,
        "case_accuracy": round(case_correct / total, 4),
        "source_accuracy": round(source_correct / total, 4),
        "case_correct": case_correct,
        "source_correct": source_correct,
        "learned_chunks": learned_chunks,
    }
    report = {"summary": summary, "results": results}
    with open(results_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return summary


def compare_reports(cold: dict, warm: dict) -> dict:
    """对比冷启动与热启动报告，分析路由迁移与学习效果。"""
    cold_map = {r["id"]: r for r in cold["results"]}
    warm_map = {r["id"]: r for r in warm["results"]}

    migrations = []
    for qid, cold_rec in cold_map.items():
        warm_rec = warm_map.get(qid)
        if not warm_rec:
            continue
        if cold_rec["predicted_case"] != warm_rec["predicted_case"]:
            migrations.append({
                "id": qid,
                "question": cold_rec["question"],
                "cold_case": cold_rec["predicted_case"],
                "warm_case": warm_rec["predicted_case"],
                "cold_source": cold_rec["predicted_source"],
                "warm_source": warm_rec["predicted_source"],
                "expected_case": cold_rec["expected_case"],
                "learned_chunks_cold": cold_rec["kb_added_chunks"],
            })

    return {
        "migrations": migrations,
        "migration_count": len(migrations),
        "cold_summary": cold["summary"],
        "warm_summary": warm["summary"],
    }


def run_three_phase():
    REPORT_DIR.mkdir(exist_ok=True)

    # 阶段 1：构建干净种子 KB
    kb_name = _fresh_kb_name()
    print(f"=== 阶段 1：构建种子 KB '{kb_name}' ===", flush=True)
    _build_seed_kb(kb_name)

    # 阶段 2：冷启动路由测试
    cold_report = REPORT_DIR / f"cold_{kb_name}.json"
    print(f"\n=== 阶段 2：冷启动路由测试 ===", flush=True)
    cold_summary = _run_phase(kb_name, "cold", cold_report)

    # 阶段 3：学习效能测试（已合并到冷启动中，因为 C/D 分支会写 KB）
    # 但我们可以专门统计学习内容
    learned_report = REPORT_DIR / f"learned_{kb_name}.json"
    with open(cold_report, "r", encoding="utf-8") as f:
        cold_data = json.load(f)
    with open(learned_report, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "phase": "learned",
                "kb_name": kb_name,
                "total_learned_chunks": cold_summary["learned_chunks"],
            },
            "learned_details": [
                {
                    "id": r["id"],
                    "kb_added_chunks": r["kb_added_chunks"],
                    "answer": r["answer"],
                }
                for r in cold_data["results"] if r["kb_added_chunks"] > 0
            ],
        }, f, ensure_ascii=False, indent=2)

    # 阶段 4：热启动路由测试（复用同一 KB）
    warm_report = REPORT_DIR / f"warm_{kb_name}.json"
    print(f"\n=== 阶段 3：热启动路由测试 ===", flush=True)
    warm_summary = _run_phase(kb_name, "warm", warm_report)

    # 对比分析
    comparison = compare_reports(cold_data, json.load(open(warm_report, encoding="utf-8")))
    comparison_report = REPORT_DIR / f"comparison_{kb_name}.json"
    with open(comparison_report, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 控制台输出总结
    print("\n" + "=" * 60)
    print("三阶段评估总结")
    print("=" * 60)
    print(f"测试 KB: {kb_name}")
    print(f"冷启动 case 准确率: {cold_summary['case_correct']}/{cold_summary['total']} = {cold_summary['case_accuracy']:.2%}")
    print(f"冷启动 source 准确率: {cold_summary['source_correct']}/{cold_summary['total']} = {cold_summary['source_accuracy']:.2%}")
    print(f"学习写入 chunks: {cold_summary['learned_chunks']}")
    print(f"热启动 case 准确率: {warm_summary['case_correct']}/{warm_summary['total']} = {warm_summary['case_accuracy']:.2%}")
    print(f"热启动 source 准确率: {warm_summary['source_correct']}/{warm_summary['total']} = {warm_summary['source_accuracy']:.2%}")
    print(f"路由迁移数量: {comparison['migration_count']}")
    for m in comparison["migrations"]:
        print(f"  - {m['id']}: {m['cold_case']} -> {m['warm_case']} (预期 {m['expected_case']})")
    print(f"\n报告目录: {REPORT_DIR}")


if __name__ == "__main__":
    run_three_phase()
