#!/usr/bin/env python3
"""
批量测试 Controller V0.1 的四象限路由准确率。
每个问题使用独立的 Controller 实例，避免上下文干扰。
"""
import json
import time
from pathlib import Path

from controller import Controller
from kb_client import KBClient

DATASET = Path(__file__).parent / "test_dataset.json"
REPORT = Path(__file__).parent / "test_report.json"
TEST_KB_NAME = "controller_test_kb_v3"


def run():
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    case_correct = 0
    source_correct = 0

    status_file = Path(__file__).parent / "batch_test_status.jsonl"
    status_file.write_text("", encoding="utf-8")

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected_case = item["expected_case"]
        expected_source = item["expected_source"]

        print(f"\n[{qid}] {question}", flush=True)
        ctrl = Controller(kb_client=KBClient(default_kb=TEST_KB_NAME))
        start = time.time()
        res = ctrl.process(question)
        elapsed = time.time() - start

        pred_case = res.get("case")
        pred_source = res.get("source")

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
            "needs_kb": item.get("needs_kb"),
            "elapsed": round(elapsed, 2),
            "answer": res.get("answer", "")[:200],
        }
        results.append(record)
        with open(status_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  case: {pred_case} (exp {expected_case}) {'✓' if case_ok else '✗'}", flush=True)
        print(f"  source: {pred_source} (exp {expected_source}) {'✓' if source_ok else '✗'}", flush=True)
        print(f"  kb_score={res.get('kb_score'):.3f}, llm_known={res.get('llm_detection', {}).get('proba', {}).get('known'):.3f}, time={elapsed:.1f}s", flush=True)

    total = len(dataset)
    summary = {
        "total": total,
        "case_accuracy": round(case_correct / total, 4),
        "source_accuracy": round(source_correct / total, 4),
        "case_correct": case_correct,
        "source_correct": source_correct,
    }

    report = {
        "summary": summary,
        "results": results,
    }

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"测试总数: {total}")
    print(f"case 准确率: {case_correct}/{total} = {summary['case_accuracy']:.2%}")
    print(f"source 准确率: {source_correct}/{total} = {summary['source_accuracy']:.2%}")
    print(f"报告已保存: {REPORT}")


if __name__ == "__main__":
    run()
