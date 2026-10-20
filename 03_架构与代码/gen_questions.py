#!/usr/bin/env python3
"""生成实验题库:
1. gold 交叉验证(2024报告的上年值 == 2023报告的本年值)
2. 公司池切分(30家入库=边界内, 其余=边界外, 固定种子)
3. 生成二元判断题(阈值取整, 正确答案两侧均衡) + 数值gold
输出: questions_raw.jsonl
"""
import json, random, re
from pathlib import Path

HERE = Path(__file__).parent
METRICS_LABELS = {"营业收入": "营业收入", "归属于上市公司股东的净利润": "归母净利润", "总资产": "总资产"}

def load():
    stocks = {s["code"]: s for s in json.loads((HERE / "stocks_sampled.json").read_text(encoding="utf-8"))}
    recs = [json.loads(l) for l in (HERE / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_code = {}
    for r in recs:
        by_code.setdefault(r["code"], {})[r["report_year"]] = r
    return stocks, by_code

def get_metric(r, label):
    m = r.get(label)
    if not m:
        return None, None
    return m.get("本年(report_year)"), m.get("上年(report_year-1)")

def main():
    stocks, by_code = load()
    verified, unverified = [], []
    for code, years in by_code.items():
        name = stocks.get(code, {}).get("name", code)
        vmap = {}
        ok = True
        for label in METRICS_LABELS:
            vals = {}
            for y, r in years.items():
                cur, _ = get_metric(r, label)
                if cur is not None:
                    vals[y] = cur
            vmap[label] = vals
        # cross-validate: 2024 report's prev == 2023 report's cur
        pairs = 0
        for label, vals in vmap.items():
            r24 = years.get(2024)
            r23 = years.get(2023)
            if r24 and r23:
                cur23, _ = get_metric(r23, label)
                _, prev23_from24 = None, None
                m24 = r24.get(label) or {}
                prev_from_24 = m24.get("上年(report_year-1)")
                if cur23 is not None and prev_from_24 is not None:
                    if abs(cur23 - prev_from_24) < 0.01:
                        pairs += 1
                    else:
                        ok = False
        entry = {"code": code, "name": name, "metrics": vmap, "cross_pairs": pairs}
        (verified if (ok and pairs >= 2) else unverified).append(entry)
    print(f"companies verified: {len(verified)}, unverified: {len(unverified)}")

    random.Random(2026).shuffle(verified)
    in_kb = verified[:30]
    out_kb = [c for c in verified[30:]] + unverified
    print(f"in-KB: {len(in_kb)}, out-KB: {len(out_kb)}")

    questions = []
    qid = 0
    for side, pool in (("in", in_kb), ("out", out_kb)):
        for c in pool:
            for label, short in METRICS_LABELS.items():
                for y, v in c["metrics"][label].items():
                    if v is None or v <= 0:
                        continue
                    yi = v / 1e8
                    thr = round(max(yi * random.Random(f"{c['code']}{y}{short}").choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3]), 3), 1)
                    correct = "超过" if yi > thr else "不超过"
                    qid += 1
                    questions.append({
                        "qid": f"{side}_{qid:04d}", "side": side, "code": c["code"], "company": c["name"],
                        "year": y, "metric": short, "value_yuan": v, "value_yi": round(yi, 2),
                        "threshold_yi": thr, "question": f"{c['name']}公司{y}年的{short}是否超过{thr}亿元？",
                        "gold": correct, "cross_verified": c["cross_pairs"] >= 2,
                    })
    random.Random(42).shuffle(questions)
    with open(HERE / "questions_raw.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    n_in = sum(1 for q in questions if q["side"] == "in")
    print(f"questions total: {len(questions)} (in: {n_in}, out: {len(questions) - n_in})")
    for q in questions[:3]:
        print(" ", q["qid"], q["question"], "->", q["gold"], f"({q['value_yi']}亿)")

if __name__ == "__main__":
    main()
