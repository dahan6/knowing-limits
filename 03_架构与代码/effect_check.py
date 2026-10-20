#!/usr/bin/env python3
"""效果检验 v2: 判分识别否定式与未答类别。
用法: python effect_check.py [N_per_side]  (默认8; 只重判已有文件用 --judge)
"""
import json, random, re, sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "controller"))

NEG = ("没有超过", "未超过", "不超过", "未能超过", "低于", "小于")
POS = ("超过了", "远超", "高于", "超过")
UNANS = ("无法确定", "无法判断", "无法获取", "无法提供", "不能确定", "未包含", "未提供")


def judge(answer, gold):
    concl = answer[-350:]
    if any(k in concl for k in UNANS):
        return "unanswered"
    import re as _re
    neg = _re.search(r"没有超过|未超过|不超过|未能超过|远低于|低于|小于", concl)
    pos = (not neg) and _re.search(r"超过|远超|高于", concl)
    if neg:
        d = "不超过"
    elif pos:
        d = "超过"
    else:
        return "ambiguous"
    return "hit" if d == gold else "miss"


def run_generation(n):
    from controller import Controller
    from kb_client import KBClient
    clean = [json.loads(l) for l in (HERE / "corpus" / "questions_clean.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ins = [q for q in clean if q["verdict"] == "in_clean"]
    outs = [q for q in clean if q["verdict"] == "out_clean"]
    rng = random.Random(23)
    sample = rng.sample(ins, n) + rng.sample(outs, n)
    rng.shuffle(sample)
    out_path = HERE / "corpus" / "effect_check.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(sample, 1):
            target = q.get("target_kb") or (f"eb4_co_{q['code']}" if q["side"] == "in" else "eb4_out")
            t0 = time.time()
            try:
                ctrl = Controller(kb_client=KBClient(default_kb=target))
                res = ctrl.process(q["question"])
                answer, case, kb = str(res.get("answer", "")), res.get("case"), res.get("kb_score")
            except Exception as e:
                answer, case, kb = f"ERROR {type(e).__name__}: {e}", None, None
            rec = dict(qid=q["qid"], side=q["side"], question=q["question"], gold=q["gold"],
                       target_kb=target, case=case, kb_score=kb,
                       elapsed=round(time.time() - t0, 1), answer=answer)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(sample)}] {q['qid']} case={case} kb={round(kb,3) if isinstance(kb,float) else kb} ({rec['elapsed']}s)", flush=True)


def report():
    rows = [json.loads(l) for l in (HERE / "corpus" / "effect_check.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print("\n===== in 题判分 =====")
    stat = {"hit": 0, "miss": 0, "unanswered": 0, "ambiguous": 0}
    for r in rows:
        if r["side"] != "in":
            continue
        stat[judge(r["answer"], r["gold"])] += 1
    print(f"对 {stat['hit']} / 错 {stat['miss']} / 未答 {stat['unanswered']} / 模糊 {stat['ambiguous']}  (n={sum(stat.values())})")
    refuse = 0
    n_out = 0
    for r in rows:
        if r["side"] != "out":
            continue
        n_out += 1
        if any(k in r["answer"] for k in ("无法", "超出", "知识范围", "不知道", "不能确定", "没有相关", "难以", "建议")):
            refuse += 1
    print(f"===== out 拒答率: {refuse}/{n_out} ======")
    for r in [x for x in rows if x["side"] == "in"][:3]:
        print(f"\n[{judge(r['answer'], r['gold'])}] Q: {r['question']}")
        print(f"  A末段: ...{r['answer'][-160:].replace(chr(10), ' ')}")


def main():
    if "--judge" in sys.argv:
        report()
        return
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run_generation(n)
    report()


if __name__ == "__main__":
    main()
