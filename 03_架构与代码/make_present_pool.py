#!/usr/bin/env python3
"""呈现池(方案B): in=clean∩bank∩verified / out=clean∩bank∩有方向硬答(携带advice_correct)"""
import json
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"

clean = [json.loads(l) for l in (CORPUS / "questions_clean.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
bank = json.loads((HERE / "responses_bank.json").read_text(encoding="utf-8"))
verified = {}
for l in (HERE / "pregen.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l)
        verified[r["qid"]] = bool(r.get("verified"))
hard2 = {}
for l in (HERE / "hard2.jsonl").read_text(encoding="utf-8").splitlines():
    if l.strip():
        r = json.loads(l)
        hard2[r["qid"]] = r

pool = []
for q in clean:
    if q["qid"] not in bank:
        continue
    if q["side"] == "in":
        if q["verdict"] == "in_clean" and verified.get(q["qid"]):
            pool.append(q)
    else:
        h = hard2.get(q["qid"])
        if q["verdict"] == "out_clean" and h and h.get("hard_dir") and "ERROR" not in h.get("hard", ""):
            q = dict(q, advice_correct=(h["hard_dir"] == q["gold"]))
            pool.append(q)

n_in = sum(1 for q in pool if q["side"] == "in")
n_out = len(pool) - n_in
n_out_wrong = sum(1 for q in pool if q["side"] == "out" and not q.get("advice_correct"))
with (CORPUS / "questions_present.jsonl").open("w", encoding="utf-8") as f:
    for q in pool:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"presentation pool: {len(pool)} (in {n_in} / out {n_out}, out中硬答错误 {n_out_wrong})")
