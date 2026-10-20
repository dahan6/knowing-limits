#!/usr/bin/env python3
"""导出问卷试次映射: 第i题 -> qid/side/advice方向与对错(按条件)"""
import json, random
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
present = [json.loads(l) for l in (CORPUS / "questions_present.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
bank = json.loads((HERE / "responses_bank.json").read_text(encoding="utf-8"))
cans = {json.loads(l)["qid"]: json.loads(l) for l in (HERE / "c_answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
qs_all = {q["qid"]: q for q in (json.loads(l) for l in (CORPUS / "questions_clean.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

rng = random.Random(2026)
ins = rng.sample([q for q in present if q["side"] == "in"], 14)
douts = rng.sample([q for q in present if q["side"] == "out"], 5)
trials = [("in", q) for q in ins] + [("C", qs_all[qid]) for qid in cans.keys()] + [("D", q) for q in douts]
rng.shuffle(trials)

import re


def direction(text):
    m = re.search(r"推理结论[:：]|结论[:：]", text)
    seg = text[m.start():m.start() + 200] if m else text[-200:]
    neg = re.search(r"没有超过|未超过|不超过|未能超过|低于|小于", seg)
    pos = (not neg) and re.search(r"超过|远超|高于", seg)
    return "不超过" if neg else ("超过" if pos else None)


METRIC_FULL = {"营业收入": "营业收入", "归母净利润": "归属于母公司股东的净利润", "总资产": "总资产"}


def refuse_dir(q):
    return None  # 拒答无方向


out = []
for i, (kind, t) in enumerate(trials, 1):
    rec = dict(trial=i, kind=kind, qid=t["qid"], question=t["question"], gold=t["gold"],
               company=t["company"], year=t["year"], metric=t["metric"])
    if kind == "in":
        adv = bank[t["qid"]]["hard"]
        rec.update(advice_dir=direction(adv), advice_correct=(direction(adv) == t["gold"]))
    elif kind == "C":
        c = cans.get(t["qid"])
        rec.update(advice_dir_hard=c["hard_dir"], advice_correct_hard=(c["hard_dir"] == t["gold"]),
                   advice_dir_infer=c["infer_dir"], advice_correct_infer=(c["infer_dir"] == t["gold"]))
    else:
        h3 = None
        for l in (HERE / "hard3.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip() and json.loads(l)["qid"] == t["qid"]:
                h3 = json.loads(l)
        rec.update(advice_dir_hard=h3["hard_dir"] if h3 else None,
                   advice_correct_hard=(h3["hard_dir"] == t["gold"]) if h3 else None)
    out.append(rec)

(HERE / "wjx_trial_map.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
kinds = "".join(r["kind"] if r["kind"] != "in" else "内" for r in out)
print(f"trial map saved: {len(out)} trials, 序列: {kinds}")
nc = sum(1 for r in out if r["kind"] == "C" and r.get("advice_correct_infer"))
print(f"C型推理方向正确: {nc}/{sum(1 for r in out if r['kind']=='C')}")
