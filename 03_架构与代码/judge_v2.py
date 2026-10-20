#!/usr/bin/env python3
"""结论句判分器: 只看最后一个'结论'段(或末200字)定方向, 无方向才算未答"""
import json, re
from pathlib import Path

rows = [json.loads(l) for l in (Path(__file__).parent / "corpus" / "effect_check.jsonl").read_text(encoding="utf-8").splitlines()]


def conclude(answer):
    idx = answer.rfind("结论")
    seg = answer[idx: idx + 300] if idx >= 0 else answer[-200:]
    return seg if seg.strip() else answer[-200:]


def judge(answer, gold):
    c = conclude(answer)
    neg = re.search(r"没有超过|未超过|不超过|未能超过|低于|小于", c)
    pos = (not neg) and re.search(r"超过|远超|高于", c)
    if neg:
        d = "不超过"
    elif pos:
        d = "超过"
    else:
        return "unanswered"
    return "hit" if d == gold else "miss"


stat = {"hit": 0, "miss": 0, "unanswered": 0}
for r in rows:
    if r["side"] != "in":
        continue
    v = judge(r["answer"], r["gold"])
    stat[v] += 1
    if v != "hit":
        print(f"[{v}] {r['question'][:44]}")
        print("   结论:", conclude(r["answer"]).replace(chr(10), " ")[:110])
print("\n结论句判分:", stat)
