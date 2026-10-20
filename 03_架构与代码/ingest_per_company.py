#!/usr/bin/env python3
"""一司一库: in公司各建 eb2_co_{code}(窗口切片), 外加空库 eb2_out。"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2] / "controller"))
from kb_client import KBClient

WINDOW_CAP, CHUNK = 80000, 1200

def window_text(txt):
    m = re.search(r"主要会计数据", txt)
    end = min(m.start() + 4000, len(txt)) if m else WINDOW_CAP
    return txt[:min(end, WINDOW_CAP)]

def main():
    qs = [json.loads(l) for l in (HERE / "questions_raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    in_codes = sorted({q["code"] for q in qs if q["side"] == "in"})
    kb = KBClient()
    kb.ensure_kb("eb2_out")
    for code in in_codes:
        name = f"eb2_co_{code}"
        kb.ensure_kb(name)
        n = 0
        for p in sorted((HERE / "txt").glob(f"{code}_*.txt")):
            text = window_text(p.read_text(encoding="utf-8", errors="ignore"))
            snippets = [{"source": p.stem, "content": c}
                        for c in [text[i:i+CHUNK] for i in range(0, len(text), CHUNK)] if len(c.strip()) > 200]
            kb.add_snippets(snippets, kb_name=name)
            n += len(snippets)
        print(f"{name}: {n} chunks", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
