#!/usr/bin/env python3
"""快速重筛: 一司一库 kb_score(纯检索) + 复用第一轮 llm_known_proba, 重算 case/verdict。"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "controller"))
from kb_client import KBClient  # noqa: E402

HIGH, LOW, PL = 0.6, 0.3, 0.4

def main():
    kb = KBClient()
    kb.ensure_kb("eb2_out")
    rows = [json.loads(l) for l in (HERE / "screened.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = HERE / "corpus" / "questions_clean.jsonl"
    n_in = n_out = 0
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            q = r["question"]
            target = f"eb2_co_{r['code']}" if r["side"] == "in" else "eb2_out"
            try:
                res = kb.search(q, kb_name=target, top_k=5)
                kb_score = float(res[0].get("raw_score", res[0].get("score", 0))) if res else 0.0
            except Exception as e:
                kb_score = None
                print(f"ERR {r[qid]}: {e}", flush=True)
            proba = r.get("llm_known_proba")
            if kb_score is None:
                verdict = "error"
                case = None
            else:
                kb_high = kb_score >= HIGH
                llm_high = (proba or 0) >= 0.7
                case = ("A" if llm_high else "B") if kb_high else ("C" if llm_high else "D")
                if r["side"] == "in" and kb_high and case in ("A", "B"):
                    verdict = "in_clean"
                elif r["side"] == "out" and kb_score < LOW and (proba or 1) < PL and case == "D":
                    verdict = "out_clean"
                else:
                    verdict = "ambiguous"
            rec = {k: r.get(k) for k in ("qid", "side", "code", "company", "year", "metric",
                                          "value_yuan", "value_yi", "threshold_yi", "question", "gold", "cross_verified")}
            rec.update(kb_score=round(kb_score, 4) if kb_score is not None else None,
                       llm_known_proba=proba, case=case, verdict=verdict, target_kb=target)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if verdict == "in_clean":
                n_in += 1
            elif verdict == "out_clean":
                n_out += 1
    print(f"in_clean={n_in} out_clean={n_out}")

    new_rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    def qstats(side):
        vals = sorted(r["kb_score"] for r in new_rows if r["side"] == side and r["kb_score"] is not None)
        return vals[:1] + [vals[len(vals)*i//4] for i in (1, 2, 3)] + vals[-1:]
    print("in  kb (min/25/50/75/max):", [round(v, 3) for v in qstats("in")])
    print("out kb (min/25/50/75/max):", [round(v, 3) for v in qstats("out")])

if __name__ == "__main__":
    main()
