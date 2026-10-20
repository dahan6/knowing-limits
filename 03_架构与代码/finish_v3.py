#!/usr/bin/env python3
"""收官: 重新生成题题库(含batch2) → 合并探针分(旧题按文本复用/新题现场探测) → eb3重筛 → 最终 questions_clean.jsonl"""
import json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
sys.path.insert(0, str(HERE.parents[1] / "controller"))


def run(cmd):
    print(">>", cmd, flush=True)
    subprocess.run([sys.executable] + cmd.split(), check=True, cwd=str(HERE))


def main():
    # 1) 重生成题库(此时batch2 txt已到)
    run("rebuild_v3.py metrics")
    questions = [json.loads(l) for l in (CORPUS / "questions_raw2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # 2) 旧探针分(按题面文本匹配)
    old_proba = {}
    for l in (HERE / "screened.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("llm_known_proba") is not None:
                old_proba[r["question"]] = r["llm_known_proba"]

    from controller import Controller
    from kb_client import KBClient

    todo = [q for q in questions if q["question"] not in old_proba]
    print(f"total {len(questions)}, probe-reuse {len(questions) - len(todo)}, probe-new {len(todo)}", flush=True)

    # 3) 新题现场探测(仅probe, 不走全流程)
    probe_file = HERE / "probe_cache.jsonl"
    probed = {}
    if probe_file.exists():
        for l in probe_file.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                probed[r["question"]] = r["llm_known_proba"]
    todo = [q for q in todo if q["question"] not in probed]
    with probe_file.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            t0 = time.time()
            try:
                ctrl = Controller(kb_client=KBClient(default_kb="eb4_out"))
                qctx = ctrl.memory.contextualize(q["question"], ctrl.llm)
                det = ctrl.classifier.detect(qctx)
                proba = float(det.get("proba", {}).get("known", 0) or 0)
            except Exception as e:
                print(f"probe-error {q['qid']}: {e}", flush=True)
                proba = None
            probed[q["question"]] = proba
            f.write(json.dumps({"qid": q["qid"], "question": q["question"], "llm_known_proba": proba}) + "\n")
            f.flush()
            print(f"[probe {i}/{len(todo)}] {q['qid']} p={proba} ({time.time()-t0:.0f}s)", flush=True)

    # 4) 合并 → eb3 重筛
    kb = KBClient()
    kb.ensure_kb("eb4_out")
    bak = CORPUS / "questions_clean_v1.bak.jsonl"
    if not bak.exists() and (CORPUS / "questions_clean.jsonl").exists():
        bak.write_text((CORPUS / "questions_clean.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    n_in = n_out = 0
    with (CORPUS / "questions_clean.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            proba = probed.get(q["question"], old_proba.get(q["question"]))
            target = f"eb4_co_{q['code']}" if q["side"] == "in" else "eb4_out"
            try:
                res = kb.search(q["question"], kb_name=target, top_k=5)
                kb_score = float(res[0].get("raw_score", res[0].get("score", 0))) if res else 0.0
            except Exception:
                kb_score = None
            if kb_score is None or proba is None:
                verdict, case = "error", None
            else:
                kb_high, llm_high = kb_score >= 0.6, proba >= 0.7
                case = ("A" if llm_high else "B") if kb_high else ("C" if llm_high else "D")
                if q["side"] == "in" and kb_high and case in ("A", "B"):
                    verdict = "in_clean"
                elif q["side"] == "out" and kb_score < 0.3 and proba < 0.4 and case == "D":
                    verdict = "out_clean"
                else:
                    verdict = "ambiguous"
            rec = dict(q, kb_score=round(kb_score, 4) if kb_score is not None else None,
                       llm_known_proba=proba, case=case, verdict=verdict, target_kb=target)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_in += verdict == "in_clean"
            n_out += verdict == "out_clean"
    print(f"FINAL: in_clean={n_in} out_clean={n_out}", flush=True)


if __name__ == "__main__":
    main()
