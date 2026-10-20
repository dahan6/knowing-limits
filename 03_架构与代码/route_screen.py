#!/usr/bin/env python3
"""刺激路由预筛 v3: 容错 + 断点续跑 + 单题硬超时(防挂起) + llama 自愈。

用法: python route_screen.py --kb exp_kb_v1 --in candidates.jsonl --out screened.jsonl
环境: ENABLE_WEB_SEARCH=false (方法论红线); KB_HIT_* 可标定
"""
import argparse
import json
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parents[2] / "controller"
sys.path.insert(0, str(CONTROLLER_DIR))

from controller import Controller  # noqa: E402
from kb_client import KBClient  # noqa: E402

QUESTION_TIMEOUT = 300
LLAMA_HEALTH = "http://127.0.0.1:8080/health"
LLAMA_RESTART = "/home/adazz18/ai-training/glm-4.5-air/start_server.sh"

KB_HIT_HIGH = 0.6
KB_HIT_LOW = 0.3
LLM_KNOWN_LOW = 0.4


class QTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise QTimeout()


signal.signal(signal.SIGALRM, _alarm)


def verdict_of(kb_score, proba, case):
    if kb_score >= KB_HIT_HIGH and case in ("A", "B"):
        return "in_clean"
    if kb_score < KB_HIT_LOW and proba < LLM_KNOWN_LOW and case == "D":
        return "out_clean"
    return "ambiguous"


def now():
    return datetime.now().strftime("%H:%M:%S")


def llama_ok():
    try:
        subprocess.run(["curl", "-s", "--max-time", "8", LLAMA_HEALTH],
                       capture_output=True, timeout=12)
        return True
    except Exception:
        return False


def llama_restart():
    print(f"[{now()}] !! restarting llama-server", flush=True)
    subprocess.run(["pkill", "-f", "llama-serve[r]"])
    time.sleep(3)
    subprocess.Popen(["setsid", "nohup", "bash", LLAMA_RESTART],
                     stdout=open("/home/adazz18/llama_restart.log", "w"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(90)
    return llama_ok()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    cands = [json.loads(l) for l in Path(args.infile).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.outfile)
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(l)["qid"])
            except Exception:
                pass
    todo = [c for c in cands if c["qid"] not in done]
    print(f"[{now()}] total {len(cands)}, done {len(done)}, todo {len(todo)}", flush=True)

    consecutive_timeout = 0
    with out.open("a", encoding="utf-8") as f:
        for i, item in enumerate(todo, 1):
            t0 = time.time()
            err = None
            try:
                signal.alarm(QUESTION_TIMEOUT)
                try:
                    ctrl = Controller(kb_client=KBClient(default_kb=args.kb))
                    res = ctrl.process(item["question"])
                finally:
                    signal.alarm(0)
                kb_score = float(res.get("kb_score", 0) or 0)
                proba = float((res.get("llm_detection") or {}).get("proba", {}).get("known", 0) or 0)
                case = res.get("case", "?")
                v = verdict_of(kb_score, proba, case)
                rec = dict(item, kb_score=round(kb_score, 4), llm_known_proba=round(proba, 4),
                           case=case, verdict=v, answer_head=str(res.get("answer", ""))[:200])
                consecutive_timeout = 0
            except QTimeout:
                err = f"timeout{QUESTION_TIMEOUT}s"
                consecutive_timeout += 1
                v = "error"
                rec = dict(item, kb_score=None, llm_known_proba=None, case=None, verdict="error", error=err)
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:120]}"
                v = "error"
                rec = dict(item, kb_score=None, llm_known_proba=None, case=None, verdict="error", error=err)
                traceback.print_exc()
            elapsed = round(time.time() - t0, 1)
            rec["elapsed"] = elapsed
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{now()}] [{i}/{len(todo)}] {item['qid']} case={rec.get('case')} "
                  f"kb={rec.get('kb_score')} p={rec.get('llm_known_proba')} -> {v} ({elapsed}s)", flush=True)
            if err:
                print(f"[{now()}] ERROR {item['qid']}: {err}", flush=True)
            if consecutive_timeout >= 2:
                if not llama_ok():
                    llama_restart()
                else:
                    print(f"[{now()}] llama health ok, timeouts may be KB/其它; 继续", flush=True)
                consecutive_timeout = 0
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
