#!/usr/bin/env python3
"""hard3: 丰富版硬答(结构化自信幻觉), 长度对齐in题回答"""
import json, re, sys, time
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
sys.path.insert(0, str(HERE.parents[1] / "controller"))

PROMPT = (
    "你是一个企业经营信息分析助手。请对下面的问题给出结构化的分析回答，格式如下：\n"
    "1. 行业与公司分析：用两三句话分析该公司所处行业的经营特点\n"
    "2. 数据依据：给出你记忆中该公司的相关财务数据（可给出大致数字）\n"
    "3. 结论：以\"结论：\"开头，明确写出\"超过\"或\"不超过\"其一\n"
    "要求：直接作答，不要拒绝，不要说无法确定。\n\n问题：{q}"
)


def direction(text):
    m = re.search(r"结论[:：]", text)
    seg = text[m.start():m.start() + 200] if m else text[-200:]
    neg = re.search(r"没有超过|未超过|不超过|未能超过|低于|小于", seg)
    pos = (not neg) and re.search(r"超过|远超|高于", seg)
    return "不超过" if neg else ("超过" if pos else None)


def main():
    from llm_client import LLMClient
    llm = LLMClient()
    outs = [json.loads(l) for l in (CORPUS / "questions_clean.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l).get("verdict") == "out_clean"]
    out_path = HERE / "hard3.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["qid"])
    stat = {"wrong": 0, "right": 0, "nodir": 0}
    with out_path.open("a", encoding="utf-8") as f:
        for i, q in enumerate([x for x in outs if x["qid"] not in done], 1):
            t0 = time.time()
            try:
                ans = llm._chat([{"role": "user", "content": PROMPT.format(q=q["question"])}], temperature=0.7)
            except Exception as e:
                ans = f"ERROR {e}"
            d = direction(ans)
            if d is None:
                stat["nodir"] += 1
            elif d == q["gold"]:
                stat["right"] += 1
            else:
                stat["wrong"] += 1
            f.write(json.dumps({"qid": q["qid"], "hard": ans, "hard_dir": d, "gold": q["gold"]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}] {q['qid']} dir={d} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE stat={json.dumps(stat)}", flush=True)
    bank_path = HERE / "responses_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    for l in out_path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r["qid"] in bank and "ERROR" not in r["hard"] and r.get("hard_dir"):
                bank[r["qid"]]["hard"] = r["hard"]
    bank_path.write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"bank merged: {len(bank)}", flush=True)


if __name__ == "__main__":
    main()
