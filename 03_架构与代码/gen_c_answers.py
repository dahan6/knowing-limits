#!/usr/bin/env python3
"""C型题回答生成: 推理披露版(治理) + 自信版(硬答组)"""
import json, re, sys, time
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
sys.path.insert(0, str(HERE.parents[1] / "controller"))

PICK = ["out_0181", "out_0214", "out_0203", "out_0175", "out_0207", "out_0187", "out_0198", "out_0184", "out_0179"]

INFER_PROMPT = (
    "你是企业经营分析助手。你的资料库中没有收录{company}的{year}年年度报告。"
    "请基于你的行业知识对该问题进行推理估计,并按以下格式回答:\n"
    "1. 先明确声明:你缺乏该公司的直接数据,以下是基于行业知识的推理,不是检索到的事实\n"
    "2. 给出推理过程(该公司所处行业的一般财务特征、结合题中市值等信息的推断)\n"
    "3. 以\"推理结论:\"开头给出明确判断(超过/不超过)\n"
    "问题:{q}"
)
HARD_PROMPT = (
    "你是一个企业经营信息分析助手。请对下面的问题给出结构化的分析回答,格式如下:\n"
    "1. 行业与公司分析:用两三句话分析该公司所处行业的经营特点\n"
    "2. 数据依据:给出你记忆中该公司的相关财务数据(可给出大致数字)\n"
    "3. 结论:以\"结论:\"开头,明确写出\"超过\"或\"不超过\"其一\n"
    "要求:直接作答,不要拒绝,不要说无法确定。\n\n问题:{q}"
)


def direction(text, key="推理结论"):
    m = re.search(key + r"[:：]", text) or re.search(r"结论[:：]", text)
    seg = text[m.start():m.start() + 200] if m else text[-200:]
    neg = re.search(r"没有超过|未超过|不超过|未能超过|低于|小于", seg)
    pos = (not neg) and re.search(r"超过|远超|高于", seg)
    return "不超过" if neg else ("超过" if pos else None)


def main():
    from llm_client import LLMClient
    llm = LLMClient()
    qs = {q["qid"]: q for q in (json.loads(l) for l in (CORPUS / "questions_clean.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    out_path = HERE / "c_answers.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["qid"])
    with out_path.open("a", encoding="utf-8") as f:
        for i, qid in enumerate(PICK, 1):
            if qid in done:
                continue
            q = qs[qid]
            t0 = time.time()
            try:
                infer = llm._chat([{"role": "user", "content": INFER_PROMPT.format(company=q["company"], year=q["year"], q=q["question"])}], temperature=0.6)
                hard = llm._chat([{"role": "user", "content": HARD_PROMPT.format(q=q["question"])}], temperature=0.7)
            except Exception as e:
                print(f"ERR {qid}: {e}", flush=True)
                continue
            rec = {"qid": qid, "question": q["question"], "gold": q["gold"],
                   "infer": infer, "infer_dir": direction(infer),
                   "hard": hard, "hard_dir": direction(hard, key="结论"),
                   "proba": q.get("llm_known_proba")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(PICK)}] {qid} infer={rec['infer_dir']} hard={rec['hard_dir']} ({time.time()-t0:.0f}s)", flush=True)
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"DONE {len(rows)}条; infer有方向{sum(1 for r in rows if r['infer_dir'])}, hard有方向{sum(1 for r in rows if r['hard_dir'])}", flush=True)


if __name__ == "__main__":
    main()
