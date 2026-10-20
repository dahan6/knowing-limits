#!/usr/bin/env python3
"""管线修复三件套:
1) 单位感知重解析 metrics → questions_raw v2 (修正603107万元/2条错行, 行级合法性校验)
2) 真表格锚点入库 eb4_co_* (锚点后须含营业收入+数字, 窗口[锚-200, 锚+3000])
用法: python rebuild_v3.py metrics|ingest
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"


def find_table_anchor(txt):
    for m in re.finditer(r"主要会计数据", txt):
        seg = txt[m.start(): m.start() + 2000]
        if "营业收入" in seg and re.search(r"\d{3}", seg):
            return m.start()
    return None


def parse_tables():
    """单位感知重解析: 返回 {(code,year): {metric: yi}}"""
    codes = sorted({p.name.split("_")[0] for p in (CORPUS / "txt").glob("*_20*.txt")})
    out = {}
    for code in codes:
        for y in (2023, 2024):
            p = CORPUS / "txt" / f"{code}_{y}.txt"
            if not p.exists():
                continue
            t = p.read_text(encoding="utf-8", errors="ignore")
            a = find_table_anchor(t)
            if a is None:
                continue
            seg = re.sub(r"\s", "", t[max(0, a - 300): a + 2600])
            mu = re.search(r"单位[:：]?(?:人民币)?(?:币种[:：]人民币)?(万元|百万元|元)", seg)
            unit = mu.group(1) if mu else "元"
            scale = {"元": 1e8, "万元": 1e4, "百万元": 1e2}[unit]
            vals = {}
            for label, key in (("营业收入", "rev"),
                               ("归属于上市公司股东的净利润", "nprofit"),
                               ("总资产", "assets")):
                mr = re.search(re.escape(label) + r"[（(]?[^0-9\-]{0,10}([\d,]+\.\d{2})", seg)
                if mr:
                    v = float(mr.group(1).replace(",", "")) / scale
                    if 0.05 <= v <= 5000:
                        vals[key] = round(v, 4)
            if len(vals) >= 2:
                out[(code, y)] = dict(vals, unit=unit)
    return out


def cmd_metrics():
    import random
    tables = parse_tables()
    print(f"parsed tables: {len(tables)}")
    stocks = {s["code"]: s for s in json.loads((CORPUS / "stocks_sampled.json").read_text(encoding="utf-8"))}
    stocks.update({s["code"]: s for s in json.loads((CORPUS / "stocks_batch2.json").read_text(encoding="utf-8"))})
    stocks.update({s["code"]: s for s in json.loads((CORPUS / "stocks_batch3.json").read_text(encoding="utf-8"))})
    # 公司级聚合(两年), 校验交叉一致性
    by_code = {}
    for (code, y), v in tables.items():
        by_code.setdefault(code, {})[y] = v
    verified = []
    for code, years in by_code.items():
        if 2023 in years and 2024 in years and len(years[2023]) >= 2 and len(years[2024]) >= 2:
            verified.append(code)
    print(f"companies with both years: {len(verified)}")
    frozen = (CORPUS / "questions_clean.jsonl")
    if frozen.exists():
        cur_in = {json.loads(l)["code"] for l in frozen.read_text(encoding="utf-8").splitlines()
                  if l.strip() and json.loads(l).get("side") == "in"}
        in_kb = [c for c in verified if c in cur_in]
    else:
        random.Random(2026).shuffle(verified)
        in_kb = verified[:30]
    LBL = {"rev": "营业收入", "nprofit": "归母净利润", "assets": "总资产"}
    questions, qid = [], 0
    for side, pool in (("in", in_kb), ("out", [c for c in stocks if c not in in_kb])):
        for code in pool:
            if code not in by_code:
                continue
            for y, v in sorted(by_code[code].items()):
                for key, lbl in LBL.items():
                    if key not in v:
                        continue
                    yi = v[key]
                    thr = round(yi * random.Random(f"{code}{y}{key}").choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3]), 1)
                    qid += 1
                    questions.append({
                        "qid": f"{side}_{qid:04d}", "side": side, "code": code,
                        "company": stocks[code]["name"], "year": y, "metric": lbl,
                        "value_yi": yi, "threshold_yi": thr,
                        "question": f"{stocks[code]['name']}公司{y}年的{lbl}是否超过{thr}亿元？",
                        "gold": "超过" if yi > thr else "不超过",
                    })
    random.Random(42).shuffle(questions)
    with (CORPUS / "questions_raw2.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    n_in = sum(1 for q in questions if q["side"] == "in")
    print(f"questions_raw2: {len(questions)} (in {n_in} / out {len(questions)-n_in})")


def cmd_ingest():
    import time
    sys.path.insert(0, str(HERE.parents[1] / "controller"))
    from kb_client import KBClient
    kb = KBClient()
    kb.ensure_kb("eb4_out")
    codes = sorted({json.loads(l)["code"]
                    for l in (CORPUS / "questions_raw2.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(l)["side"] == "in"})
    for code in codes:
        name = f"eb4_co_{code}"
        kb.ensure_kb(name)
        n = 0
        for p in sorted((CORPUS / "txt").glob(f"{code}_*.txt")):
            t = p.read_text(encoding="utf-8", errors="ignore")
            a = find_table_anchor(t)
            if a is None:
                continue
            win = t[max(0, a - 200): a + 3000]
            chunks = [win[i:i + 1100] for i in range(0, len(win), 1100)]
            def dense(c):
                s = re.sub(r"\s", "", c)
                return len(s) >= 60 and len(re.findall(r"\d", s)) >= 10
            snippets = [{"source": p.stem, "content": c} for c in chunks if dense(c)]
            kb.add_snippets(snippets, kb_name=name)
            n += len(snippets)
        print(f"{name}: {n} chunks", flush=True)
    print("DONE")


if __name__ == "__main__":
    {"metrics": cmd_metrics, "ingest": cmd_ingest}[sys.argv[1]]()
