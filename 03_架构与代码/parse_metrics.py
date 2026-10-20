#!/usr/bin/env python3
"""从年报文本解析主要会计数据: 营业收入/归母净利润/总资产, 输出 metrics.jsonl"""
import json, re, sys
from pathlib import Path

ROW_PAT = re.compile(r"^(营业收入|归属于上市公司股东的净利润|归属于上市公司股东的扣除.*?净利润|总资产|归属于上市公司股东的净资产)\s*(（[^）]*）)?\s+(-?[\d,]+\.\d{2}|—|-)\s+(-?[\d,]+\.\d{2}|—|-)")
NUM_PAT = re.compile(r"^-?[\d,]+\.\d{2}$")

def to_f(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None

def parse(txt_path):
    text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"主要会计数据", text)
    if not m:
        return None
    window = text[m.start(): m.start() + 6000]
    out = {}
    for line in window.splitlines():
        mm = ROW_PAT.match(line.strip())
        if mm:
            label, cur, prev = mm.group(1), to_f(mm.group(3)), to_f(mm.group(4))
            if cur is not None and label not in out:
                out[label] = {"cur": cur, "prev": prev}
    return out

def main():
    recs, miss = [], 0
    for p in sorted(Path("txt").glob("*.txt")):
        code, year = p.stem.split("_")
        got = parse(p)
        if not got or "营业收入" not in got:
            miss += 1
            continue
        report_year = int(year)
        rec = {"code": code, "report_year": report_year}
        for label, v in got.items():
            rec[label] = {"本年(report_year)": v["cur"], "上年(report_year-1)": v["prev"]}
        recs.append(rec)
    with open("metrics.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    full = sum(1 for r in recs if r.get("营业收入", {}).get("本年(report_year)") is not None
               and r.get("营业收入", {}).get("上年(report_year-1)") is not None)
    print(f"parsed {len(recs)} reports, fully-paired rev rows: {full}, miss-section: {miss}")
    for r in recs[:3]:
        rev = r.get("营业收入", {})
        print(" ", r["code"], r["report_year"], "rev cur:", rev.get("本年(report_year)"), "prev:", rev.get("上年(report_year-1)"))

if __name__ == "__main__":
    main()
