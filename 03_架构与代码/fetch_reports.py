#!/usr/bin/env python3
"""从巨潮资讯下载样本公司2023/2024年报PDF。用法: python fetch_reports.py [--limit N]"""
import argparse, json, re, time, urllib.parse, urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "pdf"
OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "X-Requested-With": "XMLHttpRequest",
      "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice"}

def post(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(), headers=UA)
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace"))

def get_orgid(code):
    d = post("http://www.cninfo.com.cn/new/information/topSearch/detailOfQuery", {"keyWord": code, "maxSecNum": 10, "maxListLen": 5})
    for it in d.get("keyBoardList", []):
        if it.get("code") == code:
            return it.get("orgId")
    return None

def fetch_reports(code, orgid):
    column = "sse" if code.startswith(("6", "9")) else "szse"
    d = post("http://www.cninfo.com.cn/new/hisAnnouncement/query", {
        "pageNum": 1, "pageSize": 30, "column": column, "tabName": "fulltext",
        "plate": "", "stock": f"{code},{orgid}", "searchkey": "", "secid": "",
        "category": "category_ndbg_szsh", "trade": "", "seDate": "2024-01-01~2025-12-31",
        "sortName": "", "sortType": "", "isHLtitle": "true"})
    out = []
    for ann in (d.get("announcements") or []):
        title = re.sub(r"<[^>]+>", "", ann.get("announcementTitle", ""))
        if "年度报告" in title and all(k not in title for k in ("摘要", "英文", "已取消", "更新后")):
            year = ann.get("announcementTime", 0)
            out.append({"title": title, "url": "http://static.cninfo.com.cn/" + ann["adjunctUrl"],
                        "code": code, "ts": year})
    return out

def download(url, dest):
    req = urllib.request.Request(url, headers=UA)
    data = urllib.request.urlopen(req, timeout=120).read()
    Path(dest).write_bytes(data)
    return len(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    stocks = json.loads(Path("stocks_batch3.json").read_text(encoding="utf-8"))
    if args.limit:
        stocks = stocks[: args.limit]
    ok = fail = 0
    for s in stocks:
        code = s["code"]
        try:
            orgid = get_orgid(code)
            if not orgid:
                print(f"[{code}] no orgId", flush=True); fail += 1; continue
            anns = fetch_reports(code, orgid)
            got = []
            for a in anns:
                m = re.search(r"20(23|24)", a["title"])
                if not m:
                    continue
                year = m.group(0)
                dest = OUT / f"{code}_{year}.pdf"
                if dest.exists() and dest.stat().st_size > 10000:
                    got.append(year); continue
                size = download(a["url"], dest)
                got.append(year)
                print(f"[{code}] {a['title']} -> {size//1024}KB", flush=True)
                time.sleep(1.5)
            if got:
                ok += 1
            time.sleep(1.2)
        except Exception as e:
            print(f"[{code}] ERROR {e}", flush=True); fail += 1
    print(f"DONE ok={ok} fail={fail} pdfs={len(list(OUT.glob('*.pdf')))}")

if __name__ == "__main__":
    main()
