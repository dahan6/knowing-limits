#!/usr/bin/env python3
"""抽样实验用股票池: A股, 总市值20-150亿(天然排除沪深300与微盘), 剔除ST, 随机50家, 固定种子。"""
import json, random, urllib.request

URL = ("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=6000&po=1&np=1&fltt=2&invt=2"
       "&fid=f20&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12,f14,f20")

def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())
    total = data["data"]["total"]
    stocks = data["data"]["diff"]
    print(f"total listed: {total}, fetched: {len(stocks)}")
    pool = []
    for s in stocks:
        cap = s.get("f20")
        name, code = s.get("f14", ""), s.get("f12", "")
        if not isinstance(cap, (int, float)) or cap == "-":
            continue
        cap_yi = cap / 1e8
        if 20 <= cap_yi <= 150 and "ST" not in name and "退" not in name:
            pool.append({"code": code, "name": name, "cap_yi": round(cap_yi, 1)})
    print(f"eligible pool (20-150亿, non-ST): {len(pool)}")
    random.Random(2026).shuffle(pool)
    picked = pool[:50]
    with open("stocks_sampled.json", "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=1)
    print(f"sampled 50 -> stocks_sampled.json")
    for p in picked[:5]:
        print(" ", p["code"], p["name"], p["cap_yi"], "亿")

if __name__ == "__main__":
    main()
