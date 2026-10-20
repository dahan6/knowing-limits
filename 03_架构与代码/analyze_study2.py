#!/usr/bin/env python3
"""Study 2 正式分析: 问卷星两份导出 × 试次映射钥匙 → 质控+指标+检验"""
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
TRIALS = json.loads((HERE / "wjx_trial_map.json").read_text(encoding="utf-8"))
FILES = {"hard": "386509147_按文本_（硬答组）_134_134.xlsx",
         "governed": "386509176_按文本_（治理版）_136_136.xlsx"}


def advice_for(kind, t, cond):
    """该条件下此题的建议方向(None=拒答)与对错"""
    if kind == "in":
        return t.get("advice_dir"), t.get("advice_correct")
    if kind == "C":
        if cond == "hard":
            return t.get("advice_dir_hard"), t.get("advice_correct_hard")
        return t.get("advice_dir_infer"), t.get("advice_correct_infer")
    if cond == "hard":
        return t.get("advice_dir_hard"), t.get("advice_correct_hard")
    return None, None  # 治理版D型: 拒答


def parse(cond):
    df = pd.read_excel(HERE / FILES[cond], header=0)
    df.columns = [str(c) for c in df.columns]
    cols = df.columns.tolist()

    def find(sub, contains_all=None):
        for c in cols:
            if sub in c and all(k in c for k in (contains_all or [])):
                return c
        return None

    t_init = {i: find(f"[第{i}/28题]", ["初步判断"]) for i in range(1, 29)}
    t_final = {i: find(f"[第{i}/28题]", ["重看题目"]) for i in range(1, 29)}
    bp_cols = [c for c in cols if "基于目前的体验" in c]
    t1_cols = [find(f"阶段问卷T1·{j}/6") for j in range(1, 7)]
    t2_cols = [find(f"阶段问卷T2·{j}/6") for j in range(1, 7)]
    assert all(t_init.values()) and all(t_final.values()), "trial cols missing"
    assert all(t1_cols) and all(t2_cols) and len(bp_cols) == 3

    time_col = find("所用时间")
    recs = []
    for _, row in df.iterrows():
        tm = str(row[time_col])
        m = re.search(r"(\d+)", tm)
        secs = int(m.group(1)) if m else np.nan
        dcol = None
        for c in df.columns:
            if "提交答卷时间" in str(c):
                dcol = c
                break
        dstr = str(row[dcol]) if dcol else ""
        wave = 2 if ("9/28" in dstr or "9/29" in dstr or "9/30" in dstr or "10/" in dstr) else 1
        rec = dict(cond=cond, secs=secs, wave=wave,
                   age=row.get(find("年龄段"), None), gender=row.get(find("性别"), None),
                   edu=row.get(find("学历"), None), ai_freq=row.get(find("AI工具的频率"), None),
                   fin=row.get(find("财务报表的熟悉程度"), None))
        inits = {i: str(row[t_init[i]]) for i in range(1, 29)}
        finals = {i: str(row[t_final[i]]) for i in range(1, 29)}
        rec["inits"], rec["finals"] = inits, finals
        rec["bp"] = [str(row[c]) for c in bp_cols]
        rec["t1"] = [pd.to_numeric(row[c], errors="coerce") for c in t1_cols]
        rec["t2"] = [pd.to_numeric(row[c], errors="coerce") for c in t2_cols]
        recs.append(rec)
    return recs


def screen(rec):
    """预注册剔除规则, 返回(None, 原因)或(rec, None)"""
    if rec["secs"] is None or rec["secs"] < 300:
        return None, f"时长过短({rec['secs']}s)"
    if rec["secs"] > 5400:
        return None, f"时长过长({rec['secs']}s)"
    if len(set(rec["finals"].values())) == 1:
        return None, "终判全同(直线作答)"
    if len(set(rec["inits"].values())) == 1:
        return None, "初判全同(直线作答)"
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in rec["t2"]):
        return None, "量表缺失"
    return rec, None


def metrics(rec):
    cond = rec["cond"]
    r = dict(cond=cond, secs=rec["secs"], wave=rec["wave"],
             fin=float(pd.to_numeric(rec["fin"], errors="coerce")) if str(rec["fin"]).replace(".", "").isdigit() else np.nan)
    init_ok = fin_ok = fin_ok_in = fin_ok_C = fin_ok_D = 0
    adopt = adopt_n = 0
    err_adopt = err_n = 0
    woa = woa_n = 0
    ans_ok = ans_n = 0
    for i, t in enumerate(TRIALS, 1):
        kind = t["kind"]
        init, final = rec["inits"][i], rec["finals"][i]
        gold = t["gold"]
        adv_dir, adv_ok = advice_for(kind, t, cond)
        init_ok += init == gold
        fin_ok += final == gold
        if kind == "in":
            fin_ok_in += final == gold
        elif kind == "C":
            fin_ok_C += final == gold
        else:
            fin_ok_D += final == gold
        if adv_dir:  # 有建议的题
            ans_n += 1
            ans_ok += bool(adv_ok)
            adopt_n += 1
            adopt += final == adv_dir
            if not adv_ok:
                err_n += 1
                err_adopt += final == adv_dir
            if init != adv_dir:  # WOA适用
                woa_n += 1
                woa += 1 if final == adv_dir else 0
    t1, t2 = rec["t1"], rec["t2"]
    r.update(
        init_acc=init_ok / 28, final_acc=fin_ok / 28,
        acc_in=fin_ok_in / 14, acc_C=fin_ok_C / 9, acc_D=fin_ok_D / 5,
        adopt_rate=adopt / adopt_n if adopt_n else np.nan,
        err_adopt_rate=err_adopt / err_n if err_n else np.nan,
        woa=woa / woa_n if woa_n else np.nan,
        advice_rel=ans_ok / ans_n if ans_n else np.nan,
        t1_abi=np.nanmean(t1[0:2]), t1_int=np.nanmean(t2[2:5]) if False else np.nanmean(t1[2:5]), t1_ben=t1[5],
        t2_abi=np.nanmean(t2[0:2]), t2_int=np.nanmean(t2[2:5]), t2_ben=t2[5],
        d_abi=np.nanmean(t2[0:2]) - np.nanmean(t1[0:2]),
        d_int=np.nanmean(t2[2:5]) - np.nanmean(t1[2:5]),
        d_ben=t2[5] - t1[5],
        calib_err=abs(np.nanmean(t2[0:2]) / 7 - (ans_ok / ans_n if ans_n else 0)),
        bp1=rec["bp"][0], bp2=rec["bp"][1], bp3=rec["bp"][2],
    )
    return r


def main():
    all_rows, excl = [], []
    for cond in FILES:
        recs = parse(cond)
        kept = 0
        for rec in recs:
            ok, why = screen(rec)
            if ok:
                all_rows.append(metrics(rec))
                kept += 1
            else:
                excl.append((cond, why))
        print(f"{cond}: 收{len(recs)} 留{kept} 剔{len(recs)-kept}")
    from collections import Counter
    print("剔除原因:", dict(Counter(w for _, w in excl)))

    df = pd.DataFrame(all_rows)
    df.to_csv(HERE / "study2_subjects.csv", index=False, encoding="utf-8-sig")
    h = df[df.cond == "hard"]
    g = df[df.cond == "governed"]

    def cmp(name, a, b, higher_better=True):
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2) if len(a) > 1 and len(b) > 1 else np.nan
        print(f"{name:22s} hard={a.mean():.3f}±{a.std():.3f}  governed={b.mean():.3f}±{b.std():.3f}  "
              f"diff={a.mean()-b.mean():+.3f}  t={t:.2f} p={p:.4f} d={d:+.2f}")

    print(f"\n===== 组间检验 (n_hard={len(h)}, n_gov={len(g)}) =====")
    cmp("初始准确率", h.init_acc, g.init_acc)
    cmp("最终准确率", h.final_acc, g.final_acc)
    cmp("边界内准确率", h.acc_in, g.acc_in)
    cmp("C型题准确率", h.acc_C, g.acc_C)
    cmp("D型题准确率", h.acc_D, g.acc_D)
    cmp("错误采纳率", h.err_adopt_rate, g.err_adopt_rate)
    cmp("建议采纳率", h.adopt_rate, g.adopt_rate)
    cmp("WOA", h.woa, g.woa)
    cmp("T2能力感", h.t2_abi, g.t2_abi)
    cmp("T2诚信感", h.t2_int, g.t2_int)
    cmp("T2善意感", h.t2_ben, g.t2_ben)
    cmp("Δ能力", h.d_abi, g.d_abi)
    cmp("Δ诚信", h.d_int, g.d_int)
    cmp("校准误差", h.calib_err, g.calib_err)
    print("\n边界预测轨迹(治理版):")
    for i, c in enumerate(["bp1", "bp2", "bp3"], 1):
        print(f"  第{i}次: {g[c].value_counts().to_dict()}")
    print("边界预测轨迹(硬答组):")
    for i, c in enumerate(["bp1", "bp2", "bp3"], 1):
        print(f"  第{i}次: {h[c].value_counts().to_dict()}")
    print("\n建议实际可靠率: hard=%.3f governed=%.3f" % (h.advice_rel.mean(), g.advice_rel.mean()))


if __name__ == "__main__":
    main()
