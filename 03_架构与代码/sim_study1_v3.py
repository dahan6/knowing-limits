#!/usr/bin/env python3
"""Study 1 v3: 两个扩展
A) 问责溢价(情景分析, 透明标注): 被采纳的错误建议额外扣罚 b(跟AI犯错问责更重)
B) 模型能力稳健性扫描: 以能力缩放k作用于所有>0.5的正确率锚点 (p=0.5+k*(p_base-0.5), 截断[0.5,0.99])
"""
import csv, random
from pathlib import Path

P_IN = 0.90
P_HC = 0.778
P_HD = 0.545
P_DIS = 0.667
P_OWN = 0.50
FRAC_C = 9 / 14
H_CREDIT = 0.10
G_MULT = 1.5
TAU0 = 0.5
BETA = 6.0
LAMBDAS = [0.1, 0.3, 0.5]
OUT_RATIOS = [0.2, 0.35, 0.5]
MECHS = ["hard", "refuse", "guide", "governed"]
LOSS = [1, 2, 3]
BLAMES = [0, 1]
KS = [0.5, 0.75, 1.0, 1.25, 1.5]
PERIODS = 100
REPS = 1000


def scale(p, k):
    return min(0.99, max(0.5, 0.5 + k * (p - 0.5)))


def p_adopt(tau):
    return 1.0 / (1.0 + pow(2.718281828, -BETA * (tau - 0.5)))


def run_once(mech, orat, lam, m, b, k, rng):
    tau = TAU0
    score = 0.0
    advice_n = advice_ok = err_adopt = 0
    for t in range(PERIODS):
        is_out = rng.random() < orat
        is_c = rng.random() < FRAC_C
        if (not is_out) or mech == "hard" or (mech == "governed" and is_c):
            if not is_out:
                ok = rng.random() < scale(P_IN, k)
            elif mech == "hard":
                ok = rng.random() < scale(P_HC if is_c else P_HD, k)
            else:
                ok = rng.random() < scale(P_DIS, k)
            adopted = rng.random() < p_adopt(tau)
            if adopted:
                advice_n += 1
                advice_ok += ok
                if not ok:
                    err_adopt += 1
                score += 1.0 if ok else -(m + b)
            else:
                own = rng.random() < P_OWN
                score += 1.0 if own else -m
            tau = min(1.0, tau + lam * (1 - tau)) if ok else max(0.0, tau - lam * tau)
        else:
            own = rng.random() < P_OWN
            score += 1.0 if own else -m
            if mech == "guide":
                tau = min(1.0, tau + G_MULT * lam * H_CREDIT * (1 - tau))
    return score / PERIODS, err_adopt, tau


def part_a():
    rows = []
    for mech in MECHS:
        for orat in OUT_RATIOS:
            for lam in LAMBDAS:
                for m in LOSS:
                    for b in BLAMES:
                        u = e = 0.0
                        for r in range(REPS):
                            rng = random.Random(hash((mech, orat, lam, m, b, r)) & 0xFFFFFFFF)
                            s, ea, _ = run_once(mech, orat, lam, m, b, 1.0, rng)
                            u += s
                            e += ea
                        rows.append(dict(mechanism=mech, out_ratio=orat, lam=lam, loss=m, blame=b,
                                         utility=round(u / REPS, 4), err_adopt=round(e / REPS, 1)))
    with (Path(__file__).parent / "sim_results_v3_blame.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Part A saved ({len(rows)} cells)")
    for b in BLAMES:
        for m in LOSS:
            sub = [r for r in rows if r["blame"] == b and r["loss"] == m]
            ranked = sorted(MECHS, key=lambda x: -sum(r["utility"] for r in sub if r["mechanism"] == x) / 9)
            tag = f"问责b={b} 损失x{m}: "
            print(tag + " ".join(f"{x}={sum(r['utility'] for r in sub if r['mechanism']==x)/9:.3f}" for x in ranked))


def part_b():
    rows = []
    for k in KS:
        for mech in MECHS:
            u = e = 0.0
            for r in range(REPS):
                rng = random.Random(hash(("comp", k, mech, r)) & 0xFFFFFFFF)
                s, ea, _ = run_once(mech, 0.35, 0.3, 2, 0, k, rng)
                u += s
                e += ea
            rows.append(dict(k=k, mechanism=mech, utility=round(u / REPS, 4), err_adopt=round(e / REPS, 1)))
    with (Path(__file__).parent / "sim_results_v3_competence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nPart B (模型能力扫描, out=0.35 lam=0.3 损失x2):")
    for k in KS:
        sub = sorted([r for r in rows if r["k"] == k], key=lambda x: -x["utility"])
        hard_u = next(r["utility"] for r in sub if r["mechanism"] == "hard")
        print(f"  k={k}: " + " | ".join(f"{r['mechanism']}={r['utility']:.3f}" for r in sub))


if __name__ == "__main__":
    part_a()
    part_b()
