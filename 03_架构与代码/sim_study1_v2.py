#!/usr/bin/env python3
"""Study 1 计算实验 v2: 增加损失结构因子(非对称损失 x1/x2/x3)
效用 = 正确决策+1 / 错误决策-m (每期期望)
其余规格同 v1 (设计书V2.0, 用户已确认+两项批准修改)
"""
import csv, random
from pathlib import Path

P_IN_CORRECT = 0.90
P_HARD_C = 0.778
P_HARD_D = 0.545
P_DISCLOSE_C = 0.667
P_OWN = 0.50
FRAC_C = 9 / 14
HONESTY_CREDIT = 0.10
GUIDE_MULT = 1.5

TAU0 = 0.5
BETA = 6.0
LAMBDAS = [0.1, 0.3, 0.5]
OUT_RATIOS = [0.2, 0.35, 0.5]
MECHANISMS = ["hard", "refuse", "guide", "governed"]
LOSS_MULTS = [1, 2, 3]
PERIODS = 100
REPS = 1000


def p_adopt(tau):
    return 1.0 / (1.0 + pow(2.718281828, -BETA * (tau - 0.5)))


def run_once(mech, out_ratio, lam, mult, rng):
    tau = TAU0
    score = ss_score = 0.0
    correct = advice_n = advice_ok = err_adopt = 0
    for t in range(PERIODS):
        is_out = rng.random() < out_ratio
        is_c = rng.random() < FRAC_C
        answer_trial = (not is_out) or mech == "hard" or (mech == "governed" and is_c)
        if answer_trial:
            if not is_out:
                ok = rng.random() < P_IN_CORRECT
            elif mech == "hard":
                ok = rng.random() < (P_HARD_C if is_c else P_HARD_D)
            else:
                ok = rng.random() < P_DISCLOSE_C
            if rng.random() < p_adopt(tau):
                decision_ok = ok
                advice_n += 1
                advice_ok += ok
                if not ok:
                    err_adopt += 1
            else:
                decision_ok = rng.random() < P_OWN
            tau = min(1.0, tau + lam * (1 - tau)) if ok else max(0.0, tau - lam * tau)
        else:
            decision_ok = rng.random() < P_OWN
            if mech == "guide":
                tau = min(1.0, tau + GUIDE_MULT * lam * HONESTY_CREDIT * (1 - tau))
        correct += decision_ok
        score += 1.0 if decision_ok else -mult
        if t >= PERIODS - 20:
            ss_score += 1.0 if decision_ok else -mult
    actual_rel = advice_ok / advice_n if advice_n else 1.0
    return (correct / PERIODS, score / PERIODS, ss_score / 20.0,
            abs(tau - actual_rel), err_adopt, tau)


def main():
    here = Path(__file__).parent
    rows = []
    for mech in MECHANISMS:
        for orat in OUT_RATIOS:
            for lam in LAMBDAS:
                for mult in LOSS_MULTS:
                    acc = ut = ss = cal = err = tau_e = 0.0
                    for r in range(REPS):
                        rng = random.Random(hash((mech, orat, lam, mult, r)) & 0xFFFFFFFF)
                        a, b, c, d, e, f = run_once(mech, orat, lam, mult, rng)
                        acc += a
                        ut += b
                        ss += c
                        cal += d
                        err += e
                        tau_e += f
                    n = REPS
                    rows.append(dict(mechanism=mech, out_ratio=orat, lam=lam, loss_mult=mult,
                                     acc=round(acc / n, 4), utility=round(ut / n, 4),
                                     steady_util=round(ss / n, 4), calib_err=round(cal / n, 4),
                                     err_adopt=round(err / n, 1), tau_end=round(tau_e / n, 3)))
    with (here / "sim_results_v2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved sim_results_v2.csv ({len(rows)} cells)")

    print("\n==== 分损失结构的机制排序(效用, 全格平均) ====")
    for mult in LOSS_MULTS:
        sub_all = [r for r in rows if r["loss_mult"] == mult]
        line = f"损失x{mult}: "
        ranked = sorted(MECHANISMS, key=lambda m: -sum(r["utility"] for r in sub_all if r["mechanism"] == m) /
                        len([r for r in sub_all if r["mechanism"] == m]))
        for m in ranked:
            u = sum(r["utility"] for r in sub_all if r["mechanism"] == m) / \
                len([r for r in sub_all if r["mechanism"] == m])
            line += f"{m}={u:.3f} "
        print(line)
    print("\n==== 高风险情境切片 (out=0.50, lam=0.3) ====")
    for mult in LOSS_MULTS:
        sub = [r for r in rows if r["out_ratio"] == 0.5 and r["lam"] == 0.3 and r["loss_mult"] == mult]
        for r in sorted(sub, key=lambda x: -x["utility"]):
            print(f"  x{mult} {r['mechanism']:9s} util={r['utility']:.3f} acc={r['acc']:.3f} err={r['err_adopt']}")


if __name__ == "__main__":
    main()
