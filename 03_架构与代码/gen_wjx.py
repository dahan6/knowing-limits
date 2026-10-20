#!/usr/bin/env python3
"""问卷星导入文件 v4: 四条件(硬答/拒答/拒答+指引/治理版), 统一28题骨架=14边界内+9C型+5D型"""
import json, random, re
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
present = [json.loads(l) for l in (CORPUS / "questions_present.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
bank = json.loads((HERE / "responses_bank.json").read_text(encoding="utf-8"))
cans = {json.loads(l)["qid"]: json.loads(l) for l in (HERE / "c_answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
caps = {}
for f in ("stocks_sampled.json", "stocks_batch2.json", "stocks_batch3.json"):
    for s in json.loads((CORPUS / f).read_text(encoding="utf-8")):
        caps[s["code"]] = s.get("cap_yi")
hints = json.loads((CORPUS / "hint_business.json").read_text(encoding="utf-8")) if (CORPUS / "hint_business.json").exists() else {}
qs_all = {q["qid"]: q for q in (json.loads(l) for l in (CORPUS / "questions_clean.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

rng = random.Random(2026)
ins = rng.sample([q for q in present if q["side"] == "in"], 14)
douts = rng.sample([q for q in present if q["side"] == "out"], 5)
cids = list(cans.keys())
trials = [("in", q) for q in ins] + [("C", qs_all[qid]) for qid in cids] + [("D", q) for q in douts]
rng.shuffle(trials)

TRUST = [
    "这个AI在其知识范围内给出的回答是准确的",
    "这个AI具备完成这类分析任务的专业能力",
    "这个AI会诚实地承认自己不知道的事情",
    "这个AI不会在缺乏依据时编造答案",
    "这个AI的回答是透明、可追溯的",
    "这个AI的设计考虑了使用者的利益",
]

INTRO = ("【情境】你是一家投资机构的研究助理,日常需要对A股上市公司的经营状况做出快速判断。"
         "为提高效率,公司为你配备了一套AI经营分析系统——它掌握部分上市公司的年报资料,"
         "会基于其掌握的信息给出分析建议,但它并非无所不知。"
         "【任务】共28题,每题涉及一家上市公司的一项财务指标(营业收入/归母净利润/总资产)与某个数值的比较。"
         "每题流程:①阅读公司参考信息,作出你的初步判断;②下一页查看AI系统的建议;③作出最终判断。"
         "【说明】参考信息(行业/市值等)来自公开渠道,可作为估计依据;你的判断质量将计入工作考核;"
         "AI建议仅供参考,最终判断由你负责;没有标准对错,我们关注真实决策过程。"
         "【知情同意】任务约25分钟,数据仅用于学术研究且匿名处理,可随时退出。我已知悉并同意参与")

METRIC_FULL = {"营业收入": "营业收入", "归母净利润": "归属于母公司股东的净利润", "总资产": "总资产"}


def refuse_rich(q, guide=False):
    mf = METRIC_FULL.get(q["metric"], q["metric"])
    t = (f"我理解您需要判断{q['company']}{q['year']}年的{q['metric']}情况。"
         f"但经过检索,我当前掌握的资料中没有收录{q['company']}的{q['year']}年年度报告,"
         f"无法获得其{mf}的可靠数据。为避免凭推测提供不准确的信息误导您的判断,我不能就这一问题给出结论。")
    if guide:
        t += (f"如果您能提供{q['company']}{q['year']}年年度报告(其中\"主要会计数据\"部分载有{mf}),"
              "我可以立即为您进行分析判断。")
    return t


def advice_for(kind, q):
    if kind == "in":
        return bank[q["qid"]]["hard"]
    c = cans.get(q["qid"])
    if kind == "C":
        return (c["infer"] if c else refuse_rich(q)) if COND == "governed" else (c["hard"] if c and COND == "hard" else refuse_rich(q, COND == "refuse_guide"))
    return {"hard": bank[q["qid"]]["hard"], "refuse": refuse_rich(q),
            "refuse_guide": refuse_rich(q, True), "governed": refuse_rich(q)}[COND]


def clean(t, n=650):
    return re.sub(r"\s*\n\s*", " ", str(t)).strip()[:n]


def gen(cond_key, fname):
    global COND
    COND = cond_key
    lines = []
    qn = 0

    def q(stem, opts):
        nonlocal qn
        qn += 1
        lines.append(f"{qn}、{clean(stem)}【单选题】")
        for o in opts:
            lines.append(clean(o))
        lines.append("")

    q(INTRO, ["同意,开始任务"])
    q("您的年龄段", ["18-25岁", "26-35岁", "36-45岁", "46岁及以上"])
    q("您的性别", ["男", "女"])
    q("您的学历", ["高中及以下", "大专", "本科", "硕士及以上"])
    q("您使用AI工具的频率(1=从不,7=非常频繁)", [str(k) for k in range(1, 8)])
    q("您对财务报表的熟悉程度(1=完全不熟悉,7=非常熟悉)", [str(k) for k in range(1, 8)])
    for i, (kind, t) in enumerate(trials, 1):
        cap = caps.get(t["code"])
        biz = hints.get(t["code"])
        parts = [f"{t['company']}是一家A股上市公司"]
        if biz:
            parts.append(f"{biz}")
        if cap:
            parts.append(f"当前总市值约{cap}亿元")
        hint = "参考信息:" + ",".join(parts) + "。" if len(parts) > 1 else ""
        adv = advice_for(kind, t)
        q(f"[第{i}/28题] {hint}{t['question']} 你的初步判断是(下一页将显示AI建议)", ["超过", "不超过"])
        q(f"[第{i}/28题] 重看题目:{t['question']} ——AI经营分析系统的建议:{adv} ——请作出你的最终判断", ["超过", "不超过"])
        if i % 7 == 0 and i < 28:
            q("基于目前的体验,你认为这个AI系统能可靠回答\"某公司营业收入/净利润/总资产是否超过X亿元\"这类问题吗?", ["能", "部分能", "不能"])
        if i == 14:
            for j, item in enumerate(TRUST, 1):
                q(f"[阶段问卷T1·{j}/6] {item}(1=非常不同意,7=非常同意)", [str(k) for k in range(1, 8)])
    for j, item in enumerate(TRUST, 1):
        q(f"[阶段问卷T2·{j}/6] {item}(1=非常不同意,7=非常同意)", [str(k) for k in range(1, 8)])
    q("任务完成!感谢参与。本研究关注AI系统的知识边界管理如何影响人机协同决策。请截图本页,凭完成码ZDBZD-2026领取报酬", ["已完成"])
    (HERE / fname).write_text("\n".join(lines), encoding="utf-8")
    print(f"{fname}: {qn}题")


for ck, name in (("hard", "wjx_硬答组.txt"), ("refuse", "wjx_拒答组.txt"),
                 ("refuse_guide", "wjx_拒答指引组.txt"), ("governed", "wjx_治理版.txt")):
    gen(ck, name)
print("骨架:", "".join({"in": "内", "C": "C", "D": "D"}[k] for k, _ in trials))
