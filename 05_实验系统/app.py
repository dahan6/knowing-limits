#!/usr/bin/env python3
"""《知道不知道》被试任务前端 v0.

流程: 登陆(被试号+条件分配) -> 指导语 -> 28试次(初始判断->AI建议->最终决策)
      每第7试次后插边界预测 | 第14试次后信任量表T1 | 28试次后T2+人口学 -> 完成
数据: questions 文件 + 应答库(预生成回放), 全程事件日志 JSONL.
"""
import json, os, random, time
from pathlib import Path

from flask import Flask, jsonify, request, session as websession

HERE = Path(__file__).parent
EXP = HERE.parent
STIMULUS = EXP / "stimulus" / "corpus"
LOG_DIR = HERE / "logs"
LOG_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("EXP_SECRET", "zhidaobuzhidao-dev")

N_TRIALS = int(os.environ.get("EXP_TRIALS", "28"))
CONDITIONS = ["hard", "refuse", "refuse_guide"]  # 硬答 / 拒答 / 拒答+指引


def load_questions(n=N_TRIALS):
    """取题: 边界内/外各半, 每被试固定种子抽样, 保证同条件被试可复现。"""
    path = STIMULUS / "questions_present.jsonl"
    if not path.exists():
        path = STIMULUS / "questions_present.jsonl"
    qs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if "verdict" in qs[0]:
        qs = [q for q in qs if q.get("verdict") in ("in_clean", "out_clean")]
    half = n // 2
    rng = random.Random(websession.get("sid", "common-pool"))
    ins = [q for q in qs if q["side"] == "in"]
    outs = [q for q in qs if q["side"] == "out"]
    pick = rng.sample(ins, min(half, len(ins))) + rng.sample(outs, min(n - half, len(outs)))
    rng.shuffle(pick)
    return pick[:n]


def load_bank():
    """应答库: {qid: {condition: answer_text}} 预生成回放。"""
    p = HERE / "responses_bank.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def advice_for(q, condition, bank):
    b = bank.get(q["qid"], {}).get(condition)
    if b:
        return b
    # 兜底(预生成完成前): 占位文案, 上线前必须清零
    if condition == "hard":
        return f"根据我的分析，{q['company']}公司{q['year']}年的{q['metric']}{'超过' if q['gold']=='超过' else '不超过'}{q['threshold_yi']}亿元。"
    if condition == "refuse":
        return "该问题超出我当前的知识范围，我无法给出可靠回答。"
    return f"该问题超出我当前的知识范围，我无法给出可靠回答。要回答这个问题，我需要掌握{q['company']}公司{q['year']}年年报方面的资料。"


TRUST_ITEMS = [
    ("能力", "这个AI在其知识范围内给出的回答是准确的"),
    ("能力", "这个AI具备完成这类分析任务的专业能力"),
    ("诚信", "这个AI会诚实地承认自己不知道的事情"),
    ("诚信", "这个AI不会在缺乏依据时编造答案"),
    ("诚信", "这个AI的回答是透明、可追溯的"),
    ("善意", "这个AI的设计考虑了使用者的利益"),
]
DEMO_ITEMS = ["年龄", "性别", "学历", "是否经常使用AI工具(1=从不,7=非常频繁)", "对财务报表的熟悉程度(1=完全不熟悉,7=非常熟悉)"]

HTML_HEAD = "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>body{font-family:system-ui;max-width:640px;margin:24px auto;padding:0 16px;line-height:1.7}button{font-size:17px;padding:12px 24px;margin:8px;border-radius:8px;border:1px solid #888;background:#fff;cursor:pointer}.advice{background:#f2f7ff;border-left:4px solid #4a7dc9;padding:14px;margin:14px 0;border-radius:6px;white-space:pre-wrap}.q{font-size:19px;font-weight:600}.muted{color:#777;font-size:14px}input[type=range]{width:100%}</style></head><body>"


def log(event, **kw):
    sid = websession.get("sid", "anon")
    rec = {"t": round(time.time(), 3), "sid": sid, "event": event}
    rec.update(kw)
    with open(LOG_DIR / f"{sid}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.route("/")
def index():
    return HTML_HEAD + f"""
    <h2>AI辅助经营分析决策研究</h2>
    <p class='muted'>本研究考察管理者如何借助AI系统进行经营信息判断。任务约25分钟，请在一个不受打扰的环境中完成。</p>
    <form method='post' action='/start'>
      <p>被试编号: <input name='sid' required style='font-size:18px;padding:8px'></p>
      <p class='muted'>(由问卷平台提供的编号)</p>
      <button type='submit'>开始</button>
    </form></body></html>"""


@app.route("/start", methods=["POST"])
def start():
    sid = request.form.get("sid", "").strip()
    if not sid:
        return "need sid", 400
    websession["sid"] = sid
    cond = websession.get("cond") or random.Random(sid + os.environ.get("EXP_SALT", "")).choice(CONDITIONS)
    websession["cond"] = cond
    websession["trial"] = 0
    log("start", cond=cond)
    return HTML_HEAD + """
    <h3>知情同意</h3>
    <p>本任务包含28个企业经营信息判断题。每题流程：<b>① 你先作初步判断 → ② 查看AI系统的建议 → ③ 作出最终判断</b>。请在每一阶段独立、认真作答，没有标准对错要求，我们关注的是真实决策过程。</p>
    <p>数据仅用于学术研究，匿名处理。中途可随时退出。</p>
    <form method='post' action='/go'>
    <label><input type='checkbox' required> 我已阅读并同意参与</label>
    <button type='submit'>进入任务</button></form></body></html>"""


@app.route("/go", methods=["POST"])
def go():
    websession.setdefault("qs", json.dumps(load_questions(), ensure_ascii=False))
    websession["qs"] = websession.get("qs") or json.dumps(load_questions(), ensure_ascii=False)
    return next_step()


def get_qs():
    return json.loads(websession.get("qs") or "[]")


def next_step():
    t = websession.get("trial", 0)
    if t >= N_TRIALS:
        return scale_page("T2", final=True)
    if t == 14:
        return scale_page("T1", final=False)
    if t > 0 and t % 7 == 0:
        return predict_page()
    return trial_page(t)


@app.route("/trial", methods=["POST"])
def trial():
    q = get_qs()[websession.get("trial", 0)]
    stage = request.form.get("stage")
    if stage == "initial":
        log("initial", trial=websession.get("trial"), qid=q["qid"], choice=request.form.get("choice"))
        adv = advice_for(q, websession["cond"], load_bank())
        log("advice_shown", trial=websession.get("trial"), qid=q["qid"], advice=adv)
        return HTML_HEAD + f"""
        <p class='muted'>第 {websession['trial'] + 1}/{N_TRIALS} 题 · 你的初步判断已记录</p>
        <div class='q'>{q['question']}</div>
        <p>你的初步判断：<b>{'超过' if request.form.get('choice')=='超过' else '不超过'}</b></p>
        <div class='advice'><b>AI系统建议：</b><br>{adv}</div>
        <p><b>请作出最终判断：</b></p>
        <form method='post' action='/trial'>
        <input type='hidden' name='stage' value='final'>
        <button name='choice' value='超过'>超过</button>
        <button name='choice' value='不超过'>不超过</button>
        </form></body></html>"""
    if stage == "final":
        log("final", trial=websession.get("trial"), qid=q["qid"], choice=request.form.get("choice"),
            gold=q["gold"], correct=(request.form.get("choice") == q["gold"]))
        websession["trial"] = websession.get("trial", 0) + 1
        return next_step()
    return "bad stage", 400


def trial_page(t):
    q = get_qs()[t]
    return HTML_HEAD + f"""
    <p class='muted'>第 {t + 1}/{N_TRIALS} 题</p>
    <div class='q'>{q['question']}</div>
    <p><b>你的初步判断（之后会看到AI的建议）：</b></p>
    <form method='post' action='/trial'>
    <input type='hidden' name='stage' value='initial'>
    <button name='choice' value='超过'>超过</button>
    <button name='choice' value='不超过'>不超过</button>
    </form></body></html>"""


def predict_page():
    t = websession.get("trial", 0)
    return HTML_HEAD + f"""
    <p class='muted'>第 {t}/{N_TRIALS} 题完成</p>
    <h3>小问题</h3>
    <p>基于目前的体验，你认为这个AI系统<b>能可靠回答</b>下面这类问题吗？</p>
    <p class='muted'>(示例：“某公司2024年的营业收入是否超过X亿元？”)</p>
    <form method='post' action='/predict'>
    <button name='p' value='能'>能</button>
    <button name='p' value='部分能'>部分能</button>
    <button name='p' value='不能'>不能</button>
    </form></body></html>"""


@app.route("/predict", methods=["POST"])
def predict():
    log("boundary_predict", trial=websession.get("trial"), pred=request.form.get("p"))
    return next_step()


def scale_page(tag, final):
    items = "".join(
        f"<p>{i+1}. {dim}——{text}</p>" +
        "".join(f"<label style='margin-right:10px'><input type='radio' name='i{i}' value='{v}' required>{v}</label>" for v in range(1, 8))
        for i, (dim, text) in enumerate(TRUST_ITEMS))
    extra = ""
    if final:
        extra = "".join(
            f"<p>{d}：<input name='d{i}' style='font-size:18px;padding:6px'></p>"
            for i, d in enumerate(DEMO_ITEMS))
    return HTML_HEAD + f"""
    <h3>阶段问卷{tag}</h3>
    <p class='muted'>1=非常不同意，7=非常同意</p>
    <form method='post' action='/scale'>
    <input type='hidden' name='tag' value='{tag}'>
    {items}
    {extra}
    <button type='submit'>提交</button></form></body></html>"""


@app.route("/scale", methods=["POST"])
def scale():
    vals = {k: request.form.get(k) for k in request.form if k.startswith(("i", "d"))}
    log("scale", tag=request.form.get("tag"), **vals)
    if request.form.get("tag") == "T1":
        return next_step()
    return done_page()


def done_page():
    log("done")
    return HTML_HEAD + """
    <h3>任务完成，感谢参与！</h3>
    <p class='muted'>请截图本页，返回问卷平台提交以获取报酬。</p>
    <p><b>完成码：ZDBZD-2026</b></p>
    </body></html>"""


@app.route("/health")
def health():
    return jsonify(status="ok", service="zhidaobuzhidao-experiment", n_trials=N_TRIALS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("EXP_PORT", "8899")), threaded=True)
