#!/usr/bin/env python3
"""
根据 test_dataset.json 的 needs_kb 标记，为测试构建 KB 内容。

- needs_kb=true：联网搜索 + LLM 摘要，写入 KB
- needs_kb=false：不写入，用于测试 KB 缺失场景
"""
import json
import time
from pathlib import Path

from kb_client import KBClient
from web_search import WebSearchClient
from llm_client import LLMClient

DATASET = Path(__file__).parent / "test_dataset.json"

MANUAL_CONTENT = {
    "general": {
        "中国的首都是哪里？": [
            {
                "source": "地理常识",
                "content": "中国的首都是北京。北京位于华北平原北部，是中国的政治中心、文化中心、国际交往中心和科技创新中心。"
            }
        ],
        "第二次世界大战哪一年结束？": [
            {
                "source": "历史常识",
                "content": "第二次世界大战于1939年爆发，1945年结束。1945年5月8日德国投降，欧洲战场结束；1945年9月2日日本签署投降书，战争正式结束。"
            }
        ],
        "水的化学式是什么？": [
            {
                "source": "化学常识",
                "content": "水的化学式是 H₂O，由两个氢原子和一个氧原子组成，是地球上最常见的溶剂。"
            }
        ],
        "《红楼梦》的作者是谁？": [
            {
                "source": "文学常识",
                "content": "《红楼梦》的作者是清代作家曹雪芹，后四十回一般认为是高鹗续写。"
            }
        ],
        "100的平方根是多少？": [
            {
                "source": "数学常识",
                "content": "100 的平方根是 10，因为 10 × 10 = 100。"
            }
        ],
        "HTTP 协议默认使用哪个端口？": [
            {
                "source": "计算机常识",
                "content": "HTTP 协议默认使用 80 端口，HTTPS 默认使用 443 端口。"
            }
        ],
    },
    "local_doc": {
        "Linux 主机的 MAC 地址是多少？": [
            {
                "source": "双机直连文档.md",
                "content": "Linux 主机信息：IP 192.168.10.2/24，MAC 地址 70:70:FC:09:BA:98，SSH 用户 adazz18，挂载点 /mnt/win-share。"
            }
        ],
        "Windows 共享路径是什么？": [
            {
                "source": "双机直连文档.md",
                "content": "Windows 共享配置：主机名 adazz18，IP 192.168.10.1/24，共享路径 E:\\Share，网络路径 \\\\192.168.10.1\\Share。"
            }
        ],
    },
    "custom_local": {
        "深盆科技控制机 V0.1 的架构核心是什么？": [
            {
                "source": "深盆科技控制机V0.1白皮书",
                "content": "深盆科技认知操作系统控制机 V0.1 的架构核心是：输入后先进行查询扩写与上下文思考，然后并行执行 KB 检索（长期显式记忆）与 LLM 检索（长期隐式记忆），最后根据 KB score 与 LLM known probability 进行四象限分类路由。"
            }
        ],
    },
}


def summarize_for_kb(question: str, web_results: list, llm: LLMClient) -> str:
    if not web_results:
        return ""
    context = "\n\n".join(
        f"[{i+1}] {r.get('title', '')}\n{r.get('snippet', '')}"
        for i, r in enumerate(web_results)
    )
    messages = [
        {"role": "system", "content": "你负责把搜索摘要整理成一段简洁、准确、可用于知识库检索的说明文字（100-300字）。只输出整理后的文本，不要解释。"},
        {"role": "user", "content": f"问题：{question}\n\n搜索摘要：\n{context}\n\n请整理成知识库文本："},
    ]
    return llm._chat(messages, temperature=0.2, max_tokens=512)


TEST_KB_NAME = "controller_test_kb_v3"


def build():
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    kb = KBClient(default_kb=TEST_KB_NAME)
    web = WebSearchClient()
    llm = LLMClient()

    total_added = 0

    for item in dataset:
        if not item.get("needs_kb"):
            print(f"[SKIP] {item['id']}: 不写入 KB")
            continue

        q = item["question"]
        topic = item.get("kb_topic", "")

        snippets = []
        if topic in MANUAL_CONTENT and q in MANUAL_CONTENT[topic]:
            snippets = MANUAL_CONTENT[topic][q]
            print(f"[MANUAL] {item['id']}: {q}")
        else:
            print(f"[WEB] {item['id']}: {q}", end=" ")
            results = web.search(q, top_k=3)
            if results:
                summary = summarize_for_kb(q, results, llm)
                if summary:
                    snippets = [{"source": f"web_search_{item['id']}", "content": summary}]
                    print(f"-> 摘要 {len(summary)} 字")
                else:
                    print("-> 摘要为空")
            else:
                print("-> 无搜索结果")

            # 如果联网失败，使用通用手动兜底
            if not snippets and "general" in MANUAL_CONTENT and q in MANUAL_CONTENT["general"]:
                snippets = MANUAL_CONTENT["general"][q]
                print(f"         [FALLBACK] 使用手动兜底")

        if snippets:
            added = kb.add_snippets(snippets=snippets, kb_name=TEST_KB_NAME)
            total_added += added
            print(f"         写入 KB: {added} chunks")
        time.sleep(0.5)

    print(f"\n总计写入 KB: {total_added} chunks")


if __name__ == "__main__":
    build()
