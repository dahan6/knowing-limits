"""本地完整模式使用示例"""
import sys
from pathlib import Path

# 把项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from cos import Controller, LLMProviderFactory, COSConfig


def main():
    # 方式 1：从环境变量创建
    config = COSConfig.from_env()
    config.mode = "local"
    provider = LLMProviderFactory.create(config)
    controller = Controller(llm_provider=provider, config=config)

    # 使用
    queries = [
        "中国的首都是哪里？",
        "水的化学式是什么？",
        "Linux 主机的 MAC 地址是多少？",
        "人生的意义是什么？",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"问题: {query}")
        print(f"{'='*60}")
        result = controller.process(query)
        print(f"路由: {result['case']} | 来源: {result['source']}")
        print(f"KB Score: {result['kb_score']:.3f} | LLM Known: {result['llm_detection']['proba']['known']:.3f}")
        print(f"回答: {result['answer'][:200]}...")


if __name__ == "__main__":
    main()
