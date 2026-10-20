#!/usr/bin/env python3
"""Controller V0.1 CLI 入口"""
import json
import sys

from controller import Controller


def main():
    ctrl = Controller()

    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = ctrl.process(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("Controller V0.1 交互模式。输入问题，空行退出。")
    while True:
        try:
            query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        result = ctrl.process(query)
        print(f"\n[{result['case']}/{result['source']}] {result['answer']}")


if __name__ == "__main__":
    main()
