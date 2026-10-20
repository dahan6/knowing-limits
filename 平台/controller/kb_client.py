"""本地知识库 HTTP 客户端"""
from typing import List, Dict, Optional
import requests

import config


class KBClient:
    def __init__(self, base_url: str = config.KB_API_BASE, default_kb: str = config.DEFAULT_KB_NAME):
        self.base_url = base_url.rstrip("/")
        self.default_kb = default_kb

    def search(
        self,
        query: str,
        kb_name: Optional[str] = None,
        top_k: int = config.KB_TOP_K,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_query_expansion: bool = False,
    ) -> List[Dict]:
        """调用知识库 /v1/search 端点，返回 chunks 列表。"""
        payload = {
            "kb_name": kb_name or self.default_kb,
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "use_query_expansion": use_query_expansion,
        }
        try:
            r = requests.post(f"{self.base_url}/v1/search", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get("results", [])
        except requests.exceptions.ConnectionError:
            print(f"[WARN] 无法连接知识库 API: {self.base_url}/v1/search，将跳过 KB 检索")
            return []
        except requests.exceptions.HTTPError as e:
            print(f"[WARN] 知识库检索返回错误: {e}")
            return []
        except Exception as e:
            print(f"[WARN] 知识库检索异常: {e}")
            return []

    def add_snippets(self, snippets: List[Dict], kb_name: Optional[str] = None) -> int:
        """向知识库写入文本片段，返回新增的 chunk 数量。"""
        payload = {
            "kb_name": kb_name or self.default_kb,
            "snippets": snippets,
        }
        try:
            r = requests.post(f"{self.base_url}/v1/add_snippets", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("added_chunks", 0)
        except Exception as e:
            print(f"[WARN] 知识库写入失败: {e}")
            return 0

    def ensure_kb(self, kb_name: str) -> bool:
        """Ensure a KB exists by adding a tiny placeholder snippet if empty."""
        try:
            payload = {
                "kb_name": kb_name,
                "snippets": [{"source": "_init_", "content": "KB initialized."}],
            }
            r = requests.post(f"{self.base_url}/v1/add_snippets", json=payload, timeout=30)
            return r.status_code == 200
        except Exception as e:
            print(f"[WARN] ensure_kb failed: {e}")
            return False
