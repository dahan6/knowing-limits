"""联网搜索客户端：多 provider 轮换 + 本地缓存，减少超时和失败。"""
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import warnings

import config

CACHE_FILE = Path(__file__).parent / "web_search_cache.json"
CACHE_TTL = 3600 * 24  # 缓存 24 小时


class WebSearchClient:
    def __init__(self, enabled: bool = config.ENABLE_WEB_SEARCH, region: str = config.WEB_SEARCH_REGION):
        self.enabled = enabled
        self.region = region
        self._providers = []
        self._cache = self._load_cache()

        if not enabled:
            return

        # Provider 1: DuckDuckGo (ddgs)
        try:
            from ddgs import DDGS
            self._providers.append(("ddgs", DDGS()))
        except Exception as e:
            warnings.warn(f"[WebSearch] 未安装 ddgs，跳过: {e}")

        # Provider 2: SearX public instances (optional fallback)
        # 可通过环境变量配置
        searx_url = getattr(config, "SEARX_URL", "")
        if searx_url:
            self._providers.append(("searx", searx_url))

    def _load_cache(self) -> Dict:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 清理过期缓存
                now = time.time()
                return {k: v for k, v in data.items() if now - v.get("ts", 0) < CACHE_TTL}
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _cache_key(self, query: str, top_k: int) -> str:
        return f"{query.strip().lower()}|{top_k}"

    def _search_ddgs(self, query: str, top_k: int) -> List[Dict]:
        provider = next((p for name, p in self._providers if name == "ddgs"), None)
        if provider is None:
            return []
        try:
            results = provider.text(query, region=self.region, max_results=top_k)
            out = []
            for r in results:
                out.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })
            return out
        except Exception as e:
            warnings.warn(f"[WebSearch] ddgs 搜索失败: {e}")
            return []

    def _search_searx(self, query: str, top_k: int) -> List[Dict]:
        url = next((p for name, p in self._providers if name == "searx"), None)
        if not url:
            return []
        try:
            import requests
            resp = requests.get(
                url,
                params={"q": query, "format": "json", "language": "zh"},
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            out = []
            for r in data.get("results", [])[:top_k]:
                out.append({
                    "title": r.get("title", ""),
                    "link": r.get("url", ""),
                    "snippet": r.get("content", ""),
                })
            return out
        except Exception as e:
            warnings.warn(f"[WebSearch] searx 搜索失败: {e}")
            return []

    def search(self, query: str, top_k: int = config.WEB_SEARCH_TOP_K) -> List[Dict]:
        """返回搜索结果列表，优先读缓存。"""
        if not self.enabled:
            return []

        key = self._cache_key(query, top_k)
        cached = self._cache.get(key)
        if cached:
            return cached["results"]

        # 按 provider 顺序尝试
        providers = [self._search_ddgs, self._search_searx]
        results = []
        for provider_fn in providers:
            results = provider_fn(query, top_k)
            if results:
                break

        if results:
            self._cache[key] = {"results": results, "ts": time.time()}
            self._save_cache()

        return results

    def preload_cache(self, entries: List[Dict]) -> int:
        """预加载缓存条目，用于测试可复现。"""
        count = 0
        for entry in entries:
            query = entry.get("query", "")
            top_k = entry.get("top_k", config.WEB_SEARCH_TOP_K)
            results = entry.get("results", [])
            if query and results:
                self._cache[self._cache_key(query, top_k)] = {
                    "results": results,
                    "ts": time.time(),
                }
                count += 1
        self._save_cache()
        return count
