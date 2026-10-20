"""BM25 keyword index backed by rank-bm25 + jieba."""
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import jieba
from rank_bm25 import BM25Okapi

import config


def _tokenize(text: str) -> List[str]:
    """Tokenize Chinese with jieba and English/alphanumeric with regex."""
    text = text.lower()
    # Split English words and numbers
    tokens = [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1]
    # Add jieba Chinese tokens
    tokens.extend([t for t in jieba.lcut(text) if len(t.strip()) > 0])
    return tokens


class KeywordIndex:
    """Persistent BM25 index for a single knowledge base."""

    def __init__(self, slug: str):
        self.slug = slug
        self.index_dir = config.INDEX_DIR / slug
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.pkl_path = self.index_dir / "bm25.pkl"
        self.bm25: BM25Okapi | None = None
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []

    def _save(self, tokenized_corpus: List[List[str]]) -> None:
        with open(self.pkl_path, "wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                    "tokenized": tokenized_corpus,
                },
                f,
            )

    def build(self, records: List[Dict]) -> None:
        """Build the index from scratch. Each record must have id, text, metadata."""
        self.ids = [r["id"] for r in records]
        self.texts = [r["text"] for r in records]
        self.metadatas = [r.get("metadata", {}) for r in records]
        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._save(tokenized_corpus)

    def add_records(self, records: List[Dict]) -> None:
        """Incrementally append records and rebuild BM25 in memory."""
        if not records:
            return
        self.ids.extend([r["id"] for r in records])
        self.texts.extend([r["text"] for r in records])
        self.metadatas.extend([r.get("metadata", {}) for r in records])
        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._save(tokenized_corpus)

    def update_records(self, records: List[Dict]) -> None:
        """Update existing records by id and rebuild BM25 in memory."""
        if not records:
            return
        update_map = {r["id"]: r for r in records}
        for i, doc_id in enumerate(self.ids):
            if doc_id in update_map:
                r = update_map[doc_id]
                self.texts[i] = r["text"]
                self.metadatas[i] = r.get("metadata", {})
        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._save(tokenized_corpus)

    def delete_records(self, ids: List[str]) -> None:
        """Delete records by id and rebuild BM25 in memory."""
        if not ids:
            return
        id_set = set(ids)
        kept = [
            (i, t, m)
            for i, t, m in zip(self.ids, self.texts, self.metadatas)
            if i not in id_set
        ]
        self.ids = [k[0] for k in kept]
        self.texts = [k[1] for k in kept]
        self.metadatas = [k[2] for k in kept]
        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._save(tokenized_corpus)

    def load(self) -> bool:
        """Load a previously saved index. Returns True on success."""
        if not self.pkl_path.exists():
            return False
        try:
            with open(self.pkl_path, "rb") as f:
                data = pickle.load(f)
            self.ids = data["ids"]
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]
            tokenized = data.get("tokenized") or [_tokenize(t) for t in self.texts]
            self.bm25 = BM25Okapi(tokenized) if tokenized else None
            return True
        except Exception:
            return False

    def search(self, query: str, top_n: int = 10) -> List[Dict]:
        """Return top_n BM25 results as dicts with id, text, metadata, score."""
        if not self.bm25:
            return []
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        # Get top indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_n]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(
                {
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(scores[idx]),
                }
            )
        return results
