"""Knowledge base management backed by ChromaDB + ONNX embeddings + BM25 + reranker."""
import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import jieba.posseg as pseg
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from document_loader import SUPPORTED_EXTENSIONS, IMAGE_EXTENSIONS, load_document, DocumentPart
from embeddings import embedding_fn
from retrieval import HybridSearcher


class KnowledgeBaseManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.searcher = HybridSearcher(self)

    @staticmethod
    def _slug(name: str) -> str:
        """Convert a human-readable name to a valid ASCII Chroma collection name."""
        name = name.strip()
        base = re.sub(r"[^a-zA-Z0-9]", "_", name)
        base = re.sub(r"_+", "_", base).strip("_")
        name_hash = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        slug = f"kb_{name_hash}_{base}"[:60].strip("._-")
        if len(slug) < 3:
            slug = f"kb_{name_hash}"
        return slug

    def _kb_upload_dir(self, kb_name: str) -> Path:
        path = config.UPLOAD_DIR / self._slug(kb_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_kbs(self) -> List[str]:
        """Return human-readable knowledge-base names ordered by creation time."""
        collections = self.client.list_collections()
        names = []
        for col in collections:
            try:
                meta = col.metadata or {}
                display = meta.get("kb_name")
            except Exception:
                display = None
            names.append(display if display else col.name)
        return sorted(names)

    def list_files(self, kb_name: str) -> List[str]:
        collection = self.client.get_collection(self._slug(kb_name))
        data = collection.get(include=["metadatas"])
        sources = set()
        for meta in data.get("metadatas") or []:
            source = meta.get("source") if meta else None
            if source:
                sources.add(source)
        return sorted(sources)

    def delete_kb(self, kb_name: str) -> str:
        slug = self._slug(kb_name)
        try:
            self.client.delete_collection(slug)
        except Exception:
            pass
        upload_dir = config.UPLOAD_DIR / slug
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        index_dir = config.INDEX_DIR / slug
        if index_dir.exists():
            shutil.rmtree(index_dir)
        return f"知识库 '{kb_name}' 已删除"

    def delete_file(self, kb_name: str, filename: str) -> str:
        slug = self._slug(kb_name)
        collection = self.client.get_collection(slug)
        collection.delete(where={"source": filename})
        self.searcher.rebuild_keyword_index(slug)
        upload_dir = config.UPLOAD_DIR / slug
        file_path = upload_dir / filename
        if file_path.exists():
            file_path.unlink()
        return f"文件 '{filename}' 已从知识库 '{kb_name}' 删除"

    # -----------------------------------------------------------------------
    # Metadata helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _extract_topics(text: str, top_n: int = 5) -> List[str]:
        """Extract candidate topic nouns from text using jieba POS tagging."""
        try:
            words = pseg.cut(text)
            nouns = {}
            for word, flag in words:
                w = word.strip()
                if len(w) < 2:
                    continue
                # n/NR/NS/NT/NN/NZ are noun-like tags
                if flag.startswith("n"):
                    nouns[w] = nouns.get(w, 0) + 1
            sorted_nouns = sorted(nouns.items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in sorted_nouns[:top_n]]
        except Exception:
            return []

    @staticmethod
    def _topics_to_str(topics) -> str:
        if isinstance(topics, list):
            return ",".join(str(t).strip() for t in topics if str(t).strip())
        return str(topics).strip() if topics else ""

    @staticmethod
    def _str_to_topics(topics_str: str) -> List[str]:
        if not topics_str:
            return []
        return [t.strip() for t in topics_str.split(",") if t.strip()]

    def _build_chunk_metadata(
        self,
        source: str,
        source_type: str = "text",
        chunk_index: int = 0,
        base_meta: Optional[Dict] = None,
        content: str = "",
    ) -> Dict:
        """Build metadata with memory management fields."""
        now = self._now()
        base = dict(base_meta) if base_meta else {}
        topics = base.get("topics") or self._extract_topics(content)
        return {
            "source": source,
            "source_type": base.get("source_type", source_type),
            "type": base.get("type", "text"),
            "chunk_index": chunk_index,
            "created_at": base.get("created_at", now),
            "updated_at": now,
            "verified": bool(base.get("verified", False)),
            "confidence": float(base.get("confidence", 0.5)),
            "weight": float(base.get("weight", config.MEMORY_DEFAULT_WEIGHT)),
            "access_count": int(base.get("access_count", 0)),
            "last_accessed": base.get("last_accessed", now),
            "topics": self._topics_to_str(topics),
            "version": int(base.get("version", 1)) + (1 if base_meta else 0),
            "user_id": base.get("user_id", ""),
            "tier": base.get("tier", "warm"),
        }

    @staticmethod
    def _normalize_metadata_for_output(meta: Dict) -> Dict:
        """Convert storage-friendly metadata back to richer types for API consumers."""
        if not meta:
            return meta
        out = dict(meta)
        if "topics" in out and isinstance(out["topics"], str):
            out["topics"] = KnowledgeBaseManager._str_to_topics(out["topics"])
        return out

    def _deduplicate_texts(
        self,
        slug: str,
        items: List[Dict],
        threshold: float = None,
    ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Deduplicate incoming texts against existing collection.
        Returns (new_items, update_items, skipped_ids).
        Each item is expected to have at least 'content' and may have 'id'/'metadata'.
        """
        threshold = threshold if threshold is not None else config.DEDUP_SIMILARITY_THRESHOLD
        collection = self.client.get_collection(slug, embedding_function=embedding_fn)

        new_items = []
        update_items = []
        skipped_ids = []

        for item in items:
            content = item.get("content", "").strip()
            if not content:
                continue

            # Try to find similar existing chunks
            try:
                results = collection.query(
                    query_texts=[content],
                    n_results=min(3, max(1, collection.count())),
                    include=["documents", "metadatas", "distances"],
                )
                candidates = list(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ))
            except Exception:
                candidates = []

            best_dup = None
            best_sim = 0.0
            for cid, cdoc, cmeta, dist in candidates:
                sim = 1.0 / (1.0 + float(dist))
                if sim > best_sim:
                    best_sim = sim
                    best_dup = (cid, cdoc, cmeta)

            incoming_confidence = float(item.get("metadata", {}).get("confidence", 0.5))
            incoming_weight = float(item.get("metadata", {}).get("weight", config.MEMORY_DEFAULT_WEIGHT))

            if best_dup and best_sim >= threshold:
                cid, cdoc, cmeta = best_dup
                existing_confidence = float(cmeta.get("confidence", 0.5))
                existing_weight = float(cmeta.get("weight", config.MEMORY_DEFAULT_WEIGHT))
                existing_verified = bool(cmeta.get("verified", False))
                existing_source_type = cmeta.get("source_type", "unknown")
                incoming_source_type = item.get("metadata", {}).get("source_type", "unknown")

                # Source trust ranking: higher is more trustworthy
                trust_rank = {
                    "user_added": 4,
                    "file": 3,
                    "llm_moe_verified": 2,
                    "web_search": 1,
                    "text": 0,
                    "unknown": 0,
                }
                existing_trust = trust_rank.get(existing_source_type, 0)
                incoming_trust = trust_rank.get(incoming_source_type, 0)

                # Conservative update rule:
                # 1. Protect high-trust sources from being overwritten by lower-trust sources
                # 2. Only update if incoming has higher trust, or same trust but significantly better quality
                should_update = False
                if incoming_trust > existing_trust:
                    should_update = True
                elif incoming_trust == existing_trust:
                    # Same trust: update only if clearly better (verified + higher confidence + longer)
                    if not existing_verified and item.get("metadata", {}).get("verified", False):
                        should_update = True
                    elif incoming_confidence > existing_confidence + 0.2 and len(content) > len(cdoc) * 1.5:
                        should_update = True
                # incoming_trust < existing_trust: never overwrite

                if should_update:
                    update_meta = self._build_chunk_metadata(
                        source=item.get("source", cmeta.get("source", "未知来源")),
                        source_type=incoming_source_type,
                        chunk_index=int(cmeta.get("chunk_index", 0)),
                        base_meta={**cmeta, **(item.get("metadata") or {}), "version": int(cmeta.get("version", 1))},
                        content=content,
                    )
                    update_meta["access_count"] = int(cmeta.get("access_count", 0))
                    update_meta["created_at"] = cmeta.get("created_at", update_meta["created_at"])
                    update_items.append({
                        "id": cid,
                        "content": content,
                        "metadata": update_meta,
                    })
                else:
                    skipped_ids.append(cid)
            else:
                new_items.append(item)

        return new_items, update_items, skipped_ids

    # -----------------------------------------------------------------------
    # Indexing
    # -----------------------------------------------------------------------
    def add_files(
        self,
        kb_name: str,
        file_paths: List[Path],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> str:
        slug = self._slug(kb_name)
        collection = self.client.get_or_create_collection(
            name=slug,
            metadata={"kb_name": kb_name},
            embedding_function=embedding_fn,
        )

        upload_dir = self._kb_upload_dir(kb_name)
        image_root = upload_dir / "images"
        image_root.mkdir(parents=True, exist_ok=True)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", ". ", "? ", "! ", " ", ""],
        )

        total_chunks = 0
        added_files: List[str] = []
        added_records: List[Dict] = []

        for src_path in file_paths:
            if not src_path.exists():
                continue
            if src_path.suffix.lower() not in (SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS):
                added_files.append(f"跳过 {src_path.name}（不支持的格式）")
                continue

            dst_path = upload_dir / src_path.name
            shutil.copy2(src_path, dst_path)

            try:
                doc_parts = load_document(dst_path, image_root=image_root)
            except Exception as e:
                added_files.append(f"解析失败 {src_path.name}: {e}")
                continue

            for part in doc_parts:
                if not part.content.strip():
                    continue

                chunks, metas = self._prepare_chunks(part, slug, splitter)
                if not chunks:
                    continue

                ids = [f"{part.source}_chunk_{i}" for i in range(len(chunks))]
                collection.add(
                    ids=ids,
                    documents=chunks,
                    metadatas=metas,
                )
                total_chunks += len(chunks)
                for cid, chunk, meta in zip(ids, chunks, metas):
                    added_records.append({"id": cid, "text": chunk, "metadata": meta})

            added_files.append(f"已索引 {src_path.name}")

        # Incrementally update keyword index
        if added_records:
            self.searcher.add_to_keyword_index(slug, added_records)

        return f"知识库 '{kb_name}' 处理完成；新增 {total_chunks} 个文本块。\n" + "\n".join(added_files)

    def add_texts(
        self,
        kb_name: str,
        texts: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        """直接添加文本片段到知识库，用于 Controller 回写验证后的结论。
        支持去重、增量索引和元数据管理。"""
        import uuid

        slug = self._slug(kb_name)
        collection = self.client.get_or_create_collection(
            name=slug,
            metadata={"kb_name": kb_name},
            embedding_function=embedding_fn,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", ". ", "? ", "! ", " ", ""],
        )

        # Step 1: deduplicate against existing collection
        new_items, update_items, _ = self._deduplicate_texts(slug, texts)

        # Step 2: apply updates
        updated_count = 0
        if update_items:
            update_ids = [u["id"] for u in update_items]
            update_docs = [u["content"] for u in update_items]
            update_metas = [u["metadata"] for u in update_items]
            collection.update(ids=update_ids, documents=update_docs, metadatas=update_metas)
            self.searcher.update_keyword_index(slug, update_items)
            updated_count = len(update_items)

        # Step 3: add new chunks
        added_records = []
        total_new_chunks = 0
        for item in new_items:
            source = item.get("source", "未知来源")
            content = item.get("content", "").strip()
            if not content:
                continue

            user_meta = item.get("metadata") or {}
            chunks = splitter.split_text(content)
            if not chunks:
                continue

            metadatas = [
                self._build_chunk_metadata(
                    source=source,
                    source_type=user_meta.get("source_type", "text"),
                    chunk_index=i,
                    base_meta=user_meta,
                    content=chunk,
                )
                for i, chunk in enumerate(chunks)
            ]
            ids = [f"{uuid.uuid4().hex[:8]}_chunk_{i}" for i in range(len(chunks))]
            collection.add(ids=ids, documents=chunks, metadatas=metadatas)
            total_new_chunks += len(chunks)
            for cid, chunk, meta in zip(ids, chunks, metadatas):
                added_records.append({"id": cid, "text": chunk, "metadata": meta})

        # Step 4: incremental BM25 index update
        if added_records:
            self.searcher.add_to_keyword_index(slug, added_records)

        return total_new_chunks + updated_count

    def _prepare_chunks(
        self,
        part: DocumentPart,
        slug: str,
        splitter: RecursiveCharacterTextSplitter,
    ) -> tuple[List[str], List[Dict]]:
        """Split a document part into chunks and attach metadata."""
        meta_base = self._build_chunk_metadata(
            source=part.source,
            source_type="file",
            content=part.content,
        )
        meta_base["kb_name"] = slug
        if part.type == "image":
            # Keep image OCR text as a single chunk
            return [part.content], [meta_base]
        elif part.type == "table":
            # Keep tables whole to preserve structure
            return [part.content], [meta_base]
        else:
            chunks = splitter.split_text(part.content)
            metadatas = [
                {**meta_base, "chunk_index": i} for i in range(len(chunks))
            ]
            return chunks, metadatas

    # -----------------------------------------------------------------------
    # Retrieval helpers
    # -----------------------------------------------------------------------
    def get_all_records(self, slug: str) -> List[Dict[str, Any]]:
        """Return all chunks in a collection as records for BM25/re-ranking."""
        collection = self.client.get_collection(slug, embedding_function=embedding_fn)
        data = collection.get(include=["documents", "metadatas"])
        records = []
        for doc_id, doc, meta in zip(
            data["ids"], data["documents"], data["metadatas"]
        ):
            records.append({"id": doc_id, "text": doc, "metadata": self._normalize_metadata_for_output(meta or {})})
        return records

    def get_records_by_ids(self, slug: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Return specific chunks by ID."""
        if not ids:
            return []
        collection = self.client.get_collection(slug, embedding_function=embedding_fn)
        data = collection.get(ids=ids, include=["documents", "metadatas"])
        records = []
        for doc_id, doc, meta in zip(
            data["ids"], data["documents"], data["metadatas"]
        ):
            records.append({"id": doc_id, "text": doc, "metadata": self._normalize_metadata_for_output(meta or {})})
        return records

    def vector_search(self, slug: str, query: str, top_n: int = 30) -> List[Dict]:
        """Run Chroma vector search and return normalized score list."""
        collection = self.client.get_collection(slug, embedding_function=embedding_fn)
        results = collection.query(
            query_texts=[query],
            n_results=top_n,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert L2 distance to a similarity-like score
            score = 1.0 / (1.0 + float(dist))
            items.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": self._normalize_metadata_for_output(meta or {}),
                    "score": score,
                }
            )
        return items

    def search(
        self,
        kb_name: str,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_query_expansion: bool = False,
        update_access: bool = True,
    ) -> List[Dict]:
        slug = self._slug(kb_name)

        if use_query_expansion:
            from query_understanding import rewrite_queries

            queries = rewrite_queries(query, n=2)
        else:
            queries = [query]

        all_results: List[Dict] = []
        seen_ids = set()
        for q in queries:
            results = self.searcher.search(
                slug=slug,
                query=q,
                top_k=top_k * 3 if use_query_expansion else top_k,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                rerank_candidates=config.RERANK_CANDIDATES,
            )
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    all_results.append(r)

        # Final truncation and deduplication
        from context_processor import deduplicate, compress

        final = deduplicate(all_results)[:top_k]
        final = compress(final)

        if update_access and final:
            self._update_access_stats(slug, [r["id"] for r in final])

        return final

    # -----------------------------------------------------------------------
    # Access tracking & tiering
    # -----------------------------------------------------------------------
    def _update_access_stats(self, slug: str, ids: List[str]) -> None:
        """Bump access_count and last_accessed for retrieved chunks."""
        if not ids:
            return
        collection = self.client.get_collection(slug, embedding_function=embedding_fn)
        try:
            data = collection.get(ids=ids, include=["metadatas"])
            updated_metas = []
            for meta in data.get("metadatas") or []:
                if not meta:
                    continue
                meta["access_count"] = int(meta.get("access_count", 0)) + 1
                meta["last_accessed"] = self._now()
                # Promote to hot if accessed frequently
                if meta["access_count"] >= 5:
                    meta["tier"] = "hot"
                updated_metas.append(meta)
            if updated_metas:
                collection.update(ids=ids, metadatas=updated_metas)
        except Exception:
            pass

    def archive_cold_chunks(self, kb_name: str, days: int = None) -> int:
        """Mark chunks not accessed for `days` as cold. Returns number archived."""
        days = days if days is not None else config.MEMORY_ARCHIVE_DAYS
        slug = self._slug(kb_name)
        collection = self.client.get_collection(slug, embedding_function=embedding_fn)
        now = datetime.now(timezone.utc)
        archived = 0
        try:
            data = collection.get(include=["metadatas"])
            ids_to_archive = []
            metas = []
            for doc_id, meta in zip(data["ids"], data.get("metadatas") or []):
                if not meta:
                    continue
                last_str = meta.get("last_accessed")
                if not last_str:
                    continue
                try:
                    last = datetime.fromisoformat(last_str)
                    if (now - last).days >= days and meta.get("tier") != "cold":
                        meta["tier"] = "cold"
                        ids_to_archive.append(doc_id)
                        metas.append(meta)
                except Exception:
                    continue
            if ids_to_archive:
                collection.update(ids=ids_to_archive, metadatas=metas)
                archived = len(ids_to_archive)
        except Exception:
            pass
        return archived
