"""
Hybrid Retriever — Dense + BM25
=================================
Combines semantic vector search (ChromaDB) with sparse BM25
keyword matching. Weights are controlled by the shared
RetrievalConfig and updated by Layer 3 between queries.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from src.layers.layer3_adaptation import RetrievalConfig

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Dense + BM25 hybrid retriever.

    Parameters
    ----------
    config         : RetrievalConfig — shared live config (α weights, top_k)
    chroma_dir     : str             — ChromaDB persist directory
    collection_name: str             — ChromaDB collection name
    embed_model    : str             — sentence-transformers model name
    """

    def __init__(
        self,
        config,
        chroma_dir: str = "data/chroma_db",
        collection_name: str = "asr_rag_corpus",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.config     = config
        self.embedder   = SentenceTransformer(embed_model)

        # ChromaDB setup
        self._chroma_client = chromadb.PersistentClient(path=chroma_dir)
        
        # Robust collection retrieval: avoid get_or_create_collection with metadata 
        # as it can fail if existing HNSW settings mismatch.
        try:
            self._collection = self._chroma_client.get_collection(name=collection_name)
            logger.info(f"[Retriever] Loaded existing collection: {collection_name}")
        except Exception:
            # Create new if it doesn't exist or failed to load
            self._collection = self._chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:ef_construction": 200, "hnsw:M": 16},
            )
            logger.info(f"[Retriever] Created new collection: {collection_name}")

        # BM25 — built lazily on first retrieve call
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict]  = []

    # ── Public API ─────────────────────────────────────────────────────

    def add_documents(self, docs: list[dict]):
        """
        Index a list of document chunks into ChromaDB and BM25.

        Each doc dict must have:
          - "id"     : unique string identifier
          - "text"   : passage text
          - "source" : paper title or file name
          - "section": paper section (abstract / methods / results / etc.)
        """
        texts     = [d["text"]   for d in docs]
        ids       = [d["id"]     for d in docs]
        metadatas = [{"source": d.get("source",""), "section": d.get("section","")}
                     for d in docs]

        # Use small batch size on laptops to avoid OOM during indexing
        batch_size = getattr(self.config, 'embedding_batch_size', 8)
        embeddings = self.embedder.encode(
            texts,
            batch_size        = batch_size,   # 8 for 8 GB RAM; 64 fine on GPU machines
            show_progress_bar = True
        ).tolist()

        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        # Rebuild BM25 index
        self._bm25_docs = docs
        tokenised = [d["text"].lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenised)
        logger.info(f"[Retriever] Indexed {len(docs)} documents.")

    def retrieve(self, query: str) -> list[dict]:
        """
        Run hybrid retrieval and return top-k merged results.

        Returns a list of dicts:
          {id, text, source, section, dense_score, bm25_score, hybrid_score}
        """
        k      = self.config.top_k
        alpha_d = self.config.alpha_dense
        alpha_s = self.config.alpha_sparse

        dense_results = self._dense_retrieve(query, k * 2)
        bm25_results  = self._bm25_retrieve(query, k * 2)

        merged = self._fuse(dense_results, bm25_results, alpha_d, alpha_s)
        top_k  = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)[:k]

        logger.debug(f"[Retriever] Retrieved {len(top_k)} docs "
                     f"(α_d={alpha_d:.2f}, α_s={alpha_s:.2f})")
        return top_k

    # ── Private helpers ────────────────────────────────────────────────

    def _dense_retrieve(self, query: str, n: int) -> list[dict]:
        q_emb = self.embedder.encode([query]).tolist()
        res   = self._collection.query(
            query_embeddings=q_emb, n_results=min(n, self._collection.count())
        )
        results = []
        if res["ids"] and res["ids"][0]:
            for doc_id, text, meta, dist in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0]
            ):
                # ChromaDB returns L2 distance; convert to cosine-like score
                score = max(0.0, 1.0 - dist)
                results.append({
                    "id": doc_id, "text": text,
                    "source": meta.get("source",""),
                    "section": meta.get("section",""),
                    "dense_score": score,
                })
        return results

    def _bm25_retrieve(self, query: str, n: int) -> list[dict]:
        if self._bm25 is None:
            return []
        tokenised_q = query.lower().split()
        scores      = self._bm25.get_scores(tokenised_q)
        max_score   = max(scores) if max(scores) > 0 else 1.0

        indexed = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n]

        return [
            {
                **self._bm25_docs[idx],
                "bm25_score": score / max_score,  # normalise to [0,1]
            }
            for idx, score in indexed
        ]

    @staticmethod
    def _fuse(
        dense: list[dict], bm25: list[dict], alpha_d: float, alpha_s: float
    ) -> dict[str, dict]:
        """Weighted score fusion — Reciprocal Rank Fusion variant."""
        merged: dict[str, dict] = {}

        for rank, doc in enumerate(dense):
            doc_id = doc["id"]
            merged[doc_id] = {**doc, "bm25_score": 0.0}
            # RRF score for dense
            merged[doc_id]["dense_rrf"] = alpha_d / (60 + rank + 1)

        for rank, doc in enumerate(bm25):
            doc_id = doc["id"]
            if doc_id not in merged:
                merged[doc_id] = {**doc, "dense_score": 0.0, "dense_rrf": 0.0}
            merged[doc_id]["bm25_score"]  = doc.get("bm25_score", 0.0)
            merged[doc_id]["bm25_rrf"]    = alpha_s / (60 + rank + 1)

        for doc_id, doc in merged.items():
            doc["hybrid_score"] = (
                doc.get("dense_rrf", 0.0) + doc.get("bm25_rrf", 0.0)
            )

        return merged
