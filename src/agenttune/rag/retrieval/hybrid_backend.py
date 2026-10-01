"""Hybrid lexical (BM25/FTS5) + dense (BGE-M3) retrieval backend.

BM25 alone caps gold-coverage at ~40% for the multi-hop CUAD questions
(question vocabulary diverges from the evidence; legal boilerplate ranks high).
BGE-M3 dense raises it to ~47%, and RRF-fusing both reaches ~51% (measured on
the aligned gold-passage index). This backend runs both and merges them with
reciprocal rank fusion.

Chunk embeddings are precomputed once (build_bge.py) into
<index_dir>/bge_m3_embeddings.npy + bge_m3_chunk_ids.json; only the query is
embedded at runtime (CPU).
"""

import json
import os
import sqlite3

import numpy as np

from .base import SearchResult
from .sqlite_fts import SQLiteFTSBackend

_RRF_K = 60.0


class HybridBackend:
    name = "hybrid"

    def __init__(
        self,
        index_dir: str,
        bge_model: str = "BAAI/bge-m3",
        bm25_top_k: int = 10,
        dense_top_k: int = 10,
        fuse_top_k: int = 8,
    ):
        # Pickle-safe ctor: store primitives, load clients lazily.
        self.index_dir = index_dir
        self.bge_model = bge_model
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.fuse_top_k = fuse_top_k
        self._bm: SQLiteFTSBackend | None = None
        self._embs_n = None
        self._cids: list[str] | None = None
        self._text: dict[str, str] | None = None
        self._model = None

    def _lazy_init(self) -> None:
        if self._bm is not None:
            return
        db = os.path.join(self.index_dir, "corpus.db")
        self._bm = SQLiteFTSBackend(db, match_all=False)
        embs = np.load(os.path.join(self.index_dir, "bge_m3_embeddings.npy"))
        self._embs_n = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        self._cids = json.load(open(os.path.join(self.index_dir, "bge_m3_chunk_ids.json")))
        con = sqlite3.connect(db)
        self._text = dict(con.execute("SELECT chunk_id, text FROM chunks"))
        con.close()
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.bge_model, device="cpu")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self._lazy_init()
        top_k = top_k or self.fuse_top_k
        bm_res = self._bm.search(query, top_k=self.bm25_top_k)  # type: ignore[union-attr]
        q = self._model.encode([query], convert_to_numpy=True)  # type: ignore[union-attr]
        q = q / np.linalg.norm(q)
        sims = self._embs_n @ q.ravel()
        dense_ids = [self._cids[i] for i in np.argsort(sims)[-self.dense_top_k :][::-1]]  # type: ignore[index]

        scores: dict[str, float] = {}
        for rank, r in enumerate(bm_res):
            cid = r["metadata"].get("chunk_id")
            if cid:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, cid in enumerate(dense_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        top_ids = sorted(scores, key=lambda c: -scores[c])[:top_k]
        return [
            SearchResult(
                content=self._text.get(cid, ""),
                source=cid.rsplit("::", 1)[0],
                metadata={"chunk_id": cid, "doc_id": cid.rsplit("::", 1)[0]},
                score=scores[cid],
            )
            for cid in top_ids
        ]

    def get_document(self, doc_id: str) -> SearchResult | None:
        self._lazy_init()
        return self._bm.get_document(doc_id)  # type: ignore[union-attr]

    def index(self, chunks: list[dict[str, object]]) -> None:
        raise NotImplementedError("HybridBackend reads a prebuilt index; build via build_bge.py")
