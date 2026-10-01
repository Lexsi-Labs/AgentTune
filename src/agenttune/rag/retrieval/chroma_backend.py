"""
Chroma vector retrieval backend.

Pure vector search using chromadb's own collection API end-to-end (its
default embedding function, its own similarity search) — no custom
embedding or fusion code. New dependency: chromadb (see pyproject.toml's
`rag` extras group).
"""

from typing import Any

from .base import SearchResult


class ChromaBackend:
    name = "chroma"

    def __init__(self, persist_dir: str, collection_name: str = "rag_corpus"):
        """Stores only primitive params — pickle-safe, client rebuilt lazily."""
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def _collection(self):
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_dir)
        return client.get_or_create_collection(self.collection_name)

    def index(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        collection = self._collection()
        # Chroma add() has a batch-size ceiling; chunk the inserts defensively.
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {"doc_id": c["doc_id"], "title": c["title"], "chunk_index": c["chunk_index"]}
                    for c in batch
                ],
            )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        collection = self._collection()
        if collection.count() == 0:
            return []
        result = collection.query(query_texts=[query], n_results=top_k)
        out: list[SearchResult] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        for doc, meta, dist, chunk_id in zip(docs, metas, dists, ids, strict=False):
            out.append(
                SearchResult(
                    content=doc,
                    source=meta.get("title", ""),
                    metadata={"chunk_id": chunk_id, "doc_id": meta.get("doc_id", "")},
                    score=1.0 - float(dist),  # cosine distance -> similarity-ish score
                )
            )
        return out

    def get_document(self, doc_id: str) -> SearchResult | None:
        collection = self._collection()
        result = collection.get(where={"doc_id": doc_id})
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        if not docs:
            return None
        pairs = sorted(zip(metas, docs, strict=False), key=lambda p: p[0].get("chunk_index", 0))
        title = pairs[0][0].get("title", "")
        full_text = "\n".join(text for _, text in pairs)
        return SearchResult(content=full_text, source=title, metadata={"doc_id": doc_id}, score=1.0)
