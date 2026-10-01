"""
SearchBackend protocol — the interface every retrieval backend implements.

Deliberately smaller than Castform's `SearchClient` (no `embed`/`available_modes`
/`mode="auto"`): one backend instance is one retrieval mode (lexical or vector).
Multi-backend support comes from having interchangeable implementations
(SQLiteFTSBackend, ChromaBackend), not a runtime mode switch inside one class.

Implementations MUST be pickle-safe: constructors store only primitive
connection params (paths/strings), and any live client/connection is rebuilt
lazily in a private method — required because rollout workers may run in
separate processes.
"""

from typing import Any, Protocol, TypedDict


class SearchResult(TypedDict):
    content: str
    source: str
    metadata: dict[str, Any]
    score: float


class SearchBackend(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return up to `top_k` ranked chunks relevant to `query`."""
        ...

    def get_document(self, doc_id: str) -> SearchResult | None:
        """Return the full concatenated text of a document by id, or None."""
        ...

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Bulk-index chunks. Each chunk dict has keys:
        chunk_id, doc_id, title, text, chunk_index (see corpus_loader.chunk_corpus).
        """
        ...

    @property
    def name(self) -> str:
        """Short backend identifier, e.g. 'sqlite_fts' or 'chroma'."""
        ...
