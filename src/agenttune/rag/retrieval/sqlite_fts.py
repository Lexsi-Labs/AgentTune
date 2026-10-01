"""
SQLite FTS5 lexical (BM25) retrieval backend.

Stdlib sqlite3 + FTS5, following the same schema+triggers pattern already
used in `agentic.tools.builtin.sql.SQLDatabaseTool`'s Enron FTS5 setup.
Ranking uses SQLite's own built-in `bm25()` function — reused, not
reimplemented. Zero new dependency.
"""

import sqlite3
from typing import Any

from .base import SearchResult


class SQLiteFTSBackend:
    name = "sqlite_fts"

    def __init__(self, db_path: str, match_all: bool = True):
        """Stores only the file path — pickle-safe, connection opened lazily.

        match_all=True (default, back-compat): query tokens are AND-ed (every
        token must hit the same chunk) — the HotpotQA-era behaviour, which
        works when the model emits entity-name keywords that literally appear
        in the target passage. match_all=False: tokens are OR-ed and bm25()
        ranks — standard BM25 semantics, required for FinDER's abstract
        analyst queries whose tokens don't all literally appear in the
        evidence (AND measured 90-96% empty results there; the FinDER paper's
        own BM25 baseline uses OR semantics).
        """
        self.db_path = db_path
        self.match_all = match_all

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE,
                doc_id TEXT,
                title TEXT,
                text TEXT,
                chunk_index INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                title, text, content='chunks', content_rowid='id'
            )
            """
        )
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, title, text) VALUES (new.id, new.title, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                UPDATE chunks_fts SET title = new.title, text = new.text WHERE rowid = old.id;
            END;
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        conn.commit()
        return conn

    def index(self, chunks: list[dict[str, Any]]) -> None:
        conn = self._conn()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, title, text, chunk_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (c["chunk_id"], c["doc_id"], c["title"], c["text"], c["chunk_index"])
                    for c in chunks
                ],
            )
            conn.commit()
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            conn.commit()
        finally:
            conn.close()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        conn = self._conn()
        try:
            joiner = " " if self.match_all else " OR "
            fts_query = joiner.join(f'"{t}"' for t in query.replace('"', "").split())
            if not fts_query:
                return []
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.title, c.text, bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS query (e.g. bare punctuation) — no results, not an error.
            return []
        finally:
            conn.close()
        return [
            SearchResult(
                content=text,
                source=title,
                metadata={"chunk_id": chunk_id, "doc_id": doc_id},
                score=-rank,  # bm25() returns lower-is-better; flip so higher=better
            )
            for chunk_id, doc_id, title, text, rank in rows
        ]

    def get_document(self, doc_id: str) -> SearchResult | None:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT title, text FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        title = rows[0][0]
        full_text = "\n".join(r[1] for r in rows)
        return SearchResult(content=full_text, source=title, metadata={"doc_id": doc_id}, score=1.0)
