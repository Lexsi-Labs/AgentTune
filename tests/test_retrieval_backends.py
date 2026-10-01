import hashlib

import pytest

from agenttune.rag.retrieval.chroma_backend import ChromaBackend
from agenttune.rag.retrieval.corpus_loader import CorpusDocument, build_index, chunk_corpus
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend


def _fake_embedding_call(self, input):
    """Deterministic, network-free stand-in for chromadb's DefaultEmbeddingFunction.

    ChromaBackend doesn't accept a custom embedding function (see chroma_backend.py),
    so it always falls back to chromadb's default, which lazily downloads an ONNX
    model on first use. That download is flaky/slow in sandboxed CI and caused
    test_chroma_backend_index_and_search_roundtrip to fail on httpcore.ReadTimeout.
    This patches just the embedding step with a fast sha256-derived vector so the
    test exercises real indexing/search/roundtrip logic with no network dependency.
    """
    import numpy as np

    if isinstance(input, str):
        input = [input]
    out = []
    for text in input:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        vec = np.frombuffer(digest, dtype="<f4").astype("float32")
        out.append(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0))
    return out


SAMPLE_DOCS = [
    CorpusDocument(
        doc_id="Ed Wood (film)",
        title="Ed Wood (film)",
        text=(
            "Ed Wood is a 1994 American biographical period comedy-drama film directed "
            "and produced by Tim Burton, starring Johnny Depp. The film concerns the "
            "period in Wood's life when he made his best-known films."
        ),
    ),
    CorpusDocument(
        doc_id="Scott Derrickson",
        title="Scott Derrickson",
        text=(
            "Scott Derrickson is an American director, screenwriter and producer. "
            "He is best known for directing horror films such as Sinister."
        ),
    ),
]


def test_chunk_corpus_produces_indexable_chunks():
    chunks = chunk_corpus(SAMPLE_DOCS, chunk_size=80, overlap=10)
    assert chunks
    for c in chunks:
        assert set(c.keys()) == {"chunk_id", "doc_id", "title", "text", "chunk_index"}
        assert c["text"].strip()


def test_sqlite_fts_index_and_search_roundtrip(tmp_path):
    backend = SQLiteFTSBackend(str(tmp_path / "corpus.db"))
    n = build_index(backend, SAMPLE_DOCS, chunk_size=200, overlap=20)
    assert n > 0

    results = backend.search("Tim Burton", top_k=3)
    assert results
    assert any("Burton" in r["content"] for r in results)
    assert results[0]["metadata"]["doc_id"] in {"Ed Wood (film)", "Scott Derrickson"}


def test_sqlite_fts_get_document_concatenates_chunks(tmp_path):
    backend = SQLiteFTSBackend(str(tmp_path / "corpus.db"))
    build_index(backend, SAMPLE_DOCS, chunk_size=50, overlap=10)

    doc = backend.get_document("Scott Derrickson")
    assert doc is not None
    assert "Sinister" in doc["content"]


def test_sqlite_fts_search_no_results(tmp_path):
    backend = SQLiteFTSBackend(str(tmp_path / "corpus.db"))
    build_index(backend, SAMPLE_DOCS, chunk_size=200, overlap=20)
    assert backend.search("xyznonexistentquery123", top_k=3) == []


def test_sqlite_fts_malformed_query_does_not_raise(tmp_path):
    backend = SQLiteFTSBackend(str(tmp_path / "corpus.db"))
    build_index(backend, SAMPLE_DOCS, chunk_size=200, overlap=20)
    # bare punctuation is not valid FTS5 syntax on its own — must not raise
    assert backend.search("???", top_k=3) == []


def test_sqlite_fts_get_document_missing_returns_none(tmp_path):
    backend = SQLiteFTSBackend(str(tmp_path / "corpus.db"))
    build_index(backend, SAMPLE_DOCS, chunk_size=200, overlap=20)
    assert backend.get_document("does-not-exist") is None


def test_sqlite_fts_backend_is_picklable(tmp_path):
    import pickle

    backend = SQLiteFTSBackend(str(tmp_path / "corpus.db"))
    build_index(backend, SAMPLE_DOCS, chunk_size=200, overlap=20)
    restored = pickle.loads(pickle.dumps(backend))
    assert restored.search("Burton", top_k=1)


@pytest.mark.skipif(
    pytest.importorskip("chromadb", reason="chromadb not installed") is None,
    reason="chromadb not installed",
)
def test_chroma_backend_index_and_search_roundtrip(tmp_path, monkeypatch):
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    monkeypatch.setattr(DefaultEmbeddingFunction, "__call__", _fake_embedding_call)

    backend = ChromaBackend(persist_dir=str(tmp_path / "chroma"))
    n = build_index(backend, SAMPLE_DOCS, chunk_size=200, overlap=20)
    assert n > 0
    results = backend.search("horror film director", top_k=3)
    assert results
    assert any("Derrickson" in r["content"] for r in results)
