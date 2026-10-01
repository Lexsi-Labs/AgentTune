"""
Corpus loading + chunking — wraps agenttune's existing HFLoader rather than
reimplementing HF dataset loading.
"""

from dataclasses import dataclass, field
from typing import Any

from agenttune.data.loaders.hf_loader import HFLoader

from .base import SearchBackend
from .chunker import chunk_text


@dataclass
class CorpusDocument:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_corpus_from_hf(
    dataset_name: str,
    config_name: str | None,
    split: str,
    id_col: str,
    title_col: str,
    text_col: str,
    max_docs: int | None = None,
) -> list[CorpusDocument]:
    """Thin wrapper over HFLoader — reused, not reimplemented."""
    ds = HFLoader(dataset_name, config_name=config_name, split=split).load()
    docs: list[CorpusDocument] = []
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        docs.append(
            CorpusDocument(
                doc_id=str(row[id_col]),
                title=str(row[title_col]),
                text=str(row[text_col]),
            )
        )
    return docs


def chunk_corpus(
    docs: list[CorpusDocument], chunk_size: int = 512, overlap: int = 64
) -> list[dict[str, Any]]:
    """Flatten CorpusDocuments into indexable chunk dicts."""
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{doc.doc_id}::{idx}",
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "text": piece,
                    "chunk_index": idx,
                }
            )
    return chunks


def build_index(
    backend: SearchBackend,
    docs: list[CorpusDocument],
    chunk_size: int = 512,
    overlap: int = 64,
) -> int:
    """Chunk `docs` and index them into `backend`. Returns the chunk count."""
    chunks = chunk_corpus(docs, chunk_size=chunk_size, overlap=overlap)
    backend.index(chunks)
    return len(chunks)
