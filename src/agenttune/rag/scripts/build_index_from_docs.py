"""
Build a retrieval index from real-world documents (PDF, DOCX, PPTX, etc.).

Extends build_index.py to support multi-format document ingestion. Loads files
from a directory, converts to CorpusDocuments, chunks them, and indexes into
the specified backend (SQLite FTS5 or Chroma).

Usage:
    # From a directory of mixed-format files:
    python -m agenttune.rag.scripts.build_index_from_docs \\
        --backend sqlite \\
        --input_dir /path/to/documents \\
        --output_dir rag_experiments/indexes/custom \\
        --recursive

    # From specific files:
    python -m agenttune.rag.scripts.build_index_from_docs \\
        --backend sqlite \\
        --files doc1.pdf doc2.docx doc3.pptx \\
        --output_dir rag_experiments/indexes/custom
"""

import argparse
import json
import os

from agenttune.rag.retrieval.chroma_backend import ChromaBackend
from agenttune.rag.retrieval.corpus_loader import build_index
from agenttune.rag.retrieval.document_loaders import (
    get_supported_formats,
    load_documents,
    load_file,
)
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend


def make_backend(name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if name == "sqlite":
        return SQLiteFTSBackend(os.path.join(output_dir, "corpus.db"))
    if name == "chroma":
        return ChromaBackend(persist_dir=output_dir)
    raise ValueError(f"Unknown backend '{name}'. Choose from: sqlite, chroma")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sqlite", "chroma"], required=True)
    parser.add_argument("--input_dir", default=None, help="Directory containing documents to index")
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Specific files to index (alternative to --input_dir)",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--recursive", action="store_true", default=True, help="Walk subdirectories (default True)"
    )
    parser.add_argument("--no_recursive", dest="recursive", action="store_false")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help=f"File extensions to include (default: all supported: {get_supported_formats()})",
    )
    args = parser.parse_args()

    if not args.input_dir and not args.files:
        parser.error("Either --input_dir or --files must be specified")

    # Load documents
    docs = []
    if args.input_dir:
        print(f"[build_index_from_docs] loading from {args.input_dir} (recursive={args.recursive})")
        docs = load_documents(args.input_dir, recursive=args.recursive, extensions=args.extensions)
    else:
        print(f"[build_index_from_docs] loading {len(args.files)} files")
        for fpath in args.files:
            try:
                file_docs = load_file(fpath)
                docs.extend(file_docs)
                print(f"  {fpath}: {len(file_docs)} documents")
            except Exception as e:
                print(f"  {fpath}: ERROR - {e}")

    print(f"[build_index_from_docs] {len(docs)} documents loaded")

    # Build index
    backend = make_backend(args.backend, args.output_dir)
    n_chunks = build_index(backend, docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    print(
        f"[build_index_from_docs] Indexed {n_chunks} chunks into backend='{args.backend}' at {args.output_dir}"
    )

    # Write manifest
    manifest = {
        "backend": args.backend,
        "output_dir": args.output_dir,
        "source": args.input_dir or "files",
        "num_docs": len(docs),
        "num_chunks": n_chunks,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "formats": list({d.metadata.get("format", "unknown") for d in docs}),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[build_index_from_docs] manifest written")


if __name__ == "__main__":
    main()
