"""
Phase 0: build a retrieval index (SQLite FTS5 or Chroma) from the HotpotQA
corpus. Run before verify_masking.py / train_grpo.py.

Usage:
    python -m agenttune.rag.scripts.build_index --backend sqlite --output_dir rag_experiments/indexes/sqlite
    python -m agenttune.rag.scripts.build_index --backend chroma --output_dir rag_experiments/indexes/chroma
"""

import argparse
import json
import os

from agenttune.rag.data.hotpotqa import build_corpus_from_hotpotqa, load_hotpotqa_splits
from agenttune.rag.retrieval.chroma_backend import ChromaBackend
from agenttune.rag.retrieval.corpus_loader import build_index
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
    parser.add_argument("--hotpotqa_config", default="distractor")
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    train_split, eval_split = load_hotpotqa_splits(
        config=args.hotpotqa_config, train_size=args.train_size, eval_size=args.eval_size
    )
    # Corpus is built from BOTH splits' contexts so eval questions are answerable too.
    from datasets import concatenate_datasets

    docs = build_corpus_from_hotpotqa(
        concatenate_datasets([train_split, eval_split]), max_docs=args.max_docs
    )
    print(f"[build_index] {len(docs)} unique documents from HotpotQA context.")

    backend = make_backend(args.backend, args.output_dir)
    n_chunks = build_index(backend, docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    print(
        f"[build_index] Indexed {n_chunks} chunks into backend='{args.backend}' at {args.output_dir}"
    )

    manifest = {
        "backend": args.backend,
        "output_dir": args.output_dir,
        "hotpotqa_config": args.hotpotqa_config,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "num_docs": len(docs),
        "num_chunks": n_chunks,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
