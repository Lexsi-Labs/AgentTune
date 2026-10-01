"""
E1 prep: build the FinDER retrieval index (SQLite FTS5 / BM25) + all
data artifacts for the FinDER training runs.

Pipeline:
  1. Load FinDER rows (HF, single 'train' split).
  2. Recover each row's ticker by matching its gold references against the
     raw 10-K filings (the dataset repo's 10-k.zip, extracted).
  3. Ticker-level train/val split (val tickers fully disjoint from train —
     FINNLP_EXPERIMENTS.md v2 §1.1, "copy Castform").
  4. Corpus = deduplicated union of ALL gold references (train + val), so val
     questions remain answerable (same convention as the HotpotQA index).
  5. Chunk + index the corpus; record each question's gold chunk ids.

Artifacts written to --output_dir:
  corpus.db         SQLite FTS5 index (chunks carry chunk_id = {doc_id}::{idx})
  gold_chunks.json  {question_id: {gold_doc_ids, gold_chunk_ids}}
  splits.json       {train_ids, val_ids, row_tickers, val_tickers, stats}
  manifest.json     run parameters + counts

Usage:
    python -m agenttune.rag.scripts.build_index_finder \
        --backend sqlite --filings_dir /path/to/finder/10k \
        --output_dir rag_experiments/indexes/finder_sqlite
"""

import argparse
import json
import os
from collections import Counter

from agenttune.rag.data.finder import (
    build_corpus_from_finder,
    finder_ticker_split,
    load_filings,
    load_finder_rows,
    map_rows_to_tickers,
    ref_key,
)
from agenttune.rag.retrieval.chunker import chunk_text
from agenttune.rag.retrieval.corpus_loader import build_index
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["sqlite"], default="sqlite", help="E1 uses sqlite FTS5 (BM25) only."
    )
    parser.add_argument(
        "--filings_dir",
        required=True,
        help="Dir with the extracted 10-k.zip (one <TICKER>.html per filing).",
    )
    parser.add_argument("--val_frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    # FinDER references are long filing excerpts (mean ~2-3k chars), so the
    # chunks are bigger than HotpotQA's 512: 1024 keeps most references to
    # 1-4 chunks, making gold-chunk recall a clean signal.
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[build_index_finder] loading FinDER rows ...")
    rows = load_finder_rows()
    print(f"[build_index_finder] {len(rows)} rows")

    print(f"[build_index_finder] loading filings from {args.filings_dir} ...")
    filings = load_filings(args.filings_dir)
    print(
        f"[build_index_finder] {len(filings)} filings: "
        f"{sorted(filings)[:8]}{'...' if len(filings) > 8 else ''}"
    )

    print("[build_index_finder] mapping rows to tickers (window-hash alignment) ...")
    row_ticker = map_rows_to_tickers(rows, filings)
    n_resolved = sum(1 for t in row_ticker.values() if t is not None)
    print(
        f"[build_index_finder] ticker resolved for {n_resolved}/{len(rows)} rows "
        f"({100.0 * n_resolved / max(len(rows), 1):.1f}%); unresolved go to TRAIN only"
    )

    train_rows, val_rows = finder_ticker_split(
        rows, row_ticker, val_frac=args.val_frac, seed=args.seed
    )
    val_tickers = sorted({row_ticker[r["_id"]] for r in val_rows} - {None})
    print(
        f"[build_index_finder] split: {len(train_rows)} train / {len(val_rows)} val "
        f"over {len(val_tickers)} val tickers"
    )

    print("[build_index_finder] building corpus from gold references ...")
    # ref_key -> ticker, so every reference doc's title carries its company
    # (see build_corpus_from_finder's docstring — without it, AND-semantics
    # BM25 queries containing the ticker can never match their gold chunk).
    ref_to_ticker = {}
    for row in rows:
        t = row_ticker.get(row["_id"])
        if t is None:
            continue
        for ref in row["references"]:
            ref_to_ticker.setdefault(ref_key(ref), t)
    docs, ref_to_doc_id = build_corpus_from_finder(rows, ref_to_ticker=ref_to_ticker)
    print(f"[build_index_finder] {len(docs)} unique reference documents")

    print("[build_index_finder] indexing ...")
    backend = SQLiteFTSBackend(os.path.join(args.output_dir, "corpus.db"))
    n_chunks = build_index(backend, docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    print(f"[build_index_finder] indexed {n_chunks} chunks")

    # Gold chunk ids per question: chunk every gold reference's doc exactly as
    # build_index did (same chunker, same params) and collect its chunk ids.
    print("[build_index_finder] computing gold chunk ids per question ...")
    doc_chunk_counts = {
        doc.doc_id: len(
            chunk_text(doc.text, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
        )
        for doc in docs
    }
    gold_chunks = {}
    for row in rows:
        doc_ids = []
        for ref in row["references"]:
            doc_id = ref_to_doc_id[ref_key(ref)]
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
        chunk_ids = [
            f"{doc_id}::{i}" for doc_id in doc_ids for i in range(doc_chunk_counts[doc_id])
        ]
        gold_chunks[row["_id"]] = {
            "gold_doc_ids": doc_ids,
            "gold_chunk_ids": chunk_ids,
        }
    with open(os.path.join(args.output_dir, "gold_chunks.json"), "w") as f:
        json.dump(gold_chunks, f)

    with open(os.path.join(args.output_dir, "splits.json"), "w") as f:
        json.dump(
            {
                "train_ids": [r["_id"] for r in train_rows],
                "val_ids": [r["_id"] for r in val_rows],
                "val_tickers": val_tickers,
                "row_tickers": row_ticker,
                "seed": args.seed,
                "val_frac": args.val_frac,
            },
            f,
        )

    type_dist = Counter(r["type"] for r in rows)
    manifest = {
        "dataset": "Linq-AI-Research/FinDER",
        "backend": args.backend,
        "output_dir": args.output_dir,
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_val_tickers": len(val_tickers),
        "ticker_resolved_rows": n_resolved,
        "num_docs": len(docs),
        "num_chunks": n_chunks,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "question_type_distribution": dict(type_dist),
        "gold_chunks_path": os.path.join(args.output_dir, "gold_chunks.json"),
        "splits_path": os.path.join(args.output_dir, "splits.json"),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_index_finder] DONE. manifest: {json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()
