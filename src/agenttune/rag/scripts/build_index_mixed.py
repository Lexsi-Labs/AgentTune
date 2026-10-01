"""
Build the COMBINED retrieval index for the 50/50 mixed training run
(FinDER financial + CUAD synthetic legal).

train_grpo.py builds ONE backend per `--index_dir`, and `golden_chunk_recall`
is scored against chunk ids the agent's `search_corpus` actually retrieves —
so both domains' gold chunks must live in the same corpus.db. This script:

  1. FinDER side (identical to build_index_finder.py): load rows, recover
     tickers from the raw 10-K filings, ticker-level train/val split, corpus =
     deduplicated gold references (ticker-prefixed titles).
  2. CUAD side: load the contract corpus from `documents.jsonl`
     ({doc_id, title, text} — one doc per contract), plus the synthetic GRPO
     dataset(s) for gold chunk ids (parsed from `gold_path`).
  3. Chunk EVERYTHING at the same 1024/128 (build_index_finder's default and
     the CUAD synthesis chunk_size/overlap) and index the union into one
     SQLite FTS5 (BM25) corpus.db.
  4. Write gold_chunks.json (finder question ids: {gold_doc_ids,
     gold_chunk_ids}; cuad question ids: {gold_chunk_ids}) and splits.json
     (finder train/val ids + cuad train/val ids).
  5. VERIFY alignment: for every CUAD gold_path chunk id, the chunk text in
     the NEWLY BUILT index must match the dataset's own gold_passages text
     (whitespace-normalized). If the chunker params or source text drifted
     from the synthesis run, this fails loudly instead of silently zeroing
     the CUAD half's golden-chunk-recall reward.

Artifacts written to --output_dir:
  corpus.db         SQLite FTS5 index (chunk_id = {doc_id}::{idx})
  gold_chunks.json  {question_id: {...}} for BOTH question sets
  splits.json       {train_ids, val_ids, row_tickers, val_tickers,
                     cuad_train_ids, cuad_val_ids, stats}
  manifest.json     run parameters + counts

Usage:
    python -m agenttune.rag.scripts.build_index_mixed \
        --filings_dir /path/to/finder/10k \
        --cuad_documents /path/to/cuad_run/documents.jsonl \
        --cuad_train_dataset /path/to/cuad_run/dataset_grpo.jsonl \
        --cuad_val_dataset /path/to/cuad_run/dataset_val_grpo.jsonl \
        --output_dir rag_experiments/indexes/mixed
"""

import argparse
import json
import os
import re

from agenttune.rag.data.cuad import load_cuad_grpo_rows
from agenttune.rag.data.finder import (
    build_corpus_from_finder,
    finder_ticker_split,
    load_filings,
    load_finder_rows,
    map_rows_to_tickers,
    ref_key,
)
from agenttune.rag.retrieval.corpus_loader import CorpusDocument, chunk_corpus
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _finder_gold_chunks(rows, ref_to_doc_id, finder_docs, chunk_size, overlap):
    """Same gold-chunk assignment as build_index_finder: EVERY chunk of each
    gold reference doc is gold (which is why chunk-level recall == reference-
    level recall on the FinDER half)."""
    doc_chunk_counts = {
        d.doc_id: len(chunk_corpus([d], chunk_size=chunk_size, overlap=overlap))
        for d in finder_docs
    }
    gold = {}
    for row in rows:
        doc_ids = []
        for ref in row["references"]:
            doc_id = ref_to_doc_id[ref_key(ref)]
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
        chunk_ids = [f"{d}::{i}" for d in doc_ids for i in range(doc_chunk_counts[d])]
        gold[row["_id"]] = {"gold_doc_ids": doc_ids, "gold_chunk_ids": chunk_ids}
    return gold


def _cuad_docs_from_jsonl(path):
    docs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            docs.append(
                CorpusDocument(
                    doc_id=d["doc_id"], title=d.get("title", d["doc_id"]), text=d["text"]
                )
            )
    return docs


def _cuad_questions(rows):
    """{question_id: {"gold_chunk_ids": [...], "gold_passages": {chunk_id: text}}}."""
    out = {}
    for r in rows:
        gp = r.get("gold_path", "[]")
        if isinstance(gp, str):
            gp = json.loads(gp)
        passages = {p["chunk_id"]: p.get("text", "") for p in r.get("gold_passages", [])}
        out[r["question_id"]] = {"gold_chunk_ids": list(gp or []), "gold_passages": passages}
    return out


def _verify_cuad_alignment(questions, chunk_by_id):
    total, missing, mismatched = 0, 0, 0
    examples = []
    for qid, info in questions.items():
        for cid in info["gold_chunk_ids"]:
            total += 1
            if cid not in chunk_by_id:
                missing += 1
                if len(examples) < 5:
                    examples.append(f"{qid}: {cid} NOT IN INDEX")
                continue
            gold_text = info["gold_passages"].get(cid, "")
            if gold_text and _norm(chunk_by_id[cid]["text"]) != _norm(gold_text):
                mismatched += 1
                if len(examples) < 5:
                    examples.append(f"{qid}: {cid} TEXT MISMATCH")
    print(
        f"[verify_cuad] {total} gold chunks checked: "
        f"{missing} missing from index, {mismatched} text-mismatched"
    )
    if examples:
        for e in examples:
            print(f"  ! {e}")
    return missing == 0 and mismatched == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--filings_dir",
        required=True,
        help="extracted FinDER 10-k.zip dir (one <TICKER>.html per filing)",
    )
    ap.add_argument(
        "--cuad_json",
        default=None,
        help="CUADv1.json (official release) — PREFERRED CUAD corpus source. "
        "The synthesis chunked contracts from CUADv1.json's `context`, so "
        "chunking it again reproduces the exact gold_path chunk ids. Use "
        "this; --cuad_documents is a fallback for corpora dumped from a "
        "different source (verified: re-dumps can drift whitespace and "
        "break chunk alignment).",
    )
    ap.add_argument(
        "--cuad_documents",
        default=None,
        help="CUAD contract corpus JSONL: {doc_id, title, text} per line "
        "(fallback when --cuad_json is not given)",
    )
    ap.add_argument(
        "--cuad_train_dataset",
        required=True,
        help="CUAD synthetic GRPO train dataset (.json or .jsonl)",
    )
    ap.add_argument(
        "--cuad_val_dataset",
        default=None,
        help="CUAD synthetic GRPO val dataset (.json or .jsonl) — optional",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--chunk_overlap", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── FinDER side ─────────────────────────────────────────────────────────
    print("[build_index_mixed] loading FinDER rows + filings ...")
    rows = load_finder_rows()
    filings = load_filings(args.filings_dir)
    row_ticker = map_rows_to_tickers(rows, filings)
    n_resolved = sum(1 for t in row_ticker.values() if t is not None)
    print(f"  ticker resolved for {n_resolved}/{len(rows)} rows")
    train_rows, val_rows = finder_ticker_split(
        rows, row_ticker, val_frac=args.val_frac, seed=args.seed
    )
    val_tickers = sorted({row_ticker[r["_id"]] for r in val_rows} - {None})

    ref_to_ticker = {}
    for row in rows:
        t = row_ticker.get(row["_id"])
        if t is None:
            continue
        for ref in row["references"]:
            ref_to_ticker.setdefault(ref_key(ref), t)
    finder_docs, ref_to_doc_id = build_corpus_from_finder(rows, ref_to_ticker=ref_to_ticker)
    finder_gold = _finder_gold_chunks(
        rows, ref_to_doc_id, finder_docs, args.chunk_size, args.chunk_overlap
    )
    print(
        f"  {len(finder_docs)} finder reference docs; "
        f"{len(train_rows)} train / {len(val_rows)} val rows"
    )

    # ── CUAD side ───────────────────────────────────────────────────────────
    if args.cuad_json:
        from agenttune.rag.data.cuad import build_corpus_from_cuad, load_cuad_squad_json

        print(
            f"[build_index_mixed] loading CUAD corpus from {args.cuad_json} "
            f"(the synthesis source — guarantees chunk-id alignment) ..."
        )
        cuad_docs = build_corpus_from_cuad(load_cuad_squad_json(args.cuad_json))
    else:
        assert args.cuad_documents, "need --cuad_json or --cuad_documents"
        print(f"[build_index_mixed] loading CUAD corpus from {args.cuad_documents} ...")
        cuad_docs = _cuad_docs_from_jsonl(args.cuad_documents)
    print(f"  {len(cuad_docs)} CUAD contract docs")
    cuad_train_questions = _cuad_questions(load_cuad_grpo_rows(args.cuad_train_dataset))
    cuad_val_questions = (
        _cuad_questions(load_cuad_grpo_rows(args.cuad_val_dataset)) if args.cuad_val_dataset else {}
    )
    print(
        f"  {len(cuad_train_questions)} cuad train / {len(cuad_val_questions)} cuad val questions"
    )

    # ── Combined index ──────────────────────────────────────────────────────
    combined_docs = finder_docs + cuad_docs
    print(
        f"[build_index_mixed] chunking {len(combined_docs)} docs "
        f"({args.chunk_size}/{args.chunk_overlap}) and indexing ..."
    )
    chunks = chunk_corpus(combined_docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    backend = SQLiteFTSBackend(os.path.join(args.output_dir, "corpus.db"))
    backend.index(chunks)
    print(f"  indexed {len(chunks)} chunks")

    # ── Gold chunks + splits ────────────────────────────────────────────────
    gold_chunks = dict(finder_gold)
    for qid, info in {**cuad_train_questions, **cuad_val_questions}.items():
        gold_chunks[qid] = {"gold_chunk_ids": info["gold_chunk_ids"]}
    with open(os.path.join(args.output_dir, "gold_chunks.json"), "w") as f:
        json.dump(gold_chunks, f)

    splits = {
        "train_ids": [r["_id"] for r in train_rows],
        "val_ids": [r["_id"] for r in val_rows],
        "val_tickers": val_tickers,
        "row_tickers": row_ticker,
        "cuad_train_ids": list(cuad_train_questions),
        "cuad_val_ids": list(cuad_val_questions),
        "seed": args.seed,
        "val_frac": args.val_frac,
    }
    with open(os.path.join(args.output_dir, "splits.json"), "w") as f:
        json.dump(splits, f)

    # ── Alignment verification (loud failure, not silent zero-reward) ───────
    print("[build_index_mixed] verifying CUAD gold-chunk alignment ...")
    ok = _verify_cuad_alignment({**cuad_train_questions, **cuad_val_questions}, chunk_by_id)
    if not ok:
        raise SystemExit(
            "CUAD gold-path chunk ids do NOT align with the rebuilt index. "
            "Likely causes: --chunk_size/--chunk_overlap differs from the synthesis "
            "run, or cuad_documents text differs from what the synthesis chunked. "
            "Fix before training — otherwise the CUAD half gets 0 recall reward."
        )

    manifest = {
        "datasets": ["Linq-AI-Research/FinDER", "CUAD-synthetic"],
        "backend": "sqlite",
        "output_dir": args.output_dir,
        "num_finder_rows": len(rows),
        "num_finder_train": len(train_rows),
        "num_finder_val": len(val_rows),
        "num_cuad_train": len(cuad_train_questions),
        "num_cuad_val": len(cuad_val_questions),
        "num_docs": len(combined_docs),
        "num_chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "cuad_alignment_verified": bool(ok),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_index_mixed] DONE. manifest: {json.dumps(manifest, indent=2)}")


if __name__ == "__main__":
    main()
