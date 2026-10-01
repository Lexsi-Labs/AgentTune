"""
CLI entrypoint for the CUAD within-document multi-hop track (NLLP_SynthData.md
Part A). Mirrors `synthesis/build_dataset.py`'s CLI shape, wired for CUAD:
same-document-only edges, 1024/128 chunking with offset tracking, the
legal-register generation prompt.

Validated (2026-08-12) on 14 diverse real CUAD contracts / 984 chunks / 40
sampled within-document paths: 0 cross-document edges (assert-enforced),
33/40 (82.5%) accepted by real Stage 3 verification, 7/7 rejections
correctly attributed to `non_load_bearing_hop` (chain-dependency catching
genuinely non-multi-hop questions) with zero `retrieval_leak` rejections.
See NLLP_SynthData.md Part A for the full validation writeup.

Official split (wired 2026-08-13): when `--cuad_json` + the official
`--official_train_json`/`--official_test_json` are provided, the corpus,
category spans, and train/test contract partition all come from the official
CUAD-QA release (verified: 408/102 contracts, zero overlap, spans exact) and
synthesis never touches test contracts. The txt-dir + hash-split path remains
as fallback. See `rag/data/cuad.py` module docstring.

Usage (official release):
    python -m agenttune.rag.scripts.build_cuad_dataset \\
        --cuad_json ~/cuad_data/CUADv1.json \\
        --official_train_json ~/cuad_data/train_separate_questions.json \\
        --official_test_json ~/cuad_data/test.json \\
        --out_dir ~/cuad_full_run \\
        --n_synthesis_contracts 35 --per_hop 15 \\
        --llm_base_url https://<openai-compatible-endpoint>/v1 \\
        --llm_model deepseek-v4-flash-0731 --llm_api_key $DEEPSEEK_API_KEY

Usage (fallback, txt dir + hash split):
    python -m agenttune.rag.scripts.build_cuad_dataset \\
        --cuad_dir ~/cuad_contracts --out_dir ~/cuad_full_run \\
        --n_synthesis_contracts 35 --per_hop 15 \\
        --llm_base_url https://<openai-compatible-endpoint>/v1 \\
        --llm_model deepseek-v4-flash-0731 --llm_api_key $DEEPSEEK_API_KEY
"""

from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="CUAD within-document multi-hop dataset generator")
    ap.add_argument(
        "--cuad_dir",
        default=None,
        help="directory containing CUAD full_contract_txt/ (or any dir of .txt contracts) "
        "— fallback source; prefer --cuad_json (official release)",
    )
    ap.add_argument(
        "--cuad_json",
        default=None,
        help="official CUAD-QA JSON with all contracts (CUADv1.json) — preferred source: "
        "gives contract text + category spans in one coordinate system",
    )
    ap.add_argument(
        "--official_train_json",
        default=None,
        help="official train split JSON (train_separate_questions.json) — enables the "
        "verified contract-disjoint official split instead of the hash fallback",
    )
    ap.add_argument(
        "--official_test_json",
        default=None,
        help="official test split JSON (test.json) — held-out contracts for Axis-2 "
        "native eval; never touched by synthesis",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--n_synthesis_contracts",
        type=int,
        default=35,
        help="how many TRAIN-split contracts to run Stage 0 graph-building "
        "over (bounded on purpose — see NLLP_SynthData.md A1; the "
        "retrieval index below can be much larger)",
    )
    ap.add_argument(
        "--contract_start",
        type=int,
        default=0,
        help="index of the first synthesis contract (0 = first train "
        "contract). Set >0 to build over NEW contracts after an "
        "earlier run already covered the first N — lets multiple "
        "runs accumulate disjoint contract sets into one 4K dataset",
    )
    ap.add_argument(
        "--n_retrieval_contracts",
        type=int,
        default=0,
        help="how many TRAIN-split contracts to index for retrieval "
        "(0 = all train contracts). Should be >= n_synthesis_contracts.",
    )
    ap.add_argument(
        "--val_fraction",
        type=float,
        default=0.3,
        help="fraction of contracts held out as the NATIVE CUAD test set "
        "(disjoint from synthesis — Axis 2 eval, NLLP_SynthData.md A6)",
    )
    ap.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="seed for the contract-level train/test split (deterministic fallback "
        "— see cuad.split_contracts; swap for the official CUAD-QA split when verified)",
    )
    ap.add_argument("--per_hop", type=int, default=15)
    ap.add_argument(
        "--max_paths_per_endpoints",
        type=int,
        default=3,
        help="distinct routings kept per (start,end) pair in path "
        "sampling. 3 (default) starves the scarce 2/3-hop "
        "buckets on same-doc CUAD graphs; raise to 10 for "
        "thousands-of-samples runs.",
    )
    ap.add_argument(
        "--max_paths_considered",
        type=int,
        default=20000,
        help="total paths collected before per-hop capping "
        "(bounds the exponential path enumeration)",
    )
    ap.add_argument("--target_per_cell", type=int, default=10)
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="grounded in real clause-length measurement — NLLP_SynthData.md A3",
    )
    ap.add_argument("--chunk_overlap", type=int, default=128)
    ap.add_argument(
        "--max_entity_doc_frequency",
        type=float,
        default=0.3,
        help="drop entities appearing in more than this fraction of a document's "
        "own chunks (min 5 occurrences) — prevents party-name over-connection, "
        "see graph._drop_high_doc_frequency_entities",
    )
    ap.add_argument("--backend", default="sqlite", choices=["sqlite", "chroma"])
    ap.add_argument(
        "--embedder",
        default="qwen3-0.6b",
        choices=["qwen3-0.6b", "bge-m3"],
        help="embedding model: qwen3-0.6b (Qwen3-Embedding-0.6B, default — MTEB "
        "retrieval leader among small models, ~0.7GB fp16 on GPU) or bge-m3",
    )
    ap.add_argument(
        "--embedder_device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="qwen3-0.6b is small enough to run on the GPU with headroom; "
        "bge-m3 also fits. cpu only if the box's CUDA/driver is mismatched",
    )
    ap.add_argument("--llm_base_url", required=True)
    ap.add_argument("--llm_model", required=True)
    ap.add_argument("--llm_api_key", required=True)
    ap.add_argument(
        "--llm_max_workers",
        type=int,
        default=8,
        help="parallel LLM calls per batch (DeepSeek official API allows 2500 "
        "concurrency; production runs use 1600 — set this to 64-128 for "
        "generation-scale runs; 4 was the old fixed value)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="force a fresh Stage 0 rebuild, ignoring stage0_cache.pkl",
    )
    args = ap.parse_args()

    from ..data.cuad import (
        build_corpus_from_cuad,
        get_cuad_system_prompt,
        labeled_spans_from_records,
        load_cuad_contracts,
        load_cuad_squad_json,
        official_split_contracts,
        split_contracts,
        tag_chunks_with_categories,
    )
    from ..synthesis import (
        ANSWER_FIRST_PROMPT_LEGAL,
        BGEM3Embedder,
        OpenAICompatLLMClient,
        Qwen3Embedding06B,
        run_pipeline,
    )

    # ── Source: official JSON (preferred) or txt dir (fallback) ────────────
    if args.cuad_json:
        print(f"[load] reading official CUAD-QA JSON from {args.cuad_json}...")
        contracts = load_cuad_squad_json(args.cuad_json)
        print(f"[load] {len(contracts)} contracts (with category spans)")
    elif args.cuad_dir:
        print(f"[load] reading contracts from {args.cuad_dir}...")
        contracts = load_cuad_contracts(args.cuad_dir)
        print(f"[load] {len(contracts)} contracts found")
    else:
        raise SystemExit("pass --cuad_json (official release) or --cuad_dir (txt fallback)")
    if not contracts:
        raise SystemExit("no contracts loaded")

    # ── Split: official (verified disjoint) when JSONs given, else hash ────
    if args.official_train_json and args.official_test_json:
        train_ids, test_ids = official_split_contracts(
            args.official_train_json, args.official_test_json
        )
        by_id = {c["contract_id"]: c for c in contracts}
        missing = [i for i in train_ids + test_ids if i not in by_id]
        if missing:
            print(
                f"[warn] {len(missing)} official-split contract(s) missing from "
                f"the loaded corpus (source mismatch?) — first: {missing[:3]}"
            )
        train_contracts = [by_id[i] for i in train_ids if i in by_id]
        test_contracts = [by_id[i] for i in test_ids if i in by_id]
        print(
            f"[split] OFFICIAL CUAD-QA split: {len(train_contracts)} train / "
            f"{len(test_contracts)} test contracts "
            f"(verified contract-disjoint — synthesis never touches test)"
        )
    else:
        train_contracts, test_contracts = split_contracts(
            contracts, val_fraction=args.val_fraction, seed=args.split_seed
        )
        print(
            f"[split] {len(train_contracts)} train / {len(test_contracts)} test contracts "
            f"(hash fallback split — pass the official train/test JSONs to use the "
            f"verified CUAD-QA split instead)"
        )

    n_synth = min(args.n_synthesis_contracts, len(train_contracts) - args.contract_start)
    synthesis_contracts = train_contracts[args.contract_start : args.contract_start + n_synth]
    n_retrieval = args.n_retrieval_contracts or len(train_contracts)
    retrieval_contracts = train_contracts[: min(n_retrieval, len(train_contracts))]
    print(
        f"[scope] synthesis subset: {len(synthesis_contracts)} contracts "
        f"(Stage 0 graph-building); retrieval index: {len(retrieval_contracts)} contracts"
    )

    docs = build_corpus_from_cuad(synthesis_contracts)

    llm = OpenAICompatLLMClient(
        model=args.llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        timeout=60,
        max_retries=0,
    )
    if args.embedder == "qwen3-0.6b":
        embedder = Qwen3Embedding06B(
            device=args.embedder_device,
            query_instruction="Given a legal contract, retrieve the contract "
            "passages that answer the question",
        )
    else:
        embedder = BGEM3Embedder(device=args.embedder_device)

    os.makedirs(args.out_dir, exist_ok=True)
    balanced = run_pipeline(
        docs=docs,
        llm=llm,
        embedder=embedder,
        out_dir=args.out_dir,
        backend=args.backend,
        per_hop=args.per_hop,
        target_per_cell=args.target_per_cell,
        max_workers=args.llm_max_workers,
        max_paths_per_endpoints=args.max_paths_per_endpoints,
        max_paths_considered=args.max_paths_considered,
        use_cache=not args.fresh,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_with_offsets=True,
        same_doc_only=True,
        max_entity_doc_frequency=args.max_entity_doc_frequency,
        prompt_template=ANSWER_FIRST_PROMPT_LEGAL,
        system_prompt=get_cuad_system_prompt(),
        val_fraction=0.2,  # this is the SYNTHETIC-question train/val split (Axis 1),
        # independent of the contract-level train/test split above
        require_answerable=True,  # hard gate: drop questions a solver can't answer
        # with the full gold path (they'd poison training)
    )
    print(f"\n[done] {len(balanced)} balanced/accepted questions written to {args.out_dir}")

    # ── Category tagging (NLLP_SynthData.md §A2, metadata) ─────────────────
    # Tag every gold chunk with the CUAD categories whose labeled spans overlap
    # it (needs the official JSON's spans — only available on the JSON path),
    # then re-dump dataset_final/dataset_all/grpo with a `cuad_categories`
    # column: the categories each question's reasoning path bridges, e.g.
    # "Confidentiality" + "Survival".
    if args.cuad_json:
        import csv as _csv
        import json as _json

        from ..synthesis.graph import chunk_documents
        from ..synthesis.io_utils import dump_table

        spans = labeled_spans_from_records(contracts)
        # Re-chunk deterministically (same params as stage0 → identical chunk
        # ids); no LLM, no embeddings needed for offset-based tagging.
        tagged_chunks = chunk_documents(
            docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap, with_offsets=True
        )
        tag_chunks_with_categories(tagged_chunks, spans)
        cat_by_chunk = {c.chunk_id: c.metadata.get("cuad_categories", []) for c in tagged_chunks}

        def _cats_for_golds(golds):
            return sorted({c for g in golds for c in cat_by_chunk.get(g, [])})

        # dataset_final.csv: re-dump from the balanced samples WITH tags.
        from ..synthesis.build_dataset import _rows_from_samples

        fin_rows = _rows_from_samples(balanced, include_rejected=False)
        for r in fin_rows:
            r["cuad_categories"] = _json.dumps(_cats_for_golds(_json.loads(r["gold_chunk_ids"])))
        dump_table(fin_rows, os.path.join(args.out_dir, "dataset_final.csv"))

        # dataset_all.csv: AUGMENT the full audit dump in place (every row,
        # incl. rejected + discard_reason — never overwrite the audit trail
        # with the balanced subset).
        all_path = os.path.join(args.out_dir, "dataset_all.csv")
        with open(all_path, newline="") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames + ["cuad_categories"]
            all_rows = []
            for row in reader:
                golds = _json.loads(row.get("gold_chunk_ids") or "[]")
                row["cuad_categories"] = _json.dumps(_cats_for_golds(golds))
                all_rows.append(row)
        with open(all_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)

        for fn in ("dataset_grpo.jsonl", "dataset_train_grpo.jsonl", "dataset_val_grpo.jsonl"):
            p = os.path.join(args.out_dir, fn)
            if not os.path.exists(p):
                continue
            lines = []
            with open(p) as f:
                for line in f:
                    row = _json.loads(line)
                    golds = _json.loads(row.get("gold_path", "[]") or "[]")
                    row["cuad_categories"] = _cats_for_golds(golds)
                    lines.append(_json.dumps(row, ensure_ascii=False))
            with open(p, "w") as f:
                f.write("\n".join(lines) + "\n")
        n_tagged = sum(1 for c in tagged_chunks if c.metadata.get("cuad_categories"))
        print(
            f"[tags] {n_tagged}/{len(tagged_chunks)} chunks tagged with CUAD "
            f"categories; questions now carry a `cuad_categories` column"
        )

    print(
        f"[next] build the retrieval index over {len(retrieval_contracts)} contracts "
        f"(scripts/build_index_from_docs.py or equivalent) and the native-CUAD Axis-2 "
        f"eval harness over the {len(test_contracts)} held-out test contracts — "
        f"see NLLP_SynthData.md A6/A8 for the remaining steps."
    )


if __name__ == "__main__":
    main()
