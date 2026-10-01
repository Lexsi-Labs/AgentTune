"""
Stage 5 / orchestrator — docs in → GRPO-ready multi-hop QA dataset out.

Mirrors `scripts/build_index.py`'s CLI shape. Runs every stage, dumps a
tabular artifact at each step (CSV/Parquet) so the pipeline is inspectable
and debuggable end-to-end:

  out_dir/
    stage0_chunks.csv         # input chunks + entities + summaries
    stage0_graph_edges.csv    # typed edges (exact/contextual/abstract)
    stage1_paths.csv          # sampled reasoning paths
    stage2_generated.csv      # generated QA + generation call metadata
    stage3_verified.csv       # verification results + discard reasons
    stage4_balanced.csv       # difficulty-labeled + balanced
    dataset_final.csv         # accepted QASamples (GRPO-ready)
    metrics.json              # full evaluation suite
    cost_summary.json         # aggregate token/cost/latency
    manifest.json             # full run config for reproducibility

The final dataset matches `hotpotqa.to_grpo_dataset`'s schema (prompt,
gold_answer, question_id) + a gold_path field for RL partial-credit rewards.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from .difficulty import balance_by_matrix, label_difficulty, reassign_difficulty_cells
from .evaluate import (
    answerability,
    cost_summary,
    coverage,
    difficulty_distribution,
    diversity,
    faithfulness,
    multi_hop_necessity,
    retrieval_recall,
    retrieval_recall_dense,
)
from .generate import generate_batch
from .graph import annotate_chunks, build_graph, chunk_documents, embed_chunks
from .io_utils import dump_table
from .paths import sample_paths
from .verify import verify_batch

logger = logging.getLogger(__name__)


def _rows_from_chunks(chunks):
    return [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "entities": json.dumps(c.entities, ensure_ascii=False),
            "keyphrases": json.dumps(c.keyphrases, ensure_ascii=False),
            "summary": c.metadata.get("summary", ""),
        }
        for c in chunks
    ]


def _rows_from_edges(edges):
    return [
        {
            "source": e.source,
            "target": e.target,
            "type": e.type,
            "weight": e.weight,
            "evidence": e.evidence,
        }
        for e in edges
    ]


def _rows_from_paths(paths):
    return [
        {
            "path_id": p.path_id,
            "hop_count": p.hop_count,
            "question_type": p.question_type,
            "specificity": p.specificity,
            "chunk_ids": json.dumps(p.chunk_ids),
            "entities": json.dumps(p.entities, ensure_ascii=False),
        }
        for p in paths
    ]


def _rows_from_samples(samples, *, include_rejected=True):
    rows = []
    for s in samples:
        if not include_rejected and s.status in ("rejected",):
            continue
        # path_entities: the named entities the question traverses (from the
        # reasoning path) — useful for analysis/filtering by topic.
        path_ents = list(getattr(s.path, "entities", []) or []) if s.path else []
        rows.append(
            {
                "sample_id": s.sample_id,
                "question": s.question,
                "answer": s.answer,
                "gold_chunk_ids": json.dumps(s.gold_chunk_ids),
                "gold_reasoning": s.gold_reasoning,
                "hop_count": s.hop_count,
                "question_type": getattr(s, "question_type", ""),
                "specificity": getattr(s, "specificity", ""),
                "path_entities": json.dumps(path_ents, ensure_ascii=False),
                "retrieval_necessity": s.retrieval_necessity,
                "retrieval_necessity_pass": s.retrieval_necessity_pass,
                "chain_dependency": s.chain_dependency,
                "chain_dependency_pass": s.chain_dependency_pass,
                "answer_grounded_span": getattr(s, "answer_grounded_span", None),
                "revision_count": s.revision_count,
                "original_question": s.original_question,
                "retrieval_difficulty": s.retrieval_difficulty,
                "difficulty_cell": s.difficulty_cell,
                "answerability_f1": getattr(s, "answerability_f1", None),
                "answerability_f1_relaxed": getattr(s, "answerability_f1_relaxed", None),
                "answerable": getattr(s, "answerable", None),
                "faithfulness_score": getattr(s, "faithfulness_score", None),
                "status": s.status,
                "discard_reason": s.discard_reason,
                "n_llm_calls": len(s.llm_calls),
                "total_tokens": sum(c.prompt_tokens + c.completion_tokens for c in s.llm_calls),
                "cost_usd": round(sum(c.cost_usd for c in s.llm_calls), 6),
                "created_at": getattr(s, "created_at", ""),
            }
        )
    return rows


def _grpo_rows(samples, system_prompt=None, chunks_by_id=None):
    """Final dataset rows: match hotpotqa.to_grpo_dataset schema + gold_path.

    `system_prompt`: defaults to hotpotqa's (backward compatible). Pass
    `cuad.get_cuad_system_prompt()` for the legal track.

    `chunks_by_id`: if given, each row also carries `gold_passages` — the
    gold-path chunk TEXT (chunk_id + text), so the dataset is self-contained
    for closed-context EVAL of arbitrary models (read the passages, answer,
    score against gold_answer), independent of the agent retrieval harness.
    """
    if system_prompt is None:
        from ..data.hotpotqa import get_system_prompt

        system_prompt = get_system_prompt("sqlite")
    out = []
    for s in samples:
        if s.status not in ("accepted", "revised_accepted"):
            continue
        row = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": s.question},
            ],
            "gold_answer": s.answer,
            "question_id": s.sample_id,
            "gold_path": json.dumps(s.gold_chunk_ids),
            "hop_count": s.hop_count,
            "difficulty_cell": s.difficulty_cell,
            "answerable": getattr(s, "answerable", None),
            "answerability_f1_relaxed": getattr(s, "answerability_f1_relaxed", None),
        }
        if chunks_by_id is not None:
            row["gold_passages"] = [
                {"chunk_id": cid, "text": chunks_by_id[cid].text}
                for cid in s.gold_chunk_ids
                if cid in chunks_by_id
            ]
        out.append(row)
    return out


def _rebuild_graph_from_edges(chunks, edges):
    """Rebuild a networkx graph from cached chunks + edges (fallback if the
    graph object itself wasn't pickled)."""
    import networkx as nx

    G = nx.Graph()
    for c in chunks:
        G.add_node(
            c.chunk_id,
            text=c.text,
            entities=c.entities,
            summary=c.metadata.get("summary", ""),
            doc_id=c.doc_id,
        )
    for e in edges:
        G.add_edge(e.source, e.target, type=e.type, weight=e.weight, evidence=e.evidence)
    return G


def run_pipeline(
    *,
    docs,
    llm,
    embedder,
    out_dir,
    backend="sqlite",
    per_hop=30,
    target_per_cell=15,
    eval_judge_llm=None,
    index_dir=None,
    seed=42,
    val_fraction=0.2,
    split_level="chunk",
    use_cache=True,
    chunk_size=512,
    chunk_overlap=64,
    chunk_with_offsets=False,
    same_doc_only=False,
    max_entity_doc_frequency=0.3,
    prompt_template=None,
    system_prompt=None,
    require_answerable=False,
    max_workers=4,
    max_paths_per_endpoints=3,
    max_paths_considered=20000,
):
    """Run the full pipeline. Returns the final accepted samples.

    `same_doc_only`, `chunk_size`/`chunk_overlap`/`chunk_with_offsets`,
    `max_entity_doc_frequency`, `prompt_template`: the CUAD/legal-track
    knobs (NLLP_SynthData.md Part A). All default to the original
    HotpotQA/FinDER behavior (cross-document edges, 512/64 chunking, generic
    prompt) — pass `same_doc_only=True` + `ANSWER_FIRST_PROMPT_LEGAL` +
    `chunk_size=1024, chunk_overlap=128, chunk_with_offsets=True` for the
    legal-contract track.

    `require_answerable=True` gates the final dataset on the per-sample
    `answerable` flag (solver F1 >= 0.5 on the FULL gold path, relaxed
    scorer): questions that can't be answered from their own gold chunks are
    dropped from the shipped dataset (they'd poison RL training) and counted
    in the manifest as `n_dropped_unanswerable`. Default False keeps the
    original HotpotQA/FinDER behavior.

    Post-Stage-5, runs a train/val split by gold-chunk disjointness
    (leakage prevention — see `split.split_by_gold_chunks`). `val_fraction`
    is the target validation share; `split_level` is "chunk" (default, gold
    chunk-id disjointness) or "entity" (path-entity disjointness, stricter).
    Emits `dataset_train.csv` / `dataset_val.csv` + per-split GRPO jsonl.

    Caching policy (only Stage 0 is cached — by design, not omission):
      - Stage 0 (entity extraction + embedding + graph build) is the ONE
        expensive LLM-heavy stage that's deterministic in the corpus (not in
        the prompt/model/seed). It's cached to `stage0_cache.pkl` so reruns
        skip ~hundreds of LLM calls. `use_cache=False` (or the `--fresh` CLI
        flag) forces a rebuild — the staleness escape hatch after a model or
        chunk-size change.
      - Stages 1-5 are deliberately NOT cached. Stage 1 (path sampling) and
        Stage 4 (difficulty) are cheap pure-CPU/embed steps where a cache only
        adds staleness risk. Stages 2/3/5 are LLM calls whose output you WANT
        to be fresh when the prompt, model, or temperature changes — a stale
        cache there would silently serve last week's questions/verdicts.
        Mid-batch resume for a failed Stage 3 is a separate (deferred) concern;
        whole-stage skip is the wrong granularity for it.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "llm_model": getattr(llm, "model", "?"),
        "embedder": getattr(embedder, "model_name", "?"),
        "per_hop": per_hop,
        "backend": backend,
        "use_cache": use_cache,
    }

    # ── Stage 0: corpus → typed graph ──────────────────────────────────────
    # Caching: Stage 0 (entity extraction + embedding + graph build) is the
    # expensive LLM-heavy part. Cache its result to a pickle so reruns during
    # testing skip straight to Stage 1 (path sampling) without re-calling the
    # LLM. Delete `stage0_cache.pkl` (or pass use_cache=False / --fresh) to
    # force a fresh Stage 0 — do this after a model/chunk-size change, since a
    # stale cache would serve the old graph.
    import pickle

    cache_path = out / "stage0_cache.pkl"
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        chunks = cached["chunks"]
        edges = cached["edges"]
        G = cached.get("graph") or _rebuild_graph_from_edges(chunks, edges)
        logger.info(f"[stage0] loaded from cache: {len(chunks)} chunks, {len(edges)} edges")
    else:
        chunks = chunk_documents(
            docs, chunk_size=chunk_size, overlap=chunk_overlap, with_offsets=chunk_with_offsets
        )
        chunks = annotate_chunks(chunks, llm, max_workers=max_workers)
        chunks = embed_chunks(chunks, embedder)
        edges, G = build_graph(
            chunks,
            llm,
            same_doc_only=same_doc_only,
            max_entity_doc_frequency=max_entity_doc_frequency,
            max_workers=max_workers,
        )
        with open(cache_path, "wb") as f:
            pickle.dump({"chunks": chunks, "edges": edges, "graph": G}, f)
        logger.info(
            f"[stage0] built + cached: {len(chunks)} chunks, {len(edges)} edges "
            f"(use_cache={use_cache})"
        )
    dump_table(_rows_from_chunks(chunks), str(out / "stage0_chunks.csv"))
    dump_table(_rows_from_edges(edges), str(out / "stage0_graph_edges.csv"))
    manifest["n_chunks"] = len(chunks)
    manifest["n_edges"] = len(edges)

    # ── Stage 1: sample paths ─────────────────────────────────────────────
    # `max_paths_per_endpoints` controls path diversity per (start,end) pair:
    # default 3 was starving 2/3-hop buckets on same-doc CUAD graphs (the
    # scarce, most-bridgeable hop counts), so runs targeting thousands of
    # samples raise it (10) to unlock the under-supplied hop counts.
    chunks_by_id = {c.chunk_id: c for c in chunks}
    paths = sample_paths(
        G,
        chunks_by_id,
        per_hop=per_hop,
        seed=seed,
        max_paths_per_endpoints=max_paths_per_endpoints,
        max_paths_considered=max_paths_considered,
    )
    dump_table(_rows_from_paths(paths), str(out / "stage1_paths.csv"))
    manifest["n_paths"] = len(paths)

    # ── Stage 2: answer-first generation ──────────────────────────────────
    # Crash-safe incremental checkpoints (see checkpoint.py): each completed
    # sample is appended as it finishes, so a crash mid-stage resumes instead
    # of re-spending the whole stage. Disabled by `--fresh`/use_cache=False —
    # a fully-fresh run must not reuse cached stages either.
    ckpt2 = str(out / "stage2_checkpoint.pkl") if use_cache else None
    if prompt_template is not None:
        samples = generate_batch(
            paths,
            chunks_by_id,
            llm,
            prompt_template=prompt_template,
            max_workers=max_workers,
            checkpoint_path=ckpt2,
        )
    else:
        samples = generate_batch(
            paths, chunks_by_id, llm, max_workers=max_workers, checkpoint_path=ckpt2
        )
    dump_table(_rows_from_samples(samples), str(out / "stage2_generated.csv"))
    manifest["n_generated"] = len(samples)

    # ── build a retrieval index for Stage 3 (reuse our SearchBackend) ──────
    # Chunks are already chunked (Stage 0), so index the chunk dicts directly
    # rather than re-chunking via corpus_loader.build_index.
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "text": c.text,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    if backend == "sqlite":
        from ..retrieval.sqlite_fts import SQLiteFTSBackend

        sb = SQLiteFTSBackend(str(out / "corpus.sqlite"))
    else:
        from ..retrieval.chroma_backend import ChromaBackend

        sb = ChromaBackend(persist_dir=str(out / "corpus.chroma"))
    sb.index(chunk_dicts)

    # ── Stage 3: closed-loop verification ─────────────────────────────────
    ckpt3 = str(out / "stage3_checkpoint.pkl") if use_cache else None
    samples = verify_batch(
        samples, chunks_by_id, sb, llm, max_workers=max_workers, checkpoint_path=ckpt3
    )
    dump_table(_rows_from_samples(samples), str(out / "stage3_verified.csv"))
    # dataset_all.csv: EVERY sample (accepted + revised + rejected) with full
    # provenance — question, answer, gold chunks, reasoning, both verification
    # scores + pass flags, revision count, original-vs-revised question,
    # difficulty cell, status, AND discard_reason (why rejected ones failed).
    # This is the complete audit trail: you can see exactly which questions
    # were dropped and why (retrieval_leak / non_load_bearing_hop / empty_qa /
    # generate_parse_failure / revise_failed).
    dump_table(_rows_from_samples(samples, include_rejected=True), str(out / "dataset_all.csv"))
    manifest["n_accepted"] = sum(1 for s in samples if s.status in ("accepted", "revised_accepted"))
    manifest["n_rejected"] = sum(1 for s in samples if s.status == "rejected")

    # ── Stage 4: difficulty + balance ─────────────────────────────────────
    samples = label_difficulty(samples, chunks_by_id, embedder)
    # Re-bucket difficulty into 3 levels calibrated to this run's distribution
    # (fixed-0.5 "hard" was degenerate on legal text — everything easy; see
    # difficulty.reassign_difficulty_cells).
    reassign_difficulty_cells(samples)
    accepted = [s for s in samples if s.status in ("accepted", "revised_accepted")]
    balanced = balance_by_matrix(accepted, target_per_cell=target_per_cell, seed=seed)
    dump_table(_rows_from_samples(samples, include_rejected=True), str(out / "stage4_balanced.csv"))

    # ── Stage 5: evaluation, THEN final dataset dump ──────────────────────
    # Eval (answerability + faithfulness) runs FIRST so the per-sample scores
    # are recorded on each QASample before the final CSV is written — that way
    # dataset_final.csv + dataset_all.csv carry the eval scores as columns.
    metrics = {
        **multi_hop_necessity(balanced),
        **retrieval_recall(balanced, sb),
        **retrieval_recall_dense(balanced, chunks_by_id, embedder),
        **diversity(balanced, embedder),
        **coverage(balanced, len(chunks)),
        **difficulty_distribution(balanced),
        **cost_summary(balanced),
    }
    # answerability + faithfulness need LLM calls — run on the balanced set
    metrics.update(answerability(balanced, chunks_by_id, llm, max_workers=max_workers))
    if eval_judge_llm is not None:
        metrics.update(
            faithfulness(balanced, chunks_by_id, eval_judge_llm, max_workers=max_workers)
        )

    # Quality gate: drop questions the solver can't answer from their own
    # gold path (relaxed F1 < 0.5). They'd poison RL — a model can never get
    # correctness reward on them no matter how well it searches.
    if require_answerable:
        n_before = len(balanced)
        balanced = [s for s in balanced if s.answerable]
        n_dropped = n_before - len(balanced)
        manifest["n_dropped_unanswerable"] = n_dropped
        metrics["n_dropped_unanswerable"] = n_dropped
        logger.info(
            f"[gate] answerable gate: kept {len(balanced)}/{n_before} "
            f"(dropped {n_dropped} unanswerable)"
        )

    # Now dump the final datasets WITH the per-sample eval scores populated.
    grpo_rows = _grpo_rows(balanced, system_prompt=system_prompt, chunks_by_id=chunks_by_id)
    dump_table(_rows_from_samples(balanced), str(out / "dataset_final.csv"))
    # re-dump dataset_all.csv so it also carries the eval scores (accepted rows
    # get them; rejected rows stay null since eval only ran on the balanced set)
    dump_table(_rows_from_samples(samples, include_rejected=True), str(out / "dataset_all.csv"))
    # Also dump the GRPO-ready dataset as JSONL (prompt is a list-of-dicts)
    with open(out / "dataset_grpo.jsonl", "w") as f:
        for r in grpo_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Self-contained eval view (NLLP_SynthData.md A6 Axis 1 eval): question +
    # gold passages + gold answer per row, so ANY model can be benchmarked
    # closed-context without the retrieval harness. Same rows as the grpo
    # file; `scripts/eval_generated_dataset.py` consumes it.
    with open(out / "dataset_eval.jsonl", "w") as f:
        for r in grpo_rows:
            f.write(
                json.dumps(
                    {
                        "question_id": r["question_id"],
                        "question": r["prompt"][-1]["content"],
                        "gold_answer": r["gold_answer"],
                        "gold_passages": r.get("gold_passages", []),
                        "hop_count": r["hop_count"],
                        "difficulty_cell": r["difficulty_cell"],
                        "answerability_f1_relaxed": r.get("answerability_f1_relaxed"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    with open(out / "cost_summary.json", "w") as f:
        json.dump(cost_summary(samples), f, indent=2, default=str)

    # ── Post-step: train/val split by gold-chunk disjointness ──────────────
    # Leakage prevention: a random row-split lets two questions that traverse
    # overlapping gold chunks land in different splits → val leaks into train.
    # Split by gold-chunk disjointness (connected components on the
    # sample-sharing graph) so no chunk appears in both splits' gold paths.
    # See split.py. Runs last so the split is over the final balanced set and
    # the per-sample eval scores are already populated.
    from .split import write_split_artifacts

    split_report = write_split_artifacts(
        balanced,
        str(out),
        val_fraction=val_fraction,
        level=split_level,
        seed=seed,
        backend=backend,
        system_prompt=system_prompt,
        chunks_by_id=chunks_by_id,
    )
    if split_report is not None:
        manifest["split"] = split_report

    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["metrics"] = metrics
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return balanced


def main():
    ap = argparse.ArgumentParser(description="Synthetic multi-hop RAG QA dataset generator")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--corpus",
        default="hotpot",
        choices=["hotpot", "news", "devdocs"],
        help="which corpus to generate over (hotpot uses our existing loader)",
    )
    ap.add_argument("--train_size", type=int, default=50, help="rows of corpus to load")
    ap.add_argument("--per_hop", type=int, default=30)
    ap.add_argument("--target_per_cell", type=int, default=15)
    ap.add_argument("--backend", default="sqlite", choices=["sqlite", "chroma"])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument(
        "--embedder",
        default="bge-m3",
        choices=["bge-m3", "qwen3-8b"],
        help="bge-m3 (default, ~2GB VRAM, robust) or qwen3-8b (~30GB, best quality)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="force a fresh Stage 0 rebuild (ignore stage0_cache.pkl). "
        "Use after a model/chunk-size change — a stale cache would "
        "serve the old graph. Only Stage 0 is cached; Stages 1-5 "
        "are always fresh by design.",
    )
    ap.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="target validation share for the gold-chunk-disjoint split",
    )
    ap.add_argument(
        "--split_level",
        default="chunk",
        choices=["chunk", "entity"],
        help="disjointness level: chunk (default, gold chunk-id) or "
        "entity (path-entity, stricter; may shrink usable split)",
    )
    args = ap.parse_args()

    # Build the real clients (production path on Colab)
    from .embeddings import BGEM3Embedder, Qwen3Embedder
    from .llm_client import GroqLLMClient

    llm = GroqLLMClient(model="qwen/qwen3.6-27b")
    if args.embedder == "qwen3-8b":
        embedder = Qwen3Embedder(device="cuda")
    else:
        embedder = BGEM3Embedder(device="cuda")

    # Load corpus — reuse our hotpotqa loader for the wiki corpus
    if args.corpus == "hotpot":
        from ..data.hotpotqa import build_corpus_from_hotpotqa, load_hotpotqa_splits

        train, _ = load_hotpotqa_splits(train_size=args.train_size, eval_size=0)
        docs = build_corpus_from_hotpotqa(train, max_docs=args.train_size * 4)
    else:
        raise SystemExit(f"corpus {args.corpus!r} not wired yet (see plan §6)")

    run_pipeline(
        docs=docs,
        llm=llm,
        embedder=embedder,
        out_dir=args.out_dir,
        backend=args.backend,
        per_hop=args.per_hop,
        target_per_cell=args.target_per_cell,
        use_cache=not args.fresh,
        val_fraction=args.val_fraction,
        split_level=args.split_level,
    )


if __name__ == "__main__":
    main()
