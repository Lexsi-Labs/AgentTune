"""
Static-corpus append mode — grow an existing dataset with new unique samples.

The corpus is static (user confirmed): the Stage-0 graph and retrieval index
never change, so there's no incremental-graph-update problem. What we DO want
is to grow a dataset over multiple runs without regenerating the same
questions — i.e. load the questions an existing `dataset_all.csv` already
holds, generate new samples via the seed-varied accumulation loop, dedup by
question text against BOTH the existing set and the new batch, and append.

  existing dataset_all.csv ──► seen_questions
        +
  run_pipeline(seed varies) ──► new samples ──► dedup ──► append

This is handoff Task 2 approach (b): load existing, generate fresh, dedup by
question, append. The graph + retrieval index don't change (static corpus).
Only escalate to pattern-aware generation (a)/(c) if new samples cluster on
the same few chunks — check `coverage_fraction` + `path_entities` before vs
after to decide (exposed in the report).

The function is library code (not a job script) so it's reusable + testable;
`vast_job_append.py` wraps it for the box run.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable

from .schema import QASample


def _norm_question(q: str) -> str:
    """Normalize a question for dedup: lowercase + stripped.

    Paraphrase-level dedup (semantic) is intentionally NOT done — two questions
    with the same meaning but different wording over different gold paths are
    legitimate training signal. We dedup only near-identical text to avoid
    trivial duplicates from the seed-varied sampler landing on the same path.
    """
    return (q or "").strip().lower()


def load_existing_questions(dataset_all_csv: str) -> tuple[set, list[dict]]:
    """Load the question set + accepted rows from an existing dataset_all.csv.

    Returns (normalized_question_set, accepted_rows). Only `accepted` /
    `revised_accepted` rows count as "seen" — rejected rows were never part of
    the dataset, so a regenerated version of a previously-rejected question is
    fine to keep. Tolerates a missing file (returns empty set + rows).
    """
    seen: set = set()
    rows: list[dict] = []
    if not dataset_all_csv or not os.path.exists(dataset_all_csv):
        return seen, rows
    with open(dataset_all_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            if r.get("status") in ("accepted", "revised_accepted"):
                q = _norm_question(r.get("question", ""))
                if q:
                    seen.add(q)
    return seen, rows


def _accepted_from_rows(rows: list[dict]) -> list[QASample]:
    """Reconstruct QASample objects from dataset_all.csv accepted rows.

    Only the fields needed for downstream dataset_final / grpo emission are
    repopulated (question, answer, gold_chunk_ids, hop_count, status,
    difficulty_cell, eval scores). Full provenance (llm_calls, path) is not
    carried — the appended dataset_final.csv is rebuilt from the combined
    accepted set, and per-call cost accounting restarts from this run.
    """
    out: list[QASample] = []
    for r in rows:
        if r.get("status") not in ("accepted", "revised_accepted"):
            continue
        try:
            gold = json.loads(r.get("gold_chunk_ids", "[]"))
        except Exception:
            gold = []
        hop = int(r.get("hop_count", 0) or 0)
        s = QASample(
            sample_id=r.get("sample_id", ""),
            question=r.get("question", ""),
            answer=r.get("answer", ""),
            gold_chunk_ids=gold,
            hop_count=hop,
            question_type=r.get("question_type", ""),
            specificity=r.get("specificity", ""),
            retrieval_difficulty=_float_or_none(r.get("retrieval_difficulty")),
            difficulty_cell=r.get("difficulty_cell", ""),
            answerability_f1=_float_or_none(r.get("answerability_f1")),
            faithfulness_score=_float_or_none(r.get("faithfulness_score")),
            status=r.get("status", "accepted"),
        )
        out.append(s)
    return out


def _float_or_none(v) -> float | None:
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def accumulate_and_append(
    *,
    run_pipeline_fn: Callable,
    docs,
    llm,
    embedder,
    out_dir: str,
    existing_csv: str | None = None,
    target_total: int = 25,
    max_iters: int = 20,
    per_hop: int = 40,
    target_per_cell: int = 15,
    backend: str = "sqlite",
    base_seed: int = 42,
    val_fraction: float = 0.2,
    split_level: str = "chunk",
    use_cache: bool = True,
) -> dict:
    """Grow a dataset to `target_total` unique accepted samples, appending to
    any existing `dataset_all.csv`.

    Loads questions already in `existing_csv` (default `<out_dir>/dataset_all.csv`)
    as the "seen" set, then loops `run_pipeline_fn` across seeds, accumulating
    NEW unique accepted samples (dedup by normalized question against existing
    + new). Stage 0 is cached after iteration 1, so iterations 2+ only re-run
    Stages 1-5.

    `run_pipeline_fn` must accept the same kwargs as `build_dataset.run_pipeline`
    (docs, llm, embedder, out_dir, backend, per_hop, target_per_cell, seed,
    val_fraction, split_level, use_cache) and return the balanced accepted
    samples. Tests inject a fake; production passes the real `run_pipeline`.

    `use_cache=False` (the `--fresh` flag) forces a Stage 0 rebuild on iter 1
    and skips cache-reuse for later iterations — the staleness escape hatch
    after a model/chunk-size change. Only Stage 0 is cached (by design; see
    `build_dataset.run_pipeline`'s caching-policy docstring).

    Writes the combined datasets to `out_dir`:
      dataset_all.csv      — existing accepted + rejected + this run's samples
      dataset_final.csv    — combined unique accepted
      dataset_grpo.jsonl   — combined GRPO-ready
      dataset_train/val.csv + _grpo.jsonl — split over the combined set
      append_report.json   — what was added / dedup stats / coverage delta

    Returns the append report dict.
    """
    import agenttune.rag.synthesis.build_dataset as BD

    from .io_utils import dump_table
    from .split import write_split_artifacts

    os.makedirs(out_dir, exist_ok=True)
    if existing_csv is None:
        existing_csv = os.path.join(out_dir, "dataset_all.csv")
    seen_questions, existing_rows = load_existing_questions(existing_csv)
    existing_accepted = _accepted_from_rows(existing_rows)
    n_existing = len(existing_accepted)

    # Coverage BEFORE (from existing accepted) — to measure whether new samples
    # add fresh chunks or just re-tread the same few (handoff Task 2 escalation
    # signal).
    def _coverage(samples, total_chunks=None):
        used = set()
        for s in samples:
            used.update(s.gold_chunk_ids)
        return used

    chunks_before = _coverage(existing_accepted)

    all_new_accepted: list[QASample] = []
    all_run_samples: list[QASample] = []  # every sample this run produced (audit)
    iter0_cache: str | None = None
    it = -1
    for it in range(max_iters):
        have = n_existing + len(all_new_accepted)
        if have >= target_total:
            break
        seed = base_seed + it
        it_out = os.path.join(out_dir, f"_iter{it}")
        os.makedirs(it_out, exist_ok=True)
        # Reuse the Stage 0 cache built by iteration 1 (same corpus/graph) so
        # iterations 2+ skip the expensive entity-extraction LLM calls — but
        # only when use_cache is set (--fresh disables this).
        it_cache = os.path.join(it_out, "stage0_cache.pkl")
        if (
            use_cache
            and not os.path.exists(it_cache)
            and iter0_cache
            and os.path.exists(iter0_cache)
        ):
            import shutil

            shutil.copy2(iter0_cache, it_cache)
        samples = run_pipeline_fn(
            docs=docs,
            llm=llm,
            embedder=embedder,
            out_dir=it_out,
            backend=backend,
            per_hop=per_hop,
            target_per_cell=target_per_cell,
            seed=seed,
            val_fraction=val_fraction,
            split_level=split_level,
            use_cache=use_cache,
        )
        if iter0_cache is None:
            built = os.path.join(it_out, "stage0_cache.pkl")
            if os.path.exists(built):
                iter0_cache = built
        accepted = [s for s in samples if s.status in ("accepted", "revised_accepted")]
        new_count = 0
        for s in accepted:
            q = _norm_question(s.question)
            if q and q not in seen_questions:
                seen_questions.add(q)
                all_new_accepted.append(s)
                new_count += 1
        all_run_samples.extend(samples)
    n_iters = it + 1 if it >= 0 else 0

    # Combined accepted set: existing + new unique. Dedup is already enforced
    # via seen_questions; existing_accepted are all unique by construction.
    combined_accepted = existing_accepted + all_new_accepted
    # Trim to target if over (keep existing + newest new first)
    if len(combined_accepted) > target_total:
        keep_new = max(0, target_total - n_existing)
        combined_accepted = existing_accepted + all_new_accepted[:keep_new]

    # Emit the combined datasets (same schema as build_dataset's dumps).
    combined_all_rows = existing_rows  # existing audit trail preserved
    # append this run's rows (all samples incl. rejected) for the full audit
    new_rows = BD._rows_from_samples(all_run_samples, include_rejected=True)
    combined_all_rows = existing_rows + new_rows
    dump_table(combined_all_rows, os.path.join(out_dir, "dataset_all.csv"))
    dump_table(
        BD._rows_from_samples(combined_accepted, include_rejected=False),
        os.path.join(out_dir, "dataset_final.csv"),
    )
    grpo_rows = BD._grpo_rows(combined_accepted)
    with open(os.path.join(out_dir, "dataset_grpo.jsonl"), "w") as f:
        for r in grpo_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Split over the COMBINED set so train/val are disjoint across old + new.
    split_report = write_split_artifacts(
        combined_accepted,
        out_dir,
        val_fraction=val_fraction,
        level=split_level,
        seed=base_seed,
        backend=backend,
    )

    chunks_after = _coverage(combined_accepted)
    new_chunks = chunks_after - chunks_before

    report = {
        "existing_csv": existing_csv,
        "n_existing_accepted": n_existing,
        "n_new_unique": len(all_new_accepted),
        "n_total_after": len(combined_accepted),
        "target_total": target_total,
        "n_iterations": n_iters,
        "n_run_samples_total": len(all_run_samples),
        "n_run_rejected": sum(1 for s in all_run_samples if s.status == "rejected"),
        "n_new_chunks_covered": len(new_chunks),
        "coverage_chunks_before": len(chunks_before),
        "coverage_chunks_after": len(chunks_after),
        "split": split_report,
        "note": (
            "static-corpus append (handoff Task 2b): dedup by normalized "
            "question against existing + new; graph/index unchanged"
        ),
    }
    with open(os.path.join(out_dir, "append_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report
