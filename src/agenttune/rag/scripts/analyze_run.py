"""
Post-run analysis for E1 training runs (FinNLP paper artifacts).

Reads a run directory produced by train_grpo.py (trace.jsonl +
training_log.json + run_manifest.json) and writes analysis/metrics.json +
per-graph CSVs under <run_dir>/analysis/. These are the training-dynamics
inputs to the paper's appendix figures (FINNLP_EXPERIMENTS.md v2 §5):

  - reward curve per component (reward/<component> per step)   -> rewards_per_step.csv
  - stability curves (loss/grad_norm/kl/entropy per step)      -> stability_per_step.csv
  - unique-query ratio per trajectory (query-echo detector)    -> behavior_per_trajectory.csv
  - search-count distribution (ECDF input)                     -> behavior_per_trajectory.csv
  - retrieval recall per trajectory (reference- + chunk-level) -> behavior_per_trajectory.csv
  - headline summary numbers                                   -> metrics.json

Usage:
    python -m agenttune.rag.scripts.analyze_run --run_dir rag_experiments/runs/e1_finder_seed42
"""

import argparse
import json
import os
import re
import statistics


def _norm_query(q: str) -> str:
    """Normalise a search query for the unique-query ratio (loop detector):
    lowercase, collapse whitespace/punctuation."""
    return re.sub(r"[^a-z0-9]+", " ", (q or "").lower()).strip()


def _doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit("::", 1)[0] if "::" in chunk_id else chunk_id


def load_trace(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def behavior_row(trace: dict) -> dict:
    queries = [
        _norm_query(
            tc.get("query", {}).get("query")
            if isinstance(tc.get("query"), dict)
            else str(tc.get("query"))
        )
        for tc in trace.get("tool_calls", [])
    ]
    n_queries = len(queries)
    n_unique = len(set(queries))
    retrieved = set(trace.get("retrieved_chunk_ids", []))
    gold = set(trace.get("gold_chunk_ids", []))
    gold_docs = {_doc_of(c) for c in gold}
    retrieved_docs = {_doc_of(c) for c in retrieved}
    comp = trace.get("reward_components", {})
    return {
        "question": trace.get("question", "")[:80],
        "n_tool_calls": trace.get("n_tool_calls", 0),
        "unique_query_ratio": (n_unique / n_queries) if n_queries else None,
        "has_answer_tag": trace.get("has_answer_tag", False),
        "reward": trace.get("reward", 0.0),
        "correctness": comp.get("correctness_numeric"),
        "ref_recall": (len(gold_docs & retrieved_docs) / len(gold_docs)) if gold_docs else None,
        "chunk_recall": (len(gold & retrieved) / len(gold)) if gold else None,
    }


def write_csv(path: str, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in columns) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ── Per-step training curves (training_log.json) ─────────────────────────
    log_path = os.path.join(args.run_dir, "training_log.json")
    stability_rows: list[dict] = []
    reward_rows: list[dict] = []
    reward_components: list[str] = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            history = json.load(f)
        for entry in history:
            if "loss" not in entry:
                continue
            step = entry.get("step")
            stability_rows.append(
                {
                    "step": step,
                    "loss": entry.get("loss"),
                    "grad_norm": entry.get("grad_norm"),
                    "kl": entry.get("kl"),
                    "entropy": entry.get("entropy"),
                    "reward": entry.get("reward", entry.get("rewards")),
                    "reward_std": entry.get("reward_std", entry.get("rewards_std")),
                    "learning_rate": entry.get("learning_rate"),
                }
            )
            row = {"step": step}
            for k, v in entry.items():
                if k.startswith("reward/"):
                    row[k] = v
                    if k not in reward_components:
                        reward_components.append(k)
            reward_rows.append(row)
        write_csv(
            os.path.join(out_dir, "stability_per_step.csv"),
            stability_rows,
            ["step", "loss", "grad_norm", "kl", "entropy", "reward", "reward_std", "learning_rate"],
        )
        write_csv(
            os.path.join(out_dir, "rewards_per_step.csv"),
            reward_rows,
            ["step"] + reward_components,
        )

    # ── Per-trajectory behavior (trace.jsonl) ────────────────────────────────
    trace_path = os.path.join(args.run_dir, "trace.jsonl")
    behavior_rows: list[dict] = []
    if os.path.exists(trace_path):
        behavior_rows = [behavior_row(t) for t in load_trace(trace_path)]
        write_csv(
            os.path.join(out_dir, "behavior_per_trajectory.csv"),
            behavior_rows,
            [
                "question",
                "n_tool_calls",
                "unique_query_ratio",
                "has_answer_tag",
                "reward",
                "correctness",
                "ref_recall",
                "chunk_recall",
            ],
        )

    # ── Headline summary ─────────────────────────────────────────────────────
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None

    # First/last quartile comparison: the training-dynamics story (does
    # retrieval grounding climb? does the echo ratio drop?) — windowed so the
    # summary is robust to run length.
    def _window_mean(key, frac_lo, frac_hi):
        n = len(behavior_rows)
        if n < 4:
            return None
        seg = behavior_rows[int(n * frac_lo) : int(n * frac_hi)] or behavior_rows
        return _mean([r[key] for r in seg])

    grad_norms = [r["grad_norm"] for r in stability_rows if r.get("grad_norm")]
    metrics = {
        "run_dir": args.run_dir,
        "n_trajectories": len(behavior_rows),
        "n_steps_logged": len(stability_rows),
        "reward_components_logged": reward_components,
        # IS-fix health: grad_norm must be non-zero after step 1 (the Sprint-2
        # blocker was grad_norm=0 across all steps).
        "grad_norm_nonzero_frac": (
            sum(1 for g in grad_norms[1:] if g and g > 0) / max(len(grad_norms) - 1, 1)
            if len(grad_norms) > 1
            else None
        ),
        "answer_tag_rate_first_quartile": _window_mean("has_answer_tag", 0.0, 0.25),
        "answer_tag_rate_last_quartile": _window_mean("has_answer_tag", 0.75, 1.0),
        "correctness_first_quartile": _window_mean("correctness", 0.0, 0.25),
        "correctness_last_quartile": _window_mean("correctness", 0.75, 1.0),
        "ref_recall_first_quartile": _window_mean("ref_recall", 0.0, 0.25),
        "ref_recall_last_quartile": _window_mean("ref_recall", 0.75, 1.0),
        "unique_query_ratio_first_quartile": _window_mean("unique_query_ratio", 0.0, 0.25),
        "unique_query_ratio_last_quartile": _window_mean("unique_query_ratio", 0.75, 1.0),
        "mean_searches_first_quartile": _window_mean("n_tool_calls", 0.0, 0.25),
        "mean_searches_last_quartile": _window_mean("n_tool_calls", 0.75, 1.0),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
