"""
E1 graph generator — every training-run metric from FINNLP_EXPERIMENTS v2 §5
that is derivable from a run directory, as PNGs in <run_dir>/graphs/.

Inputs (all written by train_grpo.py during the run):
  training_log.json   per-step trainer logs (loss/grad_norm/kl/entropy/reward
                      + reward/<component> channels from RewardSignalLogger)
  trace.jsonl         per-trajectory records (queries, retrieved+gold chunk
                      ids, per-component rewards, answer tag) — in training
                      order; `trajectories_per_step` (default 64 = 8 prompts x
                      8 generations) converts trajectory index -> approx step.

Figures (appendix artifacts, §5):
  1. reward_curve.png            mean total reward per step
  2. reward_components.png       per-component reward means per step
                                 (format/termination/correctness_numeric/
                                 golden_chunk_recall/conciseness/frugality)
  3. stability.png               loss / grad_norm / kl / entropy (4 panels) —
                                 the IS-fix proof GRPO reviewers look for first
  4. retrieval_recall_over_training.png   reference- + chunk-level gold recall
  5. behavior_over_training.png  answer-tag rate, unique-query ratio
                                 (query-echo -> reformulation), mean searches
  6. search_count_ecdf.png       search-count distribution, first vs last
                                 quartile of training (the behavior shift)
  7. correctness_over_training.png       numeric correctness component trend

Usage:
    python -m agenttune.rag.scripts.make_e1_graphs \
        --run_dir rag_experiments/runs/e1_finder_seed42_30steps
"""

import argparse
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _norm_query(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (q or "").lower()).strip()


def _doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit("::", 1)[0] if "::" in chunk_id else chunk_id


def _rolling(xs: list[float | None], w: int) -> list[float | None]:
    out = []
    for i in range(len(xs)):
        seg = [x for x in xs[max(0, i - w + 1) : i + 1] if x is not None]
        out.append(statistics.mean(seg) if seg else None)
    return out


def _xy(xs: list[float | None]):
    return ([i for i, x in enumerate(xs) if x is not None], [x for x in xs if x is not None])


def load_run(run_dir: str):
    log_path = os.path.join(run_dir, "training_log.json")
    trace_path = os.path.join(run_dir, "trace.jsonl")
    history = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            history = [e for e in json.load(f) if isinstance(e, dict)]
    traces = []
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            traces = [json.loads(l) for l in f if l.strip()]
    return history, traces


def step_series(history: list[dict], key: str) -> list[float | None]:
    """Per-step series from training_log.json (steps may repeat/skip; keep the
    last value per step and return dense step-ordered list)."""
    by_step: dict[int, float] = {}
    for e in history:
        if "step" in e and key in e and isinstance(e.get(key), int | float):
            by_step[int(e["step"])] = e[key]
    if not by_step:
        return []
    return [by_step.get(s) for s in range(1, max(by_step) + 1)]


def trace_series(traces: list[dict], key: str) -> list[float | None]:
    """Per-trajectory behavior series from trace.jsonl."""
    out: list[float | None] = []
    for t in traces:
        comp = t.get("reward_components", {})
        if key == "unique_query_ratio":
            queries = [
                _norm_query(
                    tc["query"].get("query")
                    if isinstance(tc.get("query"), dict)
                    else str(tc.get("query", ""))
                )
                for tc in t.get("tool_calls", [])
            ]
            out.append(len(set(queries)) / len(queries) if queries else None)
        elif key == "ref_recall":
            gold = set(t.get("gold_chunk_ids", []))
            retr = set(t.get("retrieved_chunk_ids", []))
            gdocs = {_doc_of(c) for c in gold}
            out.append(len(gdocs & {_doc_of(c) for c in retr}) / len(gdocs) if gdocs else None)
        elif key == "chunk_recall":
            gold = set(t.get("gold_chunk_ids", []))
            retr = set(t.get("retrieved_chunk_ids", []))
            out.append(len(gold & retr) / len(gold) if gold else None)
        elif key == "has_answer_tag":
            out.append(1.0 if t.get("has_answer_tag") else 0.0)
        elif key == "n_tool_calls":
            out.append(float(t.get("n_tool_calls", 0)))
        elif key == "reward":
            out.append(float(t.get("reward", 0.0)))
        else:  # reward component channel
            v = comp.get(key)
            out.append(float(v) if v is not None else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--trajectories_per_step",
        type=int,
        default=64,
        help="trace lines per optimizer step (prompts/step x "
        "num_generations; 8x8=64 for the E1 config)",
    )
    parser.add_argument("--smooth", type=int, default=5, help="rolling window (steps)")
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dir, "graphs")
    os.makedirs(out_dir, exist_ok=True)
    history, traces = load_run(args.run_dir)
    tps = max(args.trajectories_per_step, 1)

    def save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[make_e1_graphs] wrote {path}")

    # ── 1-2. Reward curves (per step) ────────────────────────────────────────
    reward = step_series(history, "reward") or step_series(history, "rewards")
    if reward:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs, ys = _xy(_rolling(reward, args.smooth))
        ax.plot(xs, ys, lw=1.5)
        xs_raw, ys_raw = _xy(reward)
        ax.scatter(xs_raw, ys_raw, s=4, alpha=0.25)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("mean reward")
        ax.set_title("Total reward per step")
        ax.grid(alpha=0.3)
        save(fig, "reward_curve.png")

    comp_keys = sorted({k for e in history for k in e if k.startswith("reward/")})
    if comp_keys:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in comp_keys:
            s = _rolling(step_series(history, k), args.smooth)
            xs, ys = _xy(s)
            if xs:
                ax.plot(xs, ys, lw=1.2, label=k.replace("reward/", ""))
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("component mean")
        ax.set_title("Per-component rewards per step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "reward_components.png")

    # ── 3. Stability (the IS-fix proof panel) ────────────────────────────────
    panels = [("loss", "loss"), ("grad_norm", "grad_norm"), ("kl", "KL"), ("entropy", "entropy")]
    if any(step_series(history, k) for k, _ in panels):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (key, label) in zip(axes.flat, panels, strict=False):
            s = step_series(history, key)
            xs, ys = _xy(s)
            if xs:
                ax.plot(xs, ys, lw=1.2)
            ax.set_title(label)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        fig.suptitle("Training stability (grad_norm > 0 throughout = IS fix holds)")
        fig.tight_layout()
        save(fig, "stability.png")

    # ── Per-trajectory series (x axis converted to approx steps) ─────────────
    def traj_xs(n):
        return [i / tps for i in range(n)]

    # ── 4. Retrieval recall over training ────────────────────────────────────
    ref_r = trace_series(traces, "ref_recall")
    chunk_r = trace_series(traces, "chunk_recall")
    if ref_r:
        fig, ax = plt.subplots(figsize=(7, 4))
        w = max(tps // 2, 8)
        for series, label, color in (
            (ref_r, "reference-level recall", "tab:blue"),
            (chunk_r, "chunk-level recall", "tab:orange"),
        ):
            sm = _rolling(series, w)
            xs, ys = _xy(sm)
            ax.plot([x / tps for x in xs], ys, lw=1.5, label=label, color=color)
        ax.set_xlabel("optimizer step (approx)")
        ax.set_ylabel("gold evidence recall")
        ax.set_title("Retrieval recall over training")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, "retrieval_recall_over_training.png")

    # ── 5. Behavior over training: answer-tag rate, echo ratio, searches ─────
    tags = trace_series(traces, "has_answer_tag")
    echo = trace_series(traces, "unique_query_ratio")
    searches = trace_series(traces, "n_tool_calls")
    if tags:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        w = max(tps // 2, 8)
        for ax, series, label in (
            (axes[0], tags, "answer-tag rate"),
            (axes[1], echo, "unique-query ratio"),
            (axes[2], searches, "searches / episode"),
        ):
            sm = _rolling(series, w)
            xs, ys = _xy(sm)
            ax.plot([x / tps for x in xs], ys, lw=1.5)
            ax.set_title(label)
            ax.set_xlabel("optimizer step (approx)")
            ax.grid(alpha=0.3)
        fig.suptitle("Behavior over training (echo → reformulation shift)")
        fig.tight_layout()
        save(fig, "behavior_over_training.png")

    # ── 6. Search-count ECDF, first vs last quartile ─────────────────────────
    if searches and len(searches) >= 8:
        first = sorted(x for x in searches[: len(searches) // 4] if x is not None)
        last = sorted(x for x in searches[3 * len(searches) // 4 :] if x is not None)
        fig, ax = plt.subplots(figsize=(6, 4))
        for vals, label in ((first, "first quartile of training"), (last, "last quartile")):
            if vals:
                ax.step(vals, [(i + 1) / len(vals) for i in range(len(vals))], label=label)
        ax.set_xlabel("searches per episode")
        ax.set_ylabel("ECDF")
        ax.set_title("Search-count distribution shift")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, "search_count_ecdf.png")

    # ── 7. Correctness over training ─────────────────────────────────────────
    corr = trace_series(traces, "correctness_numeric")
    if corr:
        fig, ax = plt.subplots(figsize=(7, 4))
        w = max(tps // 2, 8)
        sm = _rolling(corr, w)
        xs, ys = _xy(sm)
        ax.plot([x / tps for x in xs], ys, lw=1.5, color="tab:green")
        ax.set_xlabel("optimizer step (approx)")
        ax.set_ylabel("numeric correctness")
        ax.set_title("Answer correctness over training")
        ax.grid(alpha=0.3)
        save(fig, "correctness_over_training.png")

    print(f"[make_e1_graphs] done — {len(os.listdir(out_dir))} figures in {out_dir}")


if __name__ == "__main__":
    main()
