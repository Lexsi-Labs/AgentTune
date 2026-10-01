"""
REAL lm-eval standardized benchmarking via the documented LMEvalRunner.
=======================================================================

`agenttune.eval` exports `LMEvalConfig` / `LMEvalTask` / `LMEvalRunner` as the
integration with EleutherAI's `lm-eval` harness for standardized benchmarks::

    from agenttune.eval import LMEvalConfig, LMEvalTask, LMEvalRunner
    runner = LMEvalRunner(LMEvalConfig(model_name="..."))
    results = runner.evaluate_tasks([LMEvalTask(..., lm_eval_task_name="arc_challenge")])

`LMEvalRunner` shells out to the real `lm_eval` CLI (`--model hf ...`), then parses
the timestamped results JSON and extracts the metrics. Nothing in the repo ever
ran it — there is no test and no example that invokes `lm_eval` and reads a real
score back. This script does exactly that, end to end, with a real model.

It runs two real standardized benchmarks — **ARC-Challenge** and **HellaSwag** —
on a live `Qwen/Qwen3-0.6B`, from the local HF cache (no network), through the
documented `LMEvalRunner.evaluate_tasks` path, and reads back the real accuracies
plus the combined summary the runner writes.

HONEST SCOPE — read before quoting
----------------------------------
This proves the *integration runs for real*: the real `lm_eval` subprocess
executes, produces a results file, and `LMEvalRunner` parses genuine metrics out
of it. To keep it fast it caps each task at a small `--limit`, so the accuracies
are **subset estimates on a 0.6B model, not leaderboard numbers** — do not quote
them as ARC/HellaSwag scores. The claim is that the documented runner works, not
that this model is good.

Needs `Qwen/Qwen3-0.6B` and the `allenai/ai2_arc` (ARC-Challenge) + `Rowan/hellaswag`
(HellaSwag) datasets in the local HF cache. Run:
    python examples/lm_eval_real.py

NOTE: this originally used WinoGrande, but that dataset's HF repo still ships a
legacy loading script, which is incompatible with the `datasets`/`huggingface_hub`
versions pinned here (`HfUriError` while resolving the repo) — fails regardless of
cache state or network. Swapped for HellaSwag, which ships in the modern
Parquet-based format.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # subprocess inherits → uses cache only
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import glob
import json

from agenttune.eval import LMEvalConfig, LMEvalRunner, LMEvalTask

MODEL = "Qwen/Qwen3-0.6B"
OUT_DIR = "/tmp/agenttune-lm-eval"
LIMIT = 30  # samples per task — keeps it fast; makes these SUBSET estimates, not full scores

TASKS = [
    LMEvalTask(
        name="arc_challenge",
        category="commonsense_reasoning",
        description="AI2 ARC-Challenge multiple-choice science questions.",
        lm_eval_task_name="arc_challenge",
        metrics=["acc", "acc_norm"],
    ),
    LMEvalTask(
        name="hellaswag",
        category="commonsense_reasoning",
        description="HellaSwag commonsense sentence-completion task.",
        lm_eval_task_name="hellaswag",
        metrics=["acc", "acc_norm"],
    ),
]


def main():
    config = LMEvalConfig(
        model_name=MODEL,
        batch_size=16,
        limit=LIMIT,
        output_dir=OUT_DIR,
        save_results=True,
    )
    runner = LMEvalRunner(config)

    print(f"\n── Running real lm_eval ({MODEL}, limit={LIMIT}/task, from HF cache) ──")
    print("   (each task shells out to the real `lm_eval --model hf` CLI)")
    results = runner.evaluate_tasks(TASKS)

    print("\n── Parsed metrics (read back from the real lm_eval results JSON) ──")
    ok_tasks = 0
    for r in results:
        err = "error" in r.metrics
        acc = r.metrics.get("acc")
        print(f"  {r.task_name:<14} metrics={ {k: round(v, 4) for k, v in r.metrics.items()} }")
        if not err and isinstance(acc, int | float) and 0.0 <= acc <= 1.0:
            ok_tasks += 1

    # The runner writes a combined summary across tasks — confirm it is real.
    summary_file = os.path.join(OUT_DIR, "lm_eval_summary.json")
    summary = {}
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)

    # Confirm the real lm_eval results files exist on disk (proof the CLI ran).
    result_jsons = [
        p
        for p in glob.glob(os.path.join(OUT_DIR, "*.json"))
        if "summary" not in os.path.basename(p) and "samples" not in os.path.basename(p)
    ]

    print("\n── Summary ──")
    print(f"  tasks with a real accuracy in [0,1]   : {ok_tasks}/{len(TASKS)}")
    print(f"  lm_eval results files on disk         : {len(result_jsons)}")
    print(f"  combined summary written by runner    : {bool(summary)}")
    if summary.get("summary"):
        print(
            f"  summary stats (subset, NOT leaderboard): "
            f"{ {k: round(v, 4) for k, v in summary['summary'].items()} }"
        )
    print("  scope: integration runs for real; small-subset estimates, not full-benchmark scores")

    assert ok_tasks == len(
        TASKS
    ), "not every task returned a real accuracy in [0,1] (lm_eval/parse failed)"
    assert len(result_jsons) >= len(
        TASKS
    ), "lm_eval results JSON missing — the CLI did not produce output"
    assert summary and summary.get("summary"), "combined summary not written by the runner"
    print("\n✓ LMEvalRunner ran the real lm_eval harness end-to-end and parsed real scores.")


if __name__ == "__main__":
    main()
