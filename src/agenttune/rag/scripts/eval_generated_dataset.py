"""Closed-context evaluation of ANY OpenAI-compatible model on a generated
multi-hop legal QA dataset.

The dataset is self-contained (`dataset_eval.jsonl` — question + gold passages
+ gold_answer per row, produced by `build_cuad_dataset.py`). No retrieval
harness needed: each row's passages are given to the model, it answers, and we
score with the SAME relaxed F1 the pipeline gates answerability on — so the
reported numbers are directly comparable to the pipeline's own
`answerability_f1_relaxed` (the verifying solver = the "baseline" model).

Usage:
    python -m agenttune.rag.scripts.eval_generated_dataset \\
        --dataset /path/to/cuad/dataset_eval.jsonl \\
        --llm_base_url https://api.deepseek.com \\
        --llm_model deepseek-v4-flash \\
        --llm_api_key $DEEPSEEK_API_KEY \\
        --limit 200 --max_workers 64 --out /path/to/cuad_eval_report.json

Reports per-model: overall relaxed-F1 pass rate (@0.5) + mean F1, plus the
same broken down by hop count and difficulty cell.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def _scorers():
    try:
        from ..rewards.qa_metrics import f1_score, relaxed_f1_score

        return relaxed_f1_score, f1_score
    except Exception:
        from ..datagen import token_f1

        return token_f1, token_f1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--dataset",
        required=True,
        help="dataset_eval.jsonl (or dataset_grpo.jsonl — both " "carry gold_passages)",
    )
    ap.add_argument("--llm_base_url", required=True)
    ap.add_argument("--llm_model", required=True)
    ap.add_argument("--llm_api_key", required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="max rows to evaluate (0 = all; sample deterministically)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_workers", type=int, default=64)
    ap.add_argument("--out", default=None, help="path for the JSON report")
    ap.add_argument(
        "--pass_rate_threshold",
        type=float,
        default=0.5,
        help="relaxed-F1 threshold defining a 'correct' answer "
        "(pipeline's answerability gate uses 0.5)",
    )
    args = ap.parse_args()

    relaxed_f1, strict_f1 = _scorers()

    from ..synthesis.io_utils import extract_json
    from ..synthesis.llm_client import OpenAICompatLLMClient

    rows = []
    for line in open(args.dataset):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        sys.exit(f"no rows in {args.dataset}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    # closed-context prompt: same "answer only from these passages" contract as
    # the verifying solver, so scores are comparable to answerability_f1_relaxed
    def prompt_for(row):
        passages = row.get("gold_passages") or []
        ctx = "\n\n".join(f"[Passage {i+1}] {p['text']}" for i, p in enumerate(passages))
        return [
            {
                "role": "user",
                "content": (
                    "Answer the question using ONLY the provided passages. Do not use "
                    "outside knowledge. If the passages do not contain the answer, "
                    'respond with exactly "UNANSWERABLE".\n\n'
                    "Passages:\n"
                    + ctx
                    + "\n\nQuestion: "
                    + row["question"]
                    + '\n\nRespond ONLY with JSON: {"answer": "<short answer or '
                    'UNANSWERABLE>"}'
                ),
            }
        ]

    llm = OpenAICompatLLMClient(
        model=args.llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        timeout=120,
        max_retries=2,
    )
    msgs = [prompt_for(r) for r in rows]
    from ..synthesis.llm_client import parallel_chat

    results = parallel_chat(
        llm,
        msgs,
        max_workers=args.max_workers,
        stage="eval",
        purpose="eval_generated_dataset",
        temperature=0.0,
        max_tokens=8192,
    )

    per_row, by_hop, by_diff, total_cost = [], defaultdict(list), defaultdict(list), 0.0
    for r, (text, rec) in zip(rows, results, strict=False):
        try:
            ans = extract_json(text).get("answer", "").strip()
        except Exception:
            ans = "UNANSWERABLE"
        sc = 0.0 if ans == "UNANSWERABLE" else relaxed_f1(ans, r["gold_answer"])
        sc_strict = 0.0 if ans == "UNANSWERABLE" else strict_f1(ans, r["gold_answer"])
        total_cost += rec.cost_usd
        row = {
            "question_id": r["question_id"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "model_answer": ans,
            "relaxed_f1": round(sc, 4),
            "strict_f1": round(sc_strict, 4),
            "pass": sc >= args.pass_rate_threshold,
            "hop_count": r.get("hop_count"),
            "difficulty_cell": r.get("difficulty_cell"),
            "pipeline_answerability_f1_relaxed": r.get("answerability_f1_relaxed"),
        }
        per_row.append(row)
        by_hop[r.get("hop_count")].append(sc)
        by_diff[r.get("difficulty_cell")].append(sc)

    n = len(per_row)
    mean_f1 = sum(r["relaxed_f1"] for r in per_row) / max(1, n)
    pass_rate = sum(r["pass"] for r in per_row) / max(1, n)
    report = {
        "model": args.llm_model,
        "dataset": args.dataset,
        "n_evaluated": n,
        "mean_relaxed_f1": round(mean_f1, 4),
        f"pass_rate_at_{args.pass_rate_threshold}": round(pass_rate, 4),
        "total_cost_usd": round(total_cost, 4),
        "by_hop_count": {
            str(k): {"n": len(v), "mean_relaxed_f1": round(sum(v) / len(v), 4)}
            for k, v in sorted(by_hop.items())
        },
        "by_difficulty_cell": {
            str(k): {"n": len(v), "mean_relaxed_f1": round(sum(v) / len(v), 4)}
            for k, v in sorted(by_diff.items())
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(
            json.dumps({"report": report, "per_row": per_row}, indent=2, ensure_ascii=False)
        )
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
