"""
T1/T4 probing script — run solve-difficulty + requires-search labeling.

Probes the base model on HotpotQA questions WITHOUT retrieval to:
  - T1: compute solve_difficulty (1 - pass_rate) for curriculum staging
  - T4: label requires_search (True if model can't answer from parametric knowledge)
  - Contamination gate: drop already-solved questions

Output: per-question labels saved to JSON for use in training data prep.

Usage:
    python -m agenttune.rag.scripts.run_t1_t4_probe \\
        --model Qwen/Qwen3.5-4B \\
        --num_questions 100 --k 4 \\
        --output rag_experiments/t1_t4_labels.json
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--num_questions", type=int, default=100, help="Number of HotpotQA questions to probe"
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Samples per question (R1-Searcher uses 5; we default 4)"
    )
    parser.add_argument(
        "--pass_threshold",
        type=float,
        default=0.5,
        help="pass_rate >= this → requires_search=False",
    )
    parser.add_argument(
        "--no_vllm", action="store_true", help="Use HF generate instead of vLLM (slower)"
    )
    parser.add_argument("--output", default="rag_experiments/t1_t4_labels.json")
    args = parser.parse_args()

    from agenttune.rag.data.hotpotqa import load_hotpotqa_splits
    from agenttune.rag.synthesis import label_requires_search

    _, eval_split = load_hotpotqa_splits(
        config="distractor", train_size=1, eval_size=args.num_questions
    )
    questions = [{"question": r["question"], "answer": r["answer"]} for r in eval_split]
    print(f"[t1_t4_probe] probing {len(questions)} questions with k={args.k}")

    labeled = label_requires_search(
        questions,
        args.model,
        k=args.k,
        pass_threshold=args.pass_threshold,
        use_vllm=not args.no_vllm,
    )

    # Contamination gate stats
    n_solved = sum(1 for l in labeled if l["already_solved"])
    n_needs = sum(1 for l in labeled if l["requires_search"])
    print("\n[t1_t4_probe] Results:")
    print(f"  Already solved (pass_rate=1.0): {n_solved}/{len(labeled)}")
    print(f"  Requires search:               {n_needs}/{len(labeled)}")
    print(
        f"  Avg solve_difficulty:          {sum(l['solve_difficulty'] for l in labeled)/len(labeled):.3f}"
    )
    print(
        f"  Avg pass_rate:                 {sum(l['pass_rate'] for l in labeled)/len(labeled):.3f}"
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "model": args.model,
                "k": args.k,
                "pass_threshold": args.pass_threshold,
                "num_questions": len(labeled),
                "n_already_solved": n_solved,
                "n_requires_search": n_needs,
                "labels": labeled,
            },
            f,
            indent=2,
        )
    print(f"[t1_t4_probe] wrote {args.output}")


if __name__ == "__main__":
    main()
