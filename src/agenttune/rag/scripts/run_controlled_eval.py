"""
Controlled plain-vs-M1 eval (Sprint 2 §7 deferred deliverable).

Runs the eval harness twice on the SAME held-out question set:
  1. Untrained base model (no adapter)
  2. Trained M1 adapter (merged into base)

Each run evaluates all three conditions (plain, m1, recent_k) with identical
decode settings, so the only variable is the adapter. Produces a comparison
table + the M1 token-cut measurement (C1: ≥30% bar).

Usage:
    python -m agenttune.rag.scripts.run_controlled_eval \\
        --model Qwen/Qwen3.5-4B --backend sqlite \\
        --index_dir rag_experiments/indexes/sqlite \\
        --adapter_path rag_experiments/runs/m1_t3_real \\
        --num_questions 40 --max_steps 6 \\
        --output_dir rag_experiments/controlled_eval
"""

import argparse
import json
import os
import subprocess
import sys


def run_eval(args, adapter_path: str = None) -> dict:
    """Run eval_m1_zeroshot for one model config, return parsed results."""
    tag = "trained" if adapter_path else "base"
    output_file = os.path.join(args.output_dir, f"eval_{tag}.json")

    cmd = [
        sys.executable,
        "-m",
        "agenttune.rag.scripts.eval_m1_zeroshot",
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--index_dir",
        args.index_dir,
        "--num_questions",
        str(args.num_questions),
        "--max_steps",
        str(args.max_steps),
        "--conditions",
        "plain",
        "m1",
        "recent_k",
        "--output",
        output_file,
    ]
    if adapter_path:
        cmd += ["--adapter_path", adapter_path]

    print(f"\n[controlled_eval] running {tag} model...")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=args.cwd)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"eval failed for {tag}")
    print(result.stdout[-1000:])

    with open(output_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--backend", choices=["sqlite", "chroma"], default="sqlite")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument(
        "--adapter_path", required=True, help="Path to the trained LoRA adapter directory"
    )
    parser.add_argument("--num_questions", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--output_dir", default="rag_experiments/controlled_eval")
    parser.add_argument(
        "--cwd", default=None, help="Working directory for subprocess (default: current)"
    )
    args = parser.parse_args()
    if args.cwd is None:
        args.cwd = os.getcwd()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run both evals
    base_results = run_eval(args, adapter_path=None)
    trained_results = run_eval(args, adapter_path=args.adapter_path)

    # Build comparison table
    print("\n" + "=" * 80)
    print("CONTROLLED COMPARISON: plain vs M1-trained vs recent_k")
    print("(same 40 held-out questions, identical decode settings)")
    print("=" * 80)

    comparison = {"base": {}, "trained": {}}
    for model_tag, results in [("base", base_results), ("trained", trained_results)]:
        res = results.get("results", results)
        for cond in ["plain", "m1", "recent_k"]:
            r = res.get(cond, {})
            if "error" in r:
                print(f"  {model_tag:>7} / {cond:<10}: ERROR: {r['error'][:60]}")
                comparison[model_tag][cond] = {"error": r["error"]}
            else:
                print(
                    f"  {model_tag:>7} / {cond:<10}: "
                    f"EM={r['exact_match']:.3f}  F1={r['f1']:.3f}  "
                    f"calls={r['avg_tool_calls']:.2f}  "
                    f"tokens={r['avg_token_cost']:.0f}  "
                    f"%state={r['pct_emitted_state']:.2f}"
                )
                comparison[model_tag][cond] = {
                    "exact_match": r["exact_match"],
                    "f1": r["f1"],
                    "avg_tool_calls": r["avg_tool_calls"],
                    "avg_token_cost": r["avg_token_cost"],
                    "pct_emitted_state": r["pct_emitted_state"],
                    "pct_zero_search": r["pct_zero_search"],
                    "per_question": r.get("per_question", []),
                }

    # M1 token-cut measurement (C1: ≥30% bar)
    print("\n" + "-" * 80)
    print("M1 TOKEN-CUT MEASUREMENT (C1: ≥30% bar)")
    print("-" * 80)
    base_plain = comparison["base"].get("plain", {})
    trained_m1 = comparison["trained"].get("m1", {})
    comparison["base"].get("m1", {})

    if "avg_token_cost" in base_plain and "avg_token_cost" in trained_m1:
        base_tokens = base_plain["avg_token_cost"]
        m1_tokens = trained_m1["avg_token_cost"]
        if base_tokens > 0:
            token_cut = (base_tokens - m1_tokens) / base_tokens * 100
        else:
            token_cut = 0.0
        print(f"  Base plain avg tokens:     {base_tokens:.0f}")
        print(f"  Trained M1 avg tokens:     {m1_tokens:.0f}")
        print(f"  Token cut:                 {token_cut:.1f}%")
        print(f"  ≥30% bar:                  {'PASS ✓' if token_cut >= 30 else 'FAIL ✗'}")

        base_em = base_plain.get("exact_match", 0)
        trained_m1_em = trained_m1.get("exact_match", 0)
        print(f"  Base plain EM:             {base_em:.3f}")
        print(f"  Trained M1 EM:            {trained_m1_em:.3f}")
        em_delta = trained_m1_em - base_em
        print(
            f"  EM delta:                 {em_delta:+.3f} "
            f"({'no loss ✓' if em_delta >= -0.05 else 'LOSS ✗'})"
        )
        comparison["token_cut"] = {
            "base_plain_tokens": base_tokens,
            "trained_m1_tokens": m1_tokens,
            "token_cut_pct": token_cut,
            "passes_30pct_bar": token_cut >= 30,
            "base_plain_em": base_em,
            "trained_m1_em": trained_m1_em,
            "em_delta": em_delta,
            "no_em_loss": em_delta >= -0.05,
        }

    # Save combined results
    output_file = os.path.join(args.output_dir, "comparison.json")
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[controlled_eval] wrote {output_file}")


if __name__ == "__main__":
    main()
