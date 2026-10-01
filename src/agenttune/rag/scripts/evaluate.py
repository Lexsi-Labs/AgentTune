"""
Evaluation across a (checkpoint x backend) matrix: before/after (base vs
trained checkpoint), model-size comparison (different base models), and
retrieval-backend comparison — all expressed as rows of the same matrix.

Checkpoint spec syntax: either a bare HF model id (baseline, no adapter)
or "base_model_id::adapter_path" (a trained LoRA checkpoint).

Usage:
    python -m agenttune.rag.scripts.evaluate \\
        --checkpoints "Qwen/Qwen2.5-1.5B-Instruct" \\
                      "Qwen/Qwen2.5-1.5B-Instruct::rag_experiments/runs/qwen1.5b_sqlite" \\
        --backends sqlite chroma \\
        --index_dir_template "rag_experiments/indexes/{backend}" \\
        --output_csv rag_experiments/results/eval_matrix.csv \\
        --use_judge
"""

import argparse
import csv
import json
import os
from typing import Any

from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
from agenttune.rag.data.hotpotqa import get_system_prompt, load_hotpotqa_splits
from agenttune.rag.retrieval.chroma_backend import ChromaBackend
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.rewards.qa_metrics import exact_match_score, extract_answer_tag, f1_score
from agenttune.rag.tools import ReadDocumentTool, SearchCorpusTool
from agenttune.rag.trajectory_utils import is_tool_step


def make_backend(name: str, index_dir: str):
    if name == "sqlite":
        return SQLiteFTSBackend(os.path.join(index_dir, "corpus.db"))
    if name == "chroma":
        return ChromaBackend(persist_dir=index_dir)
    raise ValueError(f"Unknown backend '{name}'.")


def load_model_and_tokenizer(checkpoint_spec: str):
    """checkpoint_spec: 'model_id' or 'model_id::adapter_path'."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "::" in checkpoint_spec:
        base_model_id, adapter_path = checkpoint_spec.split("::", 1)
    else:
        base_model_id, adapter_path = checkpoint_spec, None

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def evaluate_checkpoint(
    checkpoint_spec: str,
    backend_name: str,
    index_dir: str,
    eval_questions: list[dict[str, str]],
    max_steps: int = 6,
    judge=None,
    trace_dir: str | None = None,
) -> dict[str, Any]:
    backend = make_backend(backend_name, index_dir)
    tools = [SearchCorpusTool(backend), ReadDocumentTool(backend)]
    model, tokenizer = load_model_and_tokenizer(checkpoint_spec)

    # Backend-aware prompt: keyword queries for BM25, natural-language for dense.
    system_prompt = get_system_prompt(backend_name)

    rollout_fn = create_rollout_fn(
        rollout_backend="transformers",
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
    )
    questions = [q["question"] for q in eval_questions]
    result = rollout_fn(questions)

    em_scores, f1_scores, tool_call_counts = [], [], []
    groundedness_scores: list[float] = []
    trace_records = []
    for q, traj in zip(eval_questions, result["trajectories"], strict=False):
        pred = extract_answer_tag(traj.final_response)
        gold = q["answer"]
        em_scores.append(exact_match_score(pred, gold))
        f1_scores.append(f1_score(pred, gold))
        n_calls = sum(1 for s in traj.steps if is_tool_step(s))
        tool_call_counts.append(n_calls)
        trace_records.append(
            {
                "question": q["question"],
                "gold_answer": gold,
                "predicted_answer": pred,
                "tool_calls": n_calls,
                "em": em_scores[-1],
                "f1": f1_scores[-1],
            }
        )

    if judge is not None:
        from agenttune.rag.rewards.judge_eval import score_groundedness

        groundedness_scores = score_groundedness(judge, result["trajectories"])
        for rec, g in zip(trace_records, groundedness_scores, strict=False):
            rec["groundedness"] = g

    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        safe_name = checkpoint_spec.replace("/", "_").replace("::", "__adapter__")
        with open(os.path.join(trace_dir, f"{safe_name}__{backend_name}.jsonl"), "w") as f:
            for rec in trace_records:
                f.write(json.dumps(rec) + "\n")

    n = len(eval_questions)
    return {
        "checkpoint": checkpoint_spec,
        "backend": backend_name,
        "num_questions": n,
        "exact_match": sum(em_scores) / n if n else 0.0,
        "f1": sum(f1_scores) / n if n else 0.0,
        "avg_tool_calls": sum(tool_call_counts) / n if n else 0.0,
        "pct_zero_search": sum(1 for c in tool_call_counts if c == 0) / n if n else 0.0,
        "groundedness_mean": (sum(groundedness_scores) / n) if groundedness_scores else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--backends", nargs="+", default=["sqlite"])
    parser.add_argument("--index_dir_template", default="rag_experiments/indexes/{backend}")
    parser.add_argument("--hotpotqa_config", default="distractor")
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--train_size", type=int, default=1)  # unused split, keep loader happy
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--use_judge", action="store_true")
    parser.add_argument("--groq_model", default="llama-3.3-70b-versatile")
    parser.add_argument("--output_csv", default="rag_experiments/results/eval_matrix.csv")
    parser.add_argument("--trace_dir", default="rag_experiments/results/traces")
    args = parser.parse_args()

    _, eval_split = load_hotpotqa_splits(
        config=args.hotpotqa_config, train_size=args.train_size, eval_size=args.eval_size
    )
    eval_questions = [{"question": r["question"], "answer": r["answer"]} for r in eval_split]

    judge = None
    if args.use_judge:
        from agenttune.rag.rewards.judge_eval import build_groq_judge

        judge = build_groq_judge(model=args.groq_model)

    rows: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        for backend_name in args.backends:
            index_dir = args.index_dir_template.format(backend=backend_name)
            print(f"[evaluate] checkpoint={checkpoint} backend={backend_name}")
            row = evaluate_checkpoint(
                checkpoint,
                backend_name,
                index_dir,
                eval_questions,
                max_steps=args.max_steps,
                judge=judge,
                trace_dir=args.trace_dir,
            )
            rows.append(row)
            print(
                f"  -> EM={row['exact_match']:.3f} F1={row['f1']:.3f} avg_tool_calls={row['avg_tool_calls']:.2f}"
            )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluate] Wrote {len(rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
