"""
Phase 0 experiment: prove that retrieved-chunk tokens are excluded from the
training loss (env_mask == 0 for tool-role tokens), using only the public
`create_rollout_fn` entry point — no access to rollout_factory internals.

Usage:
    python -m agenttune.rag.scripts.verify_masking \\
        --model_path Qwen/Qwen2.5-1.5B-Instruct \\
        --backend sqlite --index_dir rag_experiments/indexes/sqlite \\
        --output rag_experiments/phase0_masking/report.json
"""

import argparse
import json
import os
import re
from typing import Any

from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
from agenttune.rag.retrieval.chroma_backend import ChromaBackend
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.tools import ReadDocumentTool, SearchCorpusTool
from agenttune.rag.trajectory_utils import is_tool_step

DEFAULT_QUESTIONS = [
    "What nationality was the director of the film Ed Wood?",
    "Which company created the video game engine used in Half-Life?",
    "Who wrote the novel that the movie Blade Runner was based on?",
    "What year was the university founded that Alan Turing attended?",
    "Which river runs through the city where the Eiffel Tower is located?",
]


def _word_set(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _overlap_ratio(candidate: str, reference: str) -> float:
    """Fraction of reference's words also present in candidate. 0.0 if reference is empty."""
    ref_words = _word_set(reference)
    if not ref_words:
        return 0.0
    cand_words = _word_set(candidate)
    return len(cand_words & ref_words) / len(ref_words)


def run_masking_check(
    model_path: str, backend, questions: list[str], max_steps: int = 4
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from agenttune.rag.data.hotpotqa import get_system_prompt

    # Backend-aware prompt: keyword queries for BM25, natural-language for dense.
    backend_name = getattr(backend, "name", "sqlite")
    system_prompt = get_system_prompt(backend_name)

    tools = [SearchCorpusTool(backend), ReadDocumentTool(backend)]
    rollout_fn = create_rollout_fn(
        rollout_backend="transformers",
        model_path=model_path,
        tools=tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
    )
    result = rollout_fn(questions)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    per_question: list[dict[str, Any]] = []
    for i, (question, traj) in enumerate(zip(questions, result["trajectories"], strict=False)):
        comp_ids = result["completion_ids"][i] or []
        mask = result["env_mask"][i] or []
        tool_texts = [s.observation for s in traj.steps if is_tool_step(s)]
        model_texts = [s.thought for s in traj.steps if s.thought]
        combined_tool_text = " ".join(tool_texts)
        combined_model_text = " ".join(model_texts)

        masked_ids = [tid for tid, m in zip(comp_ids, mask, strict=False) if m == 0]
        unmasked_ids = [tid for tid, m in zip(comp_ids, mask, strict=False) if m == 1]
        masked_text = tokenizer.decode(masked_ids, skip_special_tokens=True) if masked_ids else ""
        unmasked_text = (
            tokenizer.decode(unmasked_ids, skip_special_tokens=True) if unmasked_ids else ""
        )

        # masked span should reproduce the tool's output (proves mask=0 == retrieved text).
        masked_vs_tool_overlap = _overlap_ratio(masked_text, combined_tool_text)
        # unmasked span should reproduce the MODEL's own generated text — not "should
        # differ from tool text", since the model legitimately paraphrases retrieved
        # facts in its own answer, which would show high word-overlap with tool text
        # despite being a completely different token span. Comparing against the
        # model's own `thought` text is the correct, unconfounded check.
        unmasked_vs_model_overlap = _overlap_ratio(unmasked_text, combined_model_text)

        # If the model never searched, there's nothing to verify for this question.
        if not tool_texts:
            spot_check_passed = True
        else:
            spot_check_passed = masked_vs_tool_overlap > 0.5 and unmasked_vs_model_overlap > 0.5

        per_question.append(
            {
                "question": question,
                "tool_calls": len(tool_texts),
                "mask_zero_token_count": len(masked_ids),
                "mask_one_token_count": len(unmasked_ids),
                "masked_text_vs_tool_output_overlap": round(masked_vs_tool_overlap, 3),
                "unmasked_text_vs_model_generation_overlap": round(unmasked_vs_model_overlap, 3),
                "spot_check_passed": spot_check_passed,
            }
        )

    any_search_happened = any(q["tool_calls"] > 0 for q in per_question)
    all_checks_passed = all(q["spot_check_passed"] for q in per_question)
    return {
        "model_path": model_path,
        "backend": backend.name,
        "num_questions": len(questions),
        "any_search_happened": any_search_happened,
        "spot_check_passed": all_checks_passed and any_search_happened,
        "per_question": per_question,
    }


def make_backend(name: str, index_dir: str):
    if name == "sqlite":
        return SQLiteFTSBackend(os.path.join(index_dir, "corpus.db"))
    if name == "chroma":
        return ChromaBackend(persist_dir=index_dir)
    raise ValueError(f"Unknown backend '{name}'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--backend", choices=["sqlite", "chroma"], default="sqlite")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--output", default="rag_experiments/phase0_masking/report.json")
    args = parser.parse_args()

    backend = make_backend(args.backend, args.index_dir)
    report = run_masking_check(
        args.model_path, backend, DEFAULT_QUESTIONS, max_steps=args.max_steps
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[verify_masking] spot_check_passed={report['spot_check_passed']} -> {args.output}")
    if not report["spot_check_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
