"""
T4 — requires-search filter stage (Sprint 2 P3).

Per rag_plan_s2.md §2 T4: probe the base model without corpus; if it answers
correctly from parametric knowledge, the question doesn't need retrieval →
requires_search=False. This label feeds T3's necessity_reward (questions that
don't need search should be answered directly, not searched).

This is the "requires-search filter" stage in T4's three-stage pipeline:
  grounded generation (have) → requires-search filter (this) → difficulty scoring (T1)

Uses solve_difficulty.probe_solve_difficulty internally: if pass_rate >= threshold,
the question is answerable without retrieval → requires_search=False.

The label flows into the training dataset as a per-row `requires_search` boolean,
consumed by necessity_reward and frugality_reward in t3_rewards.py.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def label_requires_search(
    questions: list[dict[str, str]],
    model_path: str,
    k: int = 4,
    pass_threshold: float = 0.5,
    use_vllm: bool = True,
) -> list[dict[str, Any]]:
    """Label each question with requires_search=True/False.

    Probes the base model k times WITHOUT retrieval. If the model answers
    correctly in >= pass_threshold fraction of samples, the question is
    answerable from parametric knowledge → requires_search=False.

    Args:
        questions: list of {"question": str, "answer": str}
        model_path: HF model id
        k: samples per question
        pass_threshold: if pass_rate >= this, requires_search=False
        use_vllm: use vLLM for generation

    Returns:
        list of dicts with question, gold, requires_search, pass_rate,
        solve_difficulty, samples (from the probe)
    """
    from .solve_difficulty import probe_solve_difficulty

    probe_results = probe_solve_difficulty(questions, model_path, k=k, use_vllm=use_vllm)

    labeled = []
    for q, r in zip(questions, probe_results, strict=False):
        pass_rate = r["pass_rate"]
        requires_search = pass_rate < pass_threshold
        labeled.append(
            {
                "question": q["question"],
                "gold": q["answer"],
                "requires_search": requires_search,
                "pass_rate": pass_rate,
                "solve_difficulty": r["solve_difficulty"],
                "already_solved": r["already_solved"],
                "samples": r["samples"],
            }
        )

    n_needs = sum(1 for l in labeled if l["requires_search"])
    logger.info(
        f"[requires_search] {n_needs}/{len(labeled)} questions need retrieval "
        f"(pass_rate < {pass_threshold})"
    )
    return labeled


def filter_to_requires_search(
    labeled: list[dict[str, Any]],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return only questions that require search (requires_search=True).

    This is the T4 filter: drop already-answerable questions from the training
    set so RL focuses on questions that actually need retrieval. The dropped
    questions can optionally be kept for the necessity_reward's
    requires_search=False regime (to teach the model NOT to search).
    """
    kept = [q for q, l in zip(questions, labeled, strict=False) if l["requires_search"]]
    dropped = len(questions) - len(kept)
    logger.info(
        f"[requires_search] filtered: kept {len(kept)}, dropped {dropped} "
        f"(answerable without retrieval)"
    )
    return kept
