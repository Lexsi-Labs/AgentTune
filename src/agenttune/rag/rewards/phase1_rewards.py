"""
Phase 1 training rewards for GRPO. Reuses agenttune's own reward-fn
conventions and existing builtins rather than reimplementing them:
  - format_reward         -> thin re-export of the existing builtin
  - search_usage_reward   -> thin wrapper delegating to search_grounding_reward
  - rag_correctness_reward -> new: token-F1 vs the gold_answer dataset column
Composed via the existing `combine_rewards`.
"""

from collections.abc import Callable

from agenttune.agentic.rewards.builtin_rewards.use_case import (
    format_reward as _format_reward,
)
from agenttune.agentic.rewards.builtin_rewards.use_case import (
    search_grounding_reward as _search_grounding_reward,
)
from agenttune.agentic.rewards.composite import combine_rewards
from agenttune.utils.score_logger import log_score

from .qa_metrics import extract_answer_tag, f1_score

# Re-exported as-is; no copy-paste, no drift.
format_reward = _format_reward


def search_usage_reward(prompts, completions, tool_call_counts=None, **kwargs) -> list[float]:
    return _search_grounding_reward(
        prompts, completions, tool_call_counts=tool_call_counts, **kwargs
    )


def termination_reward(
    prompts, completions, tool_call_counts=None, max_searches: int = 6, **kwargs
) -> list[float]:
    """Reward for terminating the search loop with a final answer.

    The zero-shot eval (Sprint 2) showed Qwen3.5-4B loops on tool calls and
    never emits an <answer> tag — because the default reward stack pays more
    for looping (search_usage 0.4 at 3+ calls) than for answering (format 0.1).
    This reward inverts that: emitting an answer tag is the single most
    important behavior to learn, so it gets a strong positive signal, and
    looping without answering gets a penalty that grows with search count.

      - has <answer> tag            → +1.0 (terminated correctly)
      - no tag, searched 1-2 times  →  0.0 (still exploring, neutral)
      - no tag, searched 3+ times   →  negative, growing with over-search
        (penalises the loop-to-max-steps failure mode)
    """
    import re

    if tool_call_counts is None:
        tool_call_counts = [0] * len(completions)
    scores: list[float] = []
    for i, (completion, n_search) in enumerate(zip(completions, tool_call_counts, strict=False)):
        text = completion if isinstance(completion, str) else str(completion)
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL))
        if has_answer:
            score, reason = 1.0, "emitted <answer> tag -> terminated correctly, +1.0"
        elif n_search <= 2:
            score, reason = (
                0.0,
                f"no <answer> tag but only {n_search} searches so far -> neutral 0.0 (still exploring)",
            )
        else:
            # Over-searching without answering: linear penalty, capped at -1.
            over = n_search - 2
            score = max(-1.0, -0.25 * over)
            reason = f"no <answer> tag after {n_search} searches ({over} over the 2-search grace) -> penalty {score}"
        log_score("termination_reward", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


def rag_correctness_reward(
    prompts, completions, gold_answer: list[str] | None = None, **kwargs
) -> list[float]:
    """Token-F1 between the completion's <answer> tag and the gold answer."""
    if gold_answer is None:
        log_score(
            "rag_correctness_reward",
            0.0,
            reasons=["gold_answer column missing -> 0.0 for every completion"],
            meta={"n": len(completions)},
        )
        return [0.0] * len(completions)
    scores: list[float] = []
    for i, (completion, gold) in enumerate(zip(completions, gold_answer, strict=False)):
        pred = extract_answer_tag(completion)
        score = f1_score(pred, str(gold))
        log_score(
            "rag_correctness_reward",
            score,
            reasons=[f"token-F1({pred!r}, gold={str(gold)!r}) = {score}"],
            meta={"index": i},
        )
        scores.append(score)
    return scores


def get_training_reward(weights: dict[str, float] | None = None) -> Callable:
    """Composed training reward.

    Default weights (Sprint 2 rebalance): format=0.1, termination=0.3,
    search_usage=0.1, correctness=0.5.

    WHY rebalanced: the zero-shot eval showed Qwen3.5-4B loops on tool calls
    and never emits <answer> tags. The old stack (format=0.1, search_usage=0.2,
    correctness=0.7) paid MORE for looping (search_usage 0.4 at 3+ calls) than
    for answering (format 0.1) — actively rewarding the failure mode, and
    giving every rollout in a GRPO group the same ~0.4 score → zero advantage
    → no learning. The new stack makes emitting an answer tag (termination=0.3
    + format=0.1 = 0.4) at least break-even with looping, and penalises
    over-searching without answering (termination goes negative), so a group
    of rollouts has variance: the one that answers scores higher than the ones
    that loop. Correctness (0.5) is still the dominant outcome signal.

    Pass weights={"format":..,"termination":..,"search_usage":..,"correctness":..}
    to override.
    """
    defaults = {"format": 0.1, "termination": 0.3, "search_usage": 0.1, "correctness": 0.5}
    w = {**defaults, **(weights or {})}
    return combine_rewards(
        [format_reward, termination_reward, search_usage_reward, rag_correctness_reward],
        weights=[w["format"], w["termination"], w["search_usage"], w["correctness"]],
    )


# ── Per-component reward logging (Sprint 2) ─────────────────────────────────
# combine_rewards returns only the weighted total. For debugging/training
# visibility we want to see each component (format/termination/search/correct)
# per rollout, so the trace + tensorboard can show what's driving the signal.
# This module-level dict accumulates the most recent per-component scores;
# train_grpo.py's TraceLogger reads it when writing each trace line.

_LAST_COMPONENT_SCORES: dict[str, list[float]] = {}


def get_logged_training_reward(weights: dict[str, float] | None = None) -> Callable:
    """Like get_training_reward but records per-component scores for logging.

    The returned callable stashes the last batch's component scores in
    _LAST_COMPONENT_SCORES (keyed by component name → list[float] per rollout).
    Read via `get_last_component_scores()` after each reward call.
    """
    defaults = {"format": 0.1, "termination": 0.3, "search_usage": 0.1, "correctness": 0.5}
    w = {**defaults, **(weights or {})}
    fns = [format_reward, termination_reward, search_usage_reward, rag_correctness_reward]
    names = ["format", "termination", "search_usage", "correctness"]
    norm_w = [ww / sum(w.values()) for ww in [w[n] for n in names]]

    def _logged(completions, **kwargs):
        global _LAST_COMPONENT_SCORES
        per_component: dict[str, list[float]] = {n: [] for n in names}
        totals = [0.0] * len(completions)
        for fn, n, ww in zip(fns, names, norm_w, strict=False):
            try:
                scores = fn(completions=completions, **kwargs)
            except TypeError:
                scores = fn(completions, **kwargs)
            for i, s in enumerate(scores):
                s = float(s) if s is not None else 0.0
                per_component[n].append(s)
                totals[i] += ww * s
        _LAST_COMPONENT_SCORES = per_component
        for i, total in enumerate(totals):
            reasons = [
                f"{n} = {per_component[n][i]:.3f} * weight {ww:.3f} = {per_component[n][i] * ww:.3f}"
                for n, ww in zip(names, norm_w, strict=False)
            ]
            log_score(
                "phase1_composed_reward",
                total,
                reasons=reasons,
                components={n: per_component[n][i] for n in names},
                meta={"index": i},
            )
        return totals

    _logged.__name__ = "logged_reward(" + "+".join(names) + ")"
    return _logged


def get_last_component_scores() -> dict[str, list[float]]:
    """Returns the per-component scores from the most recent reward call."""
    return _LAST_COMPONENT_SCORES
