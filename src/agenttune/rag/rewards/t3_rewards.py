"""
T3 — retrieval-necessity + frugality rewards (Sprint 2).

Splits the old single `search_usage_reward` (tiered by raw call count) into two
distinct primitives, per rag_plan_s2.md §5:

  - necessity_reward  — should the agent have searched AT ALL? Rewards answering
    directly from parametric knowledge when the question is answerable without
    retrieval (requires_search=False), and searching when it isn't. Source:
    IKEA (arxiv 2505.07596) r_kb — implemented from the paper, NOT from their
    released code (the released `ikea_knowledge_r1` repo is an open-r1 math
    fork; r_kb is not in it).

  - frugality_reward   — given that searching was needed, how MANY searches?
    Penalises searches beyond the minimum, log-scaled and capped, with a
    hardness bonus for hitting the optimal count exactly. Source: FrugalRAG
    `train_grpo.py` `_calculate_search_efficiency_reward` (L622) +
    `_find_optimal_search_count` (L595) — lifted directly.

Both follow the existing reward-fn calling convention
(`fn(prompts, completions, **kwargs)` where kwargs carries `tool_call_counts`,
`gold_answer`, and the new T3 labels `requires_search` / `optimal_search_count`).
Composed through the existing `combine_rewards` like the Phase-1 rewards.

These are ADDITIVE to phase1_rewards — they do not replace it. A training run
chooses its reward stack via `get_training_reward(weights=...)`; T3 just adds
two more callables to choose from.
"""

from collections.abc import Callable

from agenttune.agentic.rewards.composite import combine_rewards
from agenttune.utils.score_logger import log_score

from .phase1_rewards import format_reward, rag_correctness_reward, termination_reward
from .qa_metrics import exact_match_score, extract_answer_tag

# ─────────────────────────────────────────────────────────────────────────────
# Necessity (IKEA r_kb, implemented from arxiv 2505.07596)
# ─────────────────────────────────────────────────────────────────────────────
# Paper formula (paraphrased from §3.2):
#   r_ans = 1 if EM(pred, gold) else 0            # answer correctness
#   r_kb  = r_ans * (1 - n_searches / N)          # linear fewer-searches reward,
#                                                  # only when correct
#   with an over-retrieval violation: if the agent searched far more than needed,
#   r_kb is clamped (the paper uses a hard -1 on egregious over-retrieval; we
#   use a softer capped penalty so the signal stays a smooth gradient for RL).
#
# The `requires_search` label (from T4's requires-search filter, or set
# heuristically) gates the two regimes:
#   - requires_search=False (answerable from parametric knowledge):
#       searching at all is penalised; answering directly + correctly is max reward.
#   - requires_search=True (needs retrieval):
#       necessity collapses into frugality-style "search as few times as needed".
# When `requires_search` is not provided, we fall back to a conservative
# default (treat as needs-search) so the reward never punishes a correct
# search-then-answer — the safe, non-sabotaging default.


def necessity_reward(
    prompts,
    completions,
    tool_call_counts: list[int] | None = None,
    gold_answer: list[str] | None = None,
    requires_search: list[bool] | None = None,
    max_searches: int = 6,
    **kwargs,
) -> list[float]:
    """IKEA r_kb: reward correct answers, scaled by how few searches they took.

    - requires_search=False rows: max reward for 0 searches + correct answer;
      searching penalises (the agent should have answered from memory).
    - requires_search=True rows (or unknown): reward = EM * (1 - n/N), so a
      correct answer with fewer searches scores higher; 0 if wrong.
    """
    n = len(completions)
    if tool_call_counts is None:
        tool_call_counts = [0] * n
    if gold_answer is None:
        gold_answer = [""] * n
    if requires_search is None:
        # Conservative default: assume retrieval is needed. Never punish a
        # correct search-then-answer; only the frugality reward penalises
        # over-searching within the needs-search regime.
        requires_search = [True] * n

    scores: list[float] = []
    for i, (comp, gold, n_search, needs) in enumerate(
        zip(completions, gold_answer, tool_call_counts, requires_search, strict=False)
    ):
        pred = extract_answer_tag(comp)
        correct = 1.0 if (gold and exact_match_score(pred, str(gold)) == 1.0) else 0.0

        if not needs:
            # Answerable without retrieval.
            if correct and n_search == 0:
                score, reason = (
                    1.0,
                    "requires_search=False, answered correctly with 0 searches -> ideal 1.0",
                )
            elif correct and n_search > 0:
                # Correct but searched unnecessarily — partial credit, decaying
                # with each unnecessary search.
                score = max(0.0, 1.0 - n_search / max(max_searches, 1))
                reason = f"requires_search=False, correct but searched {n_search} times unnecessarily -> decayed {score}"
            else:
                score, reason = (
                    0.0,
                    "requires_search=False, wrong answer -> 0.0 regardless of searching",
                )
        else:
            # Needs retrieval: r_kb = EM * (1 - n/N). Fewer searches + correct
            # = higher reward. Wrong answer always 0 (r_ans gates r_kb).
            if correct:
                score = max(0.0, 1.0 - n_search / max(max_searches, 1))
                reason = f"requires_search=True, correct answer with {n_search} searches (of max {max_searches}) -> {score}"
            else:
                score, reason = (
                    0.0,
                    "requires_search=True, wrong answer -> r_ans=0 gates r_kb to 0.0",
                )
        log_score("necessity_reward", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Frugality (FrugalRAG, lifted from src/train/train_grpo.py)
# ─────────────────────────────────────────────────────────────────────────────
# FrugalRAG's _find_optimal_search_count: the minimum number of searches at
# which the gold supporting chunks are first retrieved (computed offline from
# gold chunk IDs). _calculate_search_efficiency_reward then scores:
#   - perfect timing (n == optimal): full reward + hardness bonus
#   - under-search (n < optimal):    partial, scaled by how close
#   - over-search  (n > optimal):    log-scaled penalty, capped
# We lift the shape; the optimal count comes from the dataset's
# `optimal_search_count` column (T4/synthesis emits it; falls back to a
# heuristic of 1 search per gold hop when absent).


def _heuristic_optimal_search_count(gold_answer: str, requires_search: bool) -> int:
    """Fallback when no gold optimal_search_count is available.
    HotpotQA is multi-hop; a reasonable lower bound is 1 search per hop. We
    can't see hops from gold_answer alone, so default to 2 (typical HotpotQA
    hop count) when retrieval is needed, 0 otherwise."""
    return 2 if requires_search else 0


def frugality_reward(
    prompts,
    completions,
    tool_call_counts: list[int] | None = None,
    gold_answer: list[str] | None = None,
    requires_search: list[bool] | None = None,
    optimal_search_count: list[int] | None = None,
    max_searches: int = 6,
    hardness_bonus: float = 0.2,
    **kwargs,
) -> list[float]:
    """FrugalRAG search-efficiency: reward hitting the optimal search count,
    penalise over-search (log-scaled, capped). Only meaningful when retrieval
    was needed; rows that don't need search get 0 (necessity handles them)."""
    import math

    n = len(completions)
    if tool_call_counts is None:
        tool_call_counts = [0] * n
    if gold_answer is None:
        gold_answer = [""] * n
    if requires_search is None:
        requires_search = [True] * n
    if optimal_search_count is None:
        optimal_search_count = [
            _heuristic_optimal_search_count(g, rs)
            for g, rs in zip(gold_answer, requires_search, strict=False)
        ]

    scores: list[float] = []
    for i, (n_search, optimal, needs) in enumerate(
        zip(tool_call_counts, optimal_search_count, requires_search, strict=False)
    ):
        if not needs:
            # No retrieval needed — frugality is vacuous; necessity covers it.
            log_score(
                "frugality_reward",
                0.0,
                reasons=[
                    "requires_search=False -> frugality is vacuous, necessity_reward covers it -> 0.0"
                ],
                meta={"index": i},
            )
            scores.append(0.0)
            continue
        if optimal <= 0:
            optimal = 1  # guard against div-by-zero / nonsensical optimal

        if n_search == optimal:
            # Perfect timing: full reward + hardness bonus for hitting it exactly.
            score = 1.0 + hardness_bonus
            reason = f"searched exactly the optimal {optimal} times -> full reward + hardness bonus = {score}"
        elif n_search < optimal:
            # Under-search: scaled by how close to optimal (still retrieved
            # something, but not enough — likely a partial/wrong answer).
            score = 0.5 * (n_search / optimal)
            reason = (
                f"under-searched: {n_search} of optimal {optimal} -> scaled partial credit {score}"
            )
        else:
            # Over-search: log-scaled penalty, capped at 0.
            # reward = max(0, 1 - log(1 + over) / log(1 + max_searches))
            over = n_search - optimal
            penalty = math.log(1 + over) / math.log(1 + max(max_searches, 1))
            score = max(0.0, 1.0 - penalty)
            reason = f"over-searched by {over} beyond optimal {optimal} -> log-scaled penalty, score {score}"
        log_score("frugality_reward", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Composed T3 reward (correctness + necessity + frugality + format)
# ─────────────────────────────────────────────────────────────────────────────
# Default T3 stack: format (small, keeps tags) + correctness (the outcome
# signal) + necessity (should-I-search) + frugality (how-many). We DROP the old
# search_usage_reward here because necessity+frugality supersede it — keeping
# all three would double-count search behaviour. Callers who want the old
# Phase-1 stack can still use phase1_rewards.get_training_reward().


def get_t3_reward(weights: dict[str, float] | None = None) -> Callable:
    """T3 reward stack: format + correctness + necessity + frugality.

    Default weights: format=0.1, correctness=0.5, necessity=0.2, frugality=0.2.
    Pass weights={"format":..,"correctness":..,"necessity":..,"frugality":..}
    to override. Weights are normalised by combine_rewards, so absolute scale
    is free.
    """
    defaults = {
        "format": 0.1,
        "correctness": 0.5,
        "necessity": 0.2,
        "frugality": 0.2,
    }
    w = {**defaults, **(weights or {})}
    return combine_rewards(
        [format_reward, rag_correctness_reward, necessity_reward, frugality_reward],
        weights=[w["format"], w["correctness"], w["necessity"], w["frugality"]],
    )


# ── Logged T3 reward (per-component capture for trace + tensorboard) ──────────
# Mirrors phase1_rewards.get_logged_training_reward: stashes per-component scores
# so train_grpo.py's TraceLogger + RewardSignalLogger can log them.
# IMPORTANT: includes termination_reward (answer-tag signal). The pure T3 stack
# (format+correctness+necessity+frugality) drops termination, but the zero-shot
# eval (Sprint2_readme §5.3-5.6) proved termination is ESSENTIAL for GRPO
# variance — without it, all-looping groups have zero variance → zero gradient.
# necessity/frugality both gate on correctness (EM), which is 0 when no answer
# tag is emitted, so they can't provide the answer-vs-loop variance signal.
# termination fills that gap. This is the training-ready T3 stack.

_LAST_T3_COMPONENT_SCORES: dict[str, list[float]] = {}


def get_logged_t3_reward(weights: dict[str, float] | None = None) -> Callable:
    """Like get_t3_reward but records per-component scores for logging, and
    includes termination_reward (answer-tag signal) for GRPO variance.

    Default weights: format=0.1, termination=0.2, correctness=0.4,
    necessity=0.15, frugality=0.15.
    """
    defaults = {
        "format": 0.1,
        "termination": 0.2,
        "correctness": 0.4,
        "necessity": 0.15,
        "frugality": 0.15,
    }
    w = {**defaults, **(weights or {})}
    fns = [
        format_reward,
        termination_reward,
        rag_correctness_reward,
        necessity_reward,
        frugality_reward,
    ]
    names = ["format", "termination", "correctness", "necessity", "frugality"]
    norm_w = [ww / sum(w.values()) for ww in [w[n] for n in names]]

    def _logged(completions, **kwargs):
        global _LAST_T3_COMPONENT_SCORES
        per_component: dict[str, list[float]] = {n: [] for n in names}
        totals = [0.0] * len(completions)
        for fn, n, ww in zip(fns, names, norm_w, strict=False):
            try:
                scores = fn(completions=completions, **kwargs)
            except TypeError:
                scores = fn(completions=completions)
            for i, s in enumerate(scores):
                s = float(s) if s is not None else 0.0
                per_component[n].append(s)
                totals[i] += ww * s
        _LAST_T3_COMPONENT_SCORES = per_component
        for i, total in enumerate(totals):
            reasons = [
                f"{n} = {per_component[n][i]:.3f} * weight {ww:.3f} = {per_component[n][i] * ww:.3f}"
                for n, ww in zip(names, norm_w, strict=False)
            ]
            log_score(
                "t3_composed_reward",
                total,
                reasons=reasons,
                components={n: per_component[n][i] for n in names},
                meta={"index": i},
            )
        return totals

    _logged.__name__ = "logged_t3_reward(" + "+".join(names) + ")"
    return _logged


def get_last_t3_component_scores() -> dict[str, list[float]]:
    """Returns the per-component scores from the most recent T3 reward call."""
    return _LAST_T3_COMPONENT_SCORES


# ─────────────────────────────────────────────────────────────────────────────
# Combined reward (phase-1 termination dominance + T3 necessity/frugality)
# ─────────────────────────────────────────────────────────────────────────────
# The 150-step M1+T3 run (m1_t3_long) collapsed to EM=0.000: 40/40 eval
# questions never emitted an <answer> tag, looping until max_steps instead.
# Diagnosis: get_logged_t3_reward's default weights give termination only 0.2
# (vs Sprint 2's phase-1 fix, which made termination+format=0.4 — see
# sprint2_results/README.md §2/§5, "the new stack makes emitting an answer
# tag at least break-even with looping"). T3 also ADDS necessity (0.15) and
# frugality (0.15), which award partial-ish credit keyed on search count and
# can be gamed by search patterns independent of ever answering. With only 40
# training questions and 150 steps (3.75 epochs), the policy found and
# exploited that loophole: keep searching, never answer, still collect reward.
#
# Fix: restore phase-1's termination dominance (format+termination=0.4, same
# ratio as the original fix) while keeping necessity+frugality as smaller
# supplementary terms (0.1 each) that add nuance on top of, not instead of,
# the answer-or-penalty signal.

_LAST_COMBINED_COMPONENT_SCORES: dict[str, list[float]] = {}


def get_combined_reward(weights: dict[str, float] | None = None) -> Callable:
    """Combined stack: phase-1's termination dominance + T3's necessity/frugality.

    Default weights: format=0.1, termination=0.3, correctness=0.4,
    necessity=0.1, frugality=0.1. Termination+format=0.4 matches the ratio
    that fixed the original loop-without-answering failure mode (Sprint 2
    README §2/§5); necessity+frugality are kept small so they add nuance
    about search efficiency without creating a search-forever loophole.
    """
    defaults = {
        "format": 0.1,
        "termination": 0.3,
        "correctness": 0.4,
        "necessity": 0.1,
        "frugality": 0.1,
    }
    w = {**defaults, **(weights or {})}
    return combine_rewards(
        [
            format_reward,
            termination_reward,
            rag_correctness_reward,
            necessity_reward,
            frugality_reward,
        ],
        weights=[w["format"], w["termination"], w["correctness"], w["necessity"], w["frugality"]],
    )


def get_logged_combined_reward(weights: dict[str, float] | None = None) -> Callable:
    """Like get_combined_reward but records per-component scores for logging."""
    defaults = {
        "format": 0.1,
        "termination": 0.3,
        "correctness": 0.4,
        "necessity": 0.1,
        "frugality": 0.1,
    }
    w = {**defaults, **(weights or {})}
    fns = [
        format_reward,
        termination_reward,
        rag_correctness_reward,
        necessity_reward,
        frugality_reward,
    ]
    names = ["format", "termination", "correctness", "necessity", "frugality"]
    norm_w = [ww / sum(w.values()) for ww in [w[n] for n in names]]

    def _logged(completions, **kwargs):
        global _LAST_COMBINED_COMPONENT_SCORES
        per_component: dict[str, list[float]] = {n: [] for n in names}
        totals = [0.0] * len(completions)
        for fn, n, ww in zip(fns, names, norm_w, strict=False):
            try:
                scores = fn(completions=completions, **kwargs)
            except TypeError:
                scores = fn(completions=completions)
            for i, s in enumerate(scores):
                s = float(s) if s is not None else 0.0
                per_component[n].append(s)
                totals[i] += ww * s
        _LAST_COMBINED_COMPONENT_SCORES = per_component
        for i, total in enumerate(totals):
            reasons = [
                f"{n} = {per_component[n][i]:.3f} * weight {ww:.3f} = {per_component[n][i] * ww:.3f}"
                for n, ww in zip(names, norm_w, strict=False)
            ]
            log_score(
                "combined_composed_reward",
                total,
                reasons=reasons,
                components={n: per_component[n][i] for n in names},
                meta={"index": i},
            )
        return totals

    _logged.__name__ = "logged_combined_reward(" + "+".join(names) + ")"
    return _logged


def get_last_combined_component_scores() -> dict[str, list[float]]:
    """Returns the per-component scores from the most recent combined reward call."""
    return _LAST_COMBINED_COMPONENT_SCORES


# ─────────────────────────────────────────────────────────────────────────────
# M2 reward (combined stack + decision-token bonus)
# ─────────────────────────────────────────────────────────────────────────────
# M2 (rag/memory/m2_decisions.py) extends M1 with an explicit
# <decision:MEMORY_OP:ACTION_OP> token the model emits each turn.
# `decision_reward` is a small additive bonus on top of whichever main stack
# is used (per its own docstring) — it must NOT dominate, since the outcome
# signal (correctness/termination) is what actually matters; the bonus just
# nudges the model toward emitting the token at all, and toward answer-now
# when it has evidence. Built on get_combined_reward rather than get_t3_reward
# since combined already fixed the termination-dilution mode collapse (see
# "Known issues" in the RAG sprint README) — M2 should not reintroduce it.

_LAST_M2_COMPONENT_SCORES: dict[str, list[float]] = {}


def get_m2_reward(weights: dict[str, float] | None = None) -> Callable:
    """M2 reward stack: combined (format+termination+correctness+necessity+
    frugality) plus a small decision-token bonus.

    Default weights: format=0.1, termination=0.3, correctness=0.4,
    necessity=0.1, frugality=0.1, decision=0.05 (added on top, then
    renormalised by combine_rewards so the stack still sums to 1.0 — decision
    ends up ~4.5% of the total, matching decision_reward's own "small
    additive bonus" design).
    """
    defaults = {
        "format": 0.1,
        "termination": 0.3,
        "correctness": 0.4,
        "necessity": 0.1,
        "frugality": 0.1,
        "decision": 0.05,
    }
    w = {**defaults, **(weights or {})}
    from agenttune.rag.memory.m2_decisions import decision_reward

    return combine_rewards(
        [
            format_reward,
            termination_reward,
            rag_correctness_reward,
            necessity_reward,
            frugality_reward,
            decision_reward,
        ],
        weights=[
            w["format"],
            w["termination"],
            w["correctness"],
            w["necessity"],
            w["frugality"],
            w["decision"],
        ],
    )


def get_logged_m2_reward(weights: dict[str, float] | None = None) -> Callable:
    """Like get_m2_reward but records per-component scores for logging."""
    defaults = {
        "format": 0.1,
        "termination": 0.3,
        "correctness": 0.4,
        "necessity": 0.1,
        "frugality": 0.1,
        "decision": 0.05,
    }
    w = {**defaults, **(weights or {})}
    from agenttune.rag.memory.m2_decisions import decision_reward

    fns = [
        format_reward,
        termination_reward,
        rag_correctness_reward,
        necessity_reward,
        frugality_reward,
        decision_reward,
    ]
    names = ["format", "termination", "correctness", "necessity", "frugality", "decision"]
    norm_w = [ww / sum(w.values()) for ww in [w[n] for n in names]]

    def _logged(completions, **kwargs):
        global _LAST_M2_COMPONENT_SCORES
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
        _LAST_M2_COMPONENT_SCORES = per_component
        for i, total in enumerate(totals):
            reasons = [
                f"{n} = {per_component[n][i]:.3f} * weight {ww:.3f} = {per_component[n][i] * ww:.3f}"
                for n, ww in zip(names, norm_w, strict=False)
            ]
            log_score(
                "m2_composed_reward",
                total,
                reasons=reasons,
                components={n: per_component[n][i] for n in names},
                meta={"index": i},
            )
        return totals

    _logged.__name__ = "logged_m2_reward(" + "+".join(names) + ")"
    return _logged


def get_last_m2_component_scores() -> dict[str, list[float]]:
    """Returns the per-component scores from the most recent M2 reward call."""
    return _LAST_M2_COMPONENT_SCORES
