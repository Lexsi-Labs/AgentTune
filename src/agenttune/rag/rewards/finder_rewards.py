"""
FinDER reward stack (E1 headline runs) — assembled, not novel.

Working stack (rebalanced 2026-08-14 toward the search/retrieval goal; see
_DEFAULT_WEIGHTS below for the rationale and normalized shares):

    format(0.1) + termination(0.3) + correctness_numeric(0.6)
    + golden_chunk_recall(0.8) + conciseness(0.15) + frugality(0.25)

Every component is a known method, composed from existing pieces:
  - format / termination   -> phase1_rewards (loop-without-answering fix; the
                              termination dominance ratio 0.4 that fixed the T3
                              mode collapse — do NOT dilute it).
  - correctness_numeric    -> finqa.py's numeric parsing (_extract_numbers /
                              _numbers_close, 2% band) ported onto the FinDER
                              reward path; F1 fallback for qualitative answers
                              (84.5% of FinDER is non-numeric).
  - golden_chunk_recall    -> Castform's anti-hacking leg / ART*E partial
                              credit: fraction of gold evidence CHUNKS surfaced
                              across all tool calls in the episode. Reward value
                              is chunk-level (finer for within-document CUAD
                              paths); reference-level is logged for eval.
  - conciseness            -> Castform hygiene leg (~10 lines): penalises
                              rambling final answers.
  - frugality              -> T3's FrugalRAG port (over-search penalty).

NOT included, deliberately (confirmed against FINNLP_EXPERIMENTS.md v2 §1.2/§3):
  - sql_grounding / calculator_grounding / placeholder_coverage from finqa.py —
    those belong to the FinQA *structured-table* substrate (finqa_tool.py's
    SQLDatabaseTool + calculator), which is the secondary transfer benchmark
    (§1.2), not the FinDER unstructured-retrieval headline. There are no SQL
    tools in this environment, so those rewards would be constant zero here.
  - necessity — the v2 working stack keeps only frugality from T3; necessity's
    requires_search label doesn't exist for FinDER (all questions need
    retrieval), so it would degenerate to a constant-scaled copy of
    correctness*frugality. Available via t3_rewards if an ablation wants it.
  - LLM-judge correctness — eval-time only (judge_eval.py, prompt rotation),
    kept out of the RL hot loop.

Plumbing: golden_chunk_recall needs the chunks the agent actually retrieved.
rollout_factory.py collects `retrieved_chunk_ids` per trajectory (parsed from
search_corpus tool outputs) and emits them in `_format_for_grpo`, so TRL passes
them to this reward as a kwarg — same path as `tool_call_counts`. The gold side
arrives as the `gold_chunk_ids` dataset column (TRL forwards dataset columns to
reward fns, same as `gold_answer`).
"""

import re
from collections.abc import Callable

# Reuse the FinQA numeric machinery verbatim (2% band, $/%/K/M/B/T, commas,
# parenthetical negatives) — ported to the FinDER path per FINNLP_EXPERIMENTS
# v2 §3 item 1. Underscore-private but stable within this repo; importing beats
# drifting copies.
from agenttune.agentic.rewards.builtin_rewards.finqa import (
    _extract_numbers,
    _numbers_close,
)
from agenttune.utils.score_logger import log_score

from .phase1_rewards import format_reward, termination_reward
from .qa_metrics import extract_answer_tag, f1_score, relaxed_f1_score
from .t3_rewards import frugality_reward

# ─────────────────────────────────────────────────────────────────────────────
# Numeric-tolerance correctness (ported from finqa.py, FinDER-adapted)
# ─────────────────────────────────────────────────────────────────────────────
# FinDER gold answers are GPT-o1-standardised explanations that LEAD with the
# direct answer, then show the calculation, e.g.:
#   "The Data and Access Solutions revenue increased by $111.5 million from
#    2021 to 2023, calculated as 539.2 million minus 427.7 million."
# So the FIRST number in the gold is the headline result; later numbers are
# intermediates. Matching any pred number against the first gold number is the
# strict signal; matching a later (intermediate) gold number is partial credit
# (the agent surfaced the right figures but didn't finish the calculation).
# Qualitative rows (gold has no numbers, ~84.5% of FinDER) fall back to token
# F1 — computed against both the full gold and its first sentence (the direct-
# answer sentence), taking the max, because F1 against a 700-char explanation
# would floor at ~0 and give GRPO no variance.

_SCALE_WORDS = {
    "thousand": "k",
    "million": "m",
    "billion": "b",
    "trillion": "t",
}
_SCALE_WORD_RE = re.compile(
    r"([\d,]*\.?\d+[KkMmBbTt]?)\s+(" + "|".join(_SCALE_WORDS) + r")s?\b",
    re.IGNORECASE,
)


def _normalize_scale_words(text: str) -> str:
    """Rewrite '<number> million|billion|...' -> '<number>m|b|...' so finqa.py's
    single-letter-suffix parser understands the word form too. FINNLP_EXPERIMENTS
    v2 §3 item 1 requires "2.5B" / "2,500,000,000" / "$2.5 billion" to all
    match — finqa.py alone handles the first two, this adds the third."""
    return _SCALE_WORD_RE.sub(lambda m: m.group(1) + _SCALE_WORDS[m.group(2).lower()], text)


def numeric_correctness_reward(
    prompts,
    completions,
    gold_answer: list[str] | None = None,
    tol: float = 0.02,
    partial_credit: float = 0.4,
    domain: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """Numeric-tolerance correctness with F1 fallback.

    - Gold has numbers: 1.0 if any pred number within `tol` of the FIRST gold
      number (headline result); `partial_credit` if it only matches a later
      (intermediate) gold number; else F1 fallback (usually ~0 for numeric).
    - Gold has no numbers (qualitative): token F1 vs gold (max over full gold
      and its first sentence).
    - CUAD (legal) rows: answers are clause/entity spans where SQuAD-style
      normalisation is brittle ("The ARC Group, Inc." vs "ARC Group", "30 days"
      vs "30 days after notice") — use the legal-tolerant relaxed F1 instead
      (max over full gold and its first sentence). Mixed-domain training
      passes a per-row `domain` column; the pure FinDER path leaves it unset
      (all rows fall through to the finance branch).
    """
    n = len(completions)
    if gold_answer is None:
        log_score(
            "numeric_correctness_reward",
            0.0,
            reasons=["gold_answer column missing -> 0.0 for every completion"],
            meta={"n": n},
        )
        return [0.0] * n

    scores: list[float] = []
    for i, (completion, gold) in enumerate(zip(completions, gold_answer, strict=False)):
        pred = extract_answer_tag(completion)
        gold = str(gold or "")
        if domain is not None and i < len(domain) and domain[i] == "cuad":
            first_sentence = re.split(r"(?<=[.!?])\s", gold, maxsplit=1)[0] if gold else ""
            score = max(relaxed_f1_score(pred, gold), relaxed_f1_score(pred, first_sentence))
            log_score(
                "numeric_correctness_reward",
                score,
                reasons=[f"domain=cuad -> relaxed F1 vs gold/first-sentence = {score}"],
                meta={"index": i},
            )
            scores.append(score)
            continue
        # Scale-word normalisation applies to NUMBER EXTRACTION ONLY; F1 below
        # keeps the original text so "million" still matches "million".
        pred_nums = _extract_numbers(_normalize_scale_words(pred))
        gold_nums = _extract_numbers(_normalize_scale_words(gold))

        numeric_score: float | None = None
        numeric_reason = None
        if gold_nums and pred_nums:
            if any(_numbers_close(p, gold_nums[0], tol=tol) for p in pred_nums):
                numeric_score = 1.0
                numeric_reason = f"predicted number matches headline gold number {gold_nums[0]} (tol={tol}) -> 1.0"
            elif any(_numbers_close(p, g, tol=tol) for p in pred_nums for g in gold_nums[1:]):
                numeric_score = partial_credit
                numeric_reason = f"predicted number matches an intermediate gold number (not headline {gold_nums[0]}) -> {partial_credit}"

        # F1 fallback / qualitative path. First-sentence gold isolates the
        # direct answer from the explanation (o1-standardised answers lead
        # with the result).
        first_sentence = re.split(r"(?<=[.!?])\s", gold, maxsplit=1)[0] if gold else ""
        f1 = max(f1_score(pred, gold), f1_score(pred, first_sentence))

        if numeric_score is not None:
            score = max(numeric_score, f1)
            reason = (
                numeric_reason
                if numeric_score >= f1
                else f"token-F1 ({f1}) beat numeric score ({numeric_score}) -> {score}"
            )
        else:
            score = f1
            reason = (
                f"no numeric gold/prediction to compare -> token-F1 vs gold/first-sentence = {f1}"
                if not gold_nums
                else f"prediction has no numeric value to compare against gold {gold_nums} -> token-F1 = {f1}"
            )
        log_score("numeric_correctness_reward", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Golden-chunk recall (Castform anti-hacking leg / ART*E partial credit)
# ─────────────────────────────────────────────────────────────────────────────
# Chunk ids are `{doc_id}::{chunk_index}` (corpus_loader.chunk_corpus), so
# reference-level coverage needs no side mapping — strip the suffix to get the
# reference (doc) id. Chunk-level is the reward: |gold ∩ retrieved| / |gold|.


def _doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit("::", 1)[0] if "::" in chunk_id else chunk_id


def _coverage(retrieved: list[str], gold: list[str]) -> dict[str, float]:
    gold_set = set(gold or [])
    retrieved_set = set(retrieved or [])
    if not gold_set:
        return {"chunk_recall": 0.0, "reference_recall": 0.0}
    chunk_recall = len(gold_set & retrieved_set) / len(gold_set)
    gold_docs = {_doc_of(c) for c in gold_set}
    retrieved_docs = {_doc_of(c) for c in retrieved_set}
    reference_recall = len(gold_docs & retrieved_docs) / len(gold_docs) if gold_docs else 0.0
    return {"chunk_recall": chunk_recall, "reference_recall": reference_recall}


# Side channels so the logged composed reward can expose both recall variants
# as tensorboard components. The reward VALUE is chunk-level (see below);
# reference-level is stashed for eval tables.
_LAST_CHUNK_RECALL: list[float] = []
_LAST_REFERENCE_RECALL: list[float] = []


def golden_chunk_recall_reward(
    prompts,
    completions,
    retrieved_chunk_ids: list[list[str]] | None = None,
    gold_chunk_ids: list[list[str]] | None = None,
    **kwargs,
) -> list[float]:
    """Fraction of gold evidence chunks surfaced across all search_corpus calls.

    Reward value is CHUNK-LEVEL recall (|gold ∩ retrieved| / |gold|). For the
    FinDER corpus this coincides with reference-level: build_index_finder.py
    marks EVERY chunk of each gold reference as gold, so retrieving any chunk
    of a reference counts. For within-document CUAD paths (all gold chunks in
    one contract) reference-level would be binary — a useless signal — while
    chunk-level gives true partial credit for surfacing part of the reasoning
    path, which is exactly the multi-hop retrieval behavior we train for.
    Reference-level recall is still stashed for logging/eval."""
    global _LAST_CHUNK_RECALL, _LAST_REFERENCE_RECALL
    n = len(completions)
    if retrieved_chunk_ids is None:
        retrieved_chunk_ids = [[]] * n
    if gold_chunk_ids is None:
        gold_chunk_ids = [[]] * n

    scores: list[float] = []
    chunk_recalls: list[float] = []
    reference_recalls: list[float] = []
    for i, (retrieved, gold) in enumerate(zip(retrieved_chunk_ids, gold_chunk_ids, strict=False)):
        cov = _coverage(retrieved, gold)
        scores.append(cov["chunk_recall"])
        chunk_recalls.append(cov["chunk_recall"])
        reference_recalls.append(cov["reference_recall"])
        gold_set, retrieved_set = set(gold or []), set(retrieved or [])
        if not gold_set:
            reason = "no gold chunk ids for this row -> 0.0"
        else:
            reason = (
                f"{len(gold_set & retrieved_set)}/{len(gold_set)} gold chunks retrieved "
                f"(reference-level recall={cov['reference_recall']:.2f}) -> chunk_recall={cov['chunk_recall']:.2f}"
            )
        log_score(
            "golden_chunk_recall_reward",
            cov["chunk_recall"],
            reasons=[reason],
            components={"reference_recall": cov["reference_recall"]},
            meta={"index": i},
        )
    _LAST_CHUNK_RECALL = chunk_recalls
    _LAST_REFERENCE_RECALL = reference_recalls
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Conciseness (Castform hygiene leg)
# ─────────────────────────────────────────────────────────────────────────────
# Financial answers should be a number + unit or a sentence, not a rambling
# paragraph. Full credit up to `free_chars`, linear decay to 0 at `max_chars`.
# Measured on the <answer> contents (what the user sees), not the whole
# completion.


def conciseness_reward(
    prompts,
    completions,
    free_chars: int = 280,
    max_chars: int = 1200,
    **kwargs,
) -> list[float]:
    scores: list[float] = []
    for i, completion in enumerate(completions):
        answer = extract_answer_tag(completion)
        L = len(answer)
        if L <= free_chars:
            score = 1.0
            reason = f"answer length {L} <= free_chars {free_chars} -> full credit 1.0"
        elif L >= max_chars:
            score = 0.0
            reason = f"answer length {L} >= max_chars {max_chars} -> 0.0"
        else:
            score = 1.0 - (L - free_chars) / (max_chars - free_chars)
            reason = f"answer length {L} between free_chars {free_chars} and max_chars {max_chars} -> linear decay {score:.3f}"
        log_score("conciseness_reward", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Composed FinDER stack (logged) — the E1 training reward
# ─────────────────────────────────────────────────────────────────────────────
# Weights from FINNLP_EXPERIMENTS.md v2 §3 ("working stack for the FinDER
# runs"). Termination+format = 0.4 preserves the phase-1 termination-dominance
# ratio that fixed the loop-without-answering collapse (Sprint2_readme §2/§5);
# correctness stays the dominant outcome signal; golden-chunk recall is the
# grounding/anti-hacking leg; conciseness + frugality are small hygiene terms.

# Rebalanced for the search/retrieval-optimized goal (2026-08-14). The old
# stack made correctness_numeric 52.6% of the (normalized) reward and the
# retrieval signal golden_chunk_recall just 15.8%, with conciseness/frugality
# at 5.3% each — effectively noise. New ratios, normalized to sum 1:
#   golden_chunk_recall 0.8 -> 36.4%  (THE retrieval-quality signal, top weight)
#   correctness_numeric 0.6 -> 27.3%  (outcome; correlated with retrieval, so
#                                       still learned via GCR, less memorize-
#                                       without-searching shortcut)
#   termination        0.3 -> 13.6%  (termination+format=0.4 total, keeps the
#                                       proven anti-loop-collapse ratio ≥ 15%)
#   frugality          0.25-> 11.4%  (efficient search — was dead at 5%)
#   conciseness        0.15->  6.8%  (hygiene — was dead at 5%)
#   format             0.1 ->  4.5%
# NOTE: GRPO optimizes a per-group RELATIVE advantage, so a component only
# learns if it has within-group variance × weight — verify each component's
# mean/std in training_log.json's reward/<component> channels, not just its
# weight.
_DEFAULT_WEIGHTS = {
    "format": 0.1,
    "termination": 0.3,
    "correctness_numeric": 0.6,
    "golden_chunk_recall": 0.8,
    "conciseness": 0.15,
    "frugality": 0.25,
}

_LAST_FINDER_COMPONENT_SCORES: dict[str, list[float]] = {}


def get_logged_finder_reward(weights: dict[str, float] | None = None) -> Callable:
    """FinDER training stack with per-component capture for trace + tensorboard.

    Components: format, termination, correctness_numeric, golden_chunk_recall
    (chunk-level; reference-level logged as
    golden_chunk_recall_referencelevel), conciseness, frugality. Weights
    normalised to sum 1.
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    fns = [
        format_reward,
        termination_reward,
        numeric_correctness_reward,
        golden_chunk_recall_reward,
        conciseness_reward,
        frugality_reward,
    ]
    names = [
        "format",
        "termination",
        "correctness_numeric",
        "golden_chunk_recall",
        "conciseness",
        "frugality",
    ]
    norm_w = [w[n] / sum(w.values()) for n in names]

    def _logged(completions, **kwargs):
        global _LAST_FINDER_COMPONENT_SCORES
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
        # Extra logging-only components: reference-level recall (the reward
        # value is chunk-level; eval tables report both).
        per_component["golden_chunk_recall_referencelevel"] = list(_LAST_REFERENCE_RECALL)
        _LAST_FINDER_COMPONENT_SCORES = per_component
        for i, total in enumerate(totals):
            reasons = [
                f"{n} = {per_component[n][i]:.3f} * weight {ww:.3f} = {per_component[n][i] * ww:.3f}"
                for n, ww in zip(names, norm_w, strict=False)
            ]
            log_score(
                "finder_composed_reward",
                total,
                reasons=reasons,
                components={n: per_component[n][i] for n in names},
                meta={"index": i},
            )
        return totals

    _logged.__name__ = "logged_finder_reward(" + "+".join(names) + ")"
    return _logged


def get_last_finder_component_scores() -> dict[str, list[float]]:
    """Per-component scores from the most recent FinDER reward call."""
    return _LAST_FINDER_COMPONENT_SCORES
