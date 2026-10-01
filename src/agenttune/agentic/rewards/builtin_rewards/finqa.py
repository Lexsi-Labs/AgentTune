"""
finqa_rewards.py
----------------
Reward functions for FinQA agentic GRPO training.

Each function has signature:
    fn(prompts, completions, *, answer=None, tool_call_counts=None, **kwargs) -> list[float]

Functions are composable — the trainer sums them up per completion.

Reward budget (max total = 2.0):
    format_reward              +0.1   <answer> tag present & non-trivial
    sql_grounding_reward       +0.3   used sql_query tool (0/1/2+ calls)
    calculator_grounding_reward+0.2   used calculator tool (0/1+ calls)
    placeholder_coverage_reward+0.4   numerical coverage in <answer> vs gold
    answer_correctness_reward  +1.0   final answer correct (exact or near-exact)
"""

from __future__ import annotations

import re
from typing import Any

from agenttune.utils.score_logger import log_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_answer(text: str) -> str:
    """Return content inside <answer>...</answer>, else the full text."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text.strip()


_UNIT_MAP = {
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
}


def _parse_number(token: str) -> float | None:
    """
    Try to parse a token as a number, handling:
      - leading $ / trailing % / trailing unit suffixes (K, M, B, T)
      - thousand-separator commas  (1,234,567)
      - parentheses as negation   (1,234) -> -1234
    Returns None if the token cannot be parsed.
    """
    t = token.strip()

    # Parentheses as negative  (123) -> -123
    negative = False
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
        negative = True

    # Strip leading currency symbols
    t = t.lstrip("$€£")

    # Trailing % kept — we normalise below
    is_percent = t.endswith("%")
    if is_percent:
        t = t[:-1]

    # Trailing unit suffix  (case-insensitive)
    multiplier = 1.0
    if t and t[-1].lower() in _UNIT_MAP:
        multiplier = _UNIT_MAP[t[-1].lower()]
        t = t[:-1]

    # Remove thousand-separator commas (only when pattern is ,\d{3})
    t = re.sub(r",(?=\d{3}(?:[,.]|$))", "", t)

    try:
        value = float(t) * multiplier
    except ValueError:
        return None

    if is_percent:
        value = value / 100.0

    return -value if negative else value


def _extract_numbers(text: str) -> list[float]:
    """
    Extract all numeric values from text, handling $, %, K/M/B/T, commas,
    and parenthetical negatives.
    """
    # Match: optional leading ($, £, €), optional leading -, digits with optional
    # thousand-separator commas and decimal, optional trailing %, K, M, B, T
    # Also match parenthetical negatives: (1,234.5)
    pattern = r"""
        (?:
            \([\$€£]?[\-]?\d[\d,]*\.?\d*[KkMmBbTt]?\)  # (1,234) negative
          | [\$€£]?[\-]?\d[\d,]*\.?\d*[KkMmBbTt]?%?    # normal number
        )
    """
    tokens = re.findall(pattern, text, re.VERBOSE)
    numbers = []
    for tok in tokens:
        val = _parse_number(tok)
        if val is not None:
            numbers.append(val)
    return numbers


def _numbers_close(pred_val: float, gold_val: float, tol: float = 0.02) -> bool:
    """True if pred_val is within `tol` (2 % by default) of gold_val."""
    if gold_val == 0:
        return abs(pred_val) < 1e-9
    return abs(pred_val - gold_val) / (abs(gold_val) + 1e-12) <= tol


# ---------------------------------------------------------------------------
# 1. Format reward  (+0.1)
# ---------------------------------------------------------------------------


def format_reward(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list[float]:
    """
    +0.1 if the completion wraps its final answer in <answer>...</answer>
    with at least 3 characters of content (avoids rewarding empty tags).
    """
    rewards = []
    for i, c in enumerate(completions):
        text = str(c)
        has_tag = bool(re.search(r"<answer>(.{3,}?)</answer>", text, re.IGNORECASE | re.DOTALL))
        score = 0.1 if has_tag else 0.0
        reason = (
            "<answer> tag with >=3 chars of content -> 0.1"
            if has_tag
            else "no <answer> tag with >=3 chars of content -> 0.0"
        )
        log_score("finqa.format_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# 2. SQL grounding reward  (+0.3)
# ---------------------------------------------------------------------------


def sql_grounding_reward(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list[float]:
    """
    Reward the model specifically for calling the query_finqa_tables tool.

    Why count query_finqa_tables specifically (not all tool calls)?
    FinQA answers always require looking up a financial table. A model that
    skips the query tool is guessing from context, not reasoning from evidence.

    The grep target "query_finqa_tables" matches the exact function name
    defined in finqa_tool.py — if you rename that function, update this too.

      0 calls  -> 0.0
      1 call   -> 0.2
      2+ calls -> 0.3
    """
    rewards = []
    for i, c in enumerate(completions):
        text = str(c)
        # Count invocations of the query tool by its function name
        n = len(re.findall(r"query_finqa_tables", text, re.IGNORECASE))
        if n == 0:
            score = 0.0
            reason = "query_finqa_tables never called -> 0.0"
        elif n == 1:
            score = 0.2
            reason = "query_finqa_tables called once -> 0.2"
        else:
            score = 0.3
            reason = f"query_finqa_tables called {n} times (>=2) -> 0.3"
        log_score("finqa.sql_grounding_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# 3. Calculator grounding reward  (+0.2)
# ---------------------------------------------------------------------------


def calculator_grounding_reward(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list[float]:
    """
    Reward the model for using the calculator tool.

    FinQA questions typically require at least one arithmetic step after
    retrieving values. Rewarding calculator use (separately from sql_query)
    encourages a retrieve-then-compute pattern.

      0 calls  -> 0.0
      1+ calls -> 0.2
    """
    rewards = []
    for i, c in enumerate(completions):
        text = str(c)
        n = len(re.findall(r"calculator", text, re.IGNORECASE))
        score = 0.2 if n >= 1 else 0.0
        reason = (
            f"calculator referenced {n} times -> 0.2"
            if n >= 1
            else "calculator never referenced -> 0.0"
        )
        log_score("finqa.calculator_grounding_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# 4. Placeholder / numerical coverage reward  (+0.4)
# ---------------------------------------------------------------------------


def placeholder_coverage_reward(
    prompts: list,
    completions: list,
    answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Fraction of gold numerical values present in the prediction.

    FIX over original:
      - Uses _parse_number to understand units ($, %, K/M/B) before comparing,
        so "2.5B" and "2500000000" are treated as the same value.
      - Uses _numbers_close() (2 % tolerance) instead of substring containment,
        which prevented "5" from falsely matching inside "50" or "500".
      - Extracts numbers from the <answer> block only (not the whole completion).

    Scoring:
      >= 80 % coverage -> +0.4
      >= 50 %          -> +0.2
      >= 20 %          -> +0.1
       < 20 %          ->  0.0
    """
    rewards = []
    for i, (c, gold) in enumerate(
        zip(completions, answer or [""] * len(completions), strict=False)
    ):
        pred = _extract_answer(str(c))
        gold_str = str(gold)

        gold_numbers = _extract_numbers(gold_str)
        if not gold_numbers:
            # No numeric gold — give partial credit for non-empty answer
            score = 0.1 if pred.strip() else 0.0
            reason = (
                "gold has no numeric values; prediction is non-empty -> 0.1"
                if score
                else "gold has no numeric values and prediction is empty -> 0.0"
            )
            log_score(
                "finqa.placeholder_coverage_reward", score, reasons=[reason], meta={"index": i}
            )
            rewards.append(score)
            continue

        pred_numbers = _extract_numbers(pred)
        if not pred_numbers:
            log_score(
                "finqa.placeholder_coverage_reward",
                0.0,
                reasons=["prediction contains no numeric values to compare -> 0.0"],
                meta={"index": i},
            )
            rewards.append(0.0)
            continue

        # For each gold number, check if ANY pred number is close enough
        hits = sum(1 for g in gold_numbers if any(_numbers_close(p, g) for p in pred_numbers))
        coverage = hits / len(gold_numbers)

        if coverage >= 0.8:
            score = 0.4
        elif coverage >= 0.5:
            score = 0.2
        elif coverage >= 0.2:
            score = 0.1
        else:
            score = 0.0

        log_score(
            "finqa.placeholder_coverage_reward",
            score,
            reasons=[
                f"{hits}/{len(gold_numbers)} gold numbers matched ({coverage:.0%} coverage) -> {score}"
            ],
            components={"coverage": coverage},
            meta={"index": i},
        )
        rewards.append(score)

    return rewards


# ---------------------------------------------------------------------------
# 5. Answer correctness reward  (+1.0)  — PRIMARY SIGNAL
# ---------------------------------------------------------------------------


def answer_correctness_reward(
    prompts: list,
    completions: list,
    answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Primary correctness signal.  Checks the content of <answer>...</answer>.

    Numeric answers (the majority in FinQA):
      +1.0  if the primary gold number is in pred within 2 % tolerance
      +0.5  if within 10 % tolerance  (partial credit)
       0.0  otherwise

    Text answers (question_type == 'span' etc.):
      +1.0  exact match (case-insensitive, stripped)
      +0.5  one string contains the other
       0.0  otherwise

    FIX over original numerical_match_reward:
      - Tests ALL gold numbers, not just the first one.
      - Handles unit suffixes and % via _parse_number.
      - Falls back gracefully to string comparison when no numbers present.
    """
    rewards = []
    for i, (c, gold) in enumerate(
        zip(completions, answer or [""] * len(completions), strict=False)
    ):
        pred = _extract_answer(str(c))
        gold_str = str(gold).strip()

        gold_numbers = _extract_numbers(gold_str)
        pred_numbers = _extract_numbers(pred)

        if gold_numbers:
            # --- numeric path ---
            # Best match: for each gold number find closest pred number
            best_score = 0.0
            reason = f"no predicted number within 10% of any gold number {gold_numbers} -> 0.0"
            for g in gold_numbers:
                for p in pred_numbers:
                    if _numbers_close(p, g, tol=0.02):
                        if best_score < 1.0:
                            best_score = 1.0
                            reason = f"predicted {p} within 2% of gold {g} -> 1.0"
                    elif _numbers_close(p, g, tol=0.10):
                        if best_score < 0.5:
                            best_score = 0.5
                            reason = f"predicted {p} within 10% (not 2%) of gold {g} -> 0.5"
            log_score(
                "finqa.answer_correctness_reward", best_score, reasons=[reason], meta={"index": i}
            )
            rewards.append(best_score)
        else:
            # --- text / categorical path ---
            pred_clean = pred.lower().strip()
            gold_clean = gold_str.lower().strip()
            if not gold_clean:
                score, reason = 0.0, "gold answer is empty -> 0.0"
            elif pred_clean == gold_clean:
                score, reason = 1.0, f"prediction exactly matches gold {gold_clean!r} -> 1.0"
            elif gold_clean in pred_clean or pred_clean in gold_clean:
                score, reason = (
                    0.5,
                    "prediction and gold partially overlap (substring match) -> 0.5",
                )
            else:
                score, reason = (
                    0.0,
                    f"prediction {pred_clean!r} does not match gold {gold_clean!r} -> 0.0",
                )
            log_score("finqa.answer_correctness_reward", score, reasons=[reason], meta={"index": i})
            rewards.append(score)

    return rewards
