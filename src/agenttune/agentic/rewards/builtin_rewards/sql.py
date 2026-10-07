"""
Built-in reward functions for agentic GRPO training.
Add new functions here — they are auto-discovered via REWARD_REGISTRY below.
"""

import logging
import re

from agenttune.agentic.rollout_engines.tool_call_parse import _THINK_BLOCK_RE
from agenttune.utils.score_logger import log_score

logger = logging.getLogger(__name__)

# A reasoning block cut off by the token limit has no closing tag; drop it to the end.
_OPEN_THINK_RE = re.compile(r"(?:<think>|<\|START_THINKING\|>).*\Z", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove reasoning blocks (Qwen ``<think>``, Cohere ``<|START_THINKING|>``,
    Harmony analysis) so only the answer itself is scored. Text without them is
    returned unchanged."""
    return _OPEN_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", text))


# ── Individual reward functions ───────────────────────────────────────────────


def correctness_reward(completions, answer=None, **kwargs):
    """Reward *yes*/*no* correctness enclosed in stars."""
    rewards = []
    for i, (completion, ans) in enumerate(zip(completions, answer or [], strict=False)):
        raw = (
            completion[-1]["content"].lower()
            if isinstance(completion, list)
            else str(completion).lower()
        )
        match = re.search(r"\*(yes|no)\*", raw)
        guess = match.group(1) if match else None
        if guess is None:
            score = -0.5
            reason = "no *yes*/*no* guess found in completion -> -0.5"
        elif guess == ans.lower():
            score = 0.6
            reason = f"guess {guess!r} matches gold {ans!r} -> 0.6"
        else:
            score = -1.0
            reason = f"guess {guess!r} does not match gold {ans!r} -> -1.0"
        log_score("sql.correctness_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


def structure_reward(completions, **kwargs):
    """Reward proper tool-call → response → content structure."""
    rewards = []
    for i, completion in enumerate(completions):
        has_call = has_response = has_other = False
        for turn in completion if isinstance(completion, list) else []:
            role = turn.get("role")
            if role == "assistant" and turn.get("tool_calls"):
                has_call = True
            elif role == "tool":
                has_response = True
            elif turn.get("content", "").strip() not in ["", "<think>"]:
                has_other = True
        if has_call and has_response:
            score = 0.1 if has_other else 0.05
            reason = (
                "tool call + tool response + other content -> 0.1"
                if has_other
                else "tool call + tool response, no other content -> 0.05"
            )
        elif has_call:
            score = -0.15
            reason = "tool call made but no tool response -> -0.15"
        else:
            score = 0.0
            reason = "no tool call made -> 0.0"
        log_score("sql.structure_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


def query_reward(completions, answer=None, **kwargs):
    """Reward effective SQL query strategy."""
    rewards = []
    for i, (completion, ans) in enumerate(zip(completions, answer or [], strict=False)):
        reward = 0.0
        reasons = []
        sql_queries, tool_results = [], []
        for turn in completion if isinstance(completion, list) else []:
            if turn.get("tool_calls"):
                for call in turn["tool_calls"]:
                    sql = call["function"]["arguments"].get("sql_command", "").lower()
                    sql_queries.append(sql)
            if turn.get("role") == "tool" and turn.get("content"):
                tool_results.append(turn["content"])
        if len(sql_queries) > 3:
            reward -= 1.5
            reasons.append(f"{len(sql_queries)} SQL queries (>3) -> -1.5")
        where_count = 0
        for q in sql_queries:
            if "limit 1" in q:
                reward -= 1.0
                reasons.append(f"query {q!r} uses 'limit 1' -> -1.0")
            if " where " not in q:
                reward -= 0.5
                reasons.append(f"query {q!r} has no WHERE clause -> -0.5")
            else:
                where_count += 1
        if where_count:
            bonus = min(where_count, 3) * 0.4
            reward += bonus
            reasons.append(f"{where_count} queries with WHERE (capped at 3) -> +{bonus}")
        combined_results, error_detected = [], False
        for res in tool_results:
            if isinstance(res, dict) and "error" in res:
                error_detected = True
            elif isinstance(res, list):
                combined_results.extend(res)
        if error_detected:
            reward -= 2.0
            reasons.append("tool result contained an error -> -2.0")
        elif not sql_queries:
            reward -= 1.5
            reasons.append("no SQL queries issued at all -> -1.5")
        else:
            has_hits = len(combined_results) > 0
            correct_answer = (ans or "").lower()
            if (has_hits and correct_answer == "yes") or (not has_hits and correct_answer == "no"):
                reward += 2.0
                reasons.append(
                    f"has_hits={has_hits} agrees with gold answer {correct_answer!r} -> +2.0"
                )
            else:
                reward -= 1.5
                reasons.append(
                    f"has_hits={has_hits} disagrees with gold answer {correct_answer!r} -> -1.5"
                )
        log_score("sql.query_reward", reward, reasons=reasons, meta={"index": i})
        rewards.append(reward)
    return rewards


def reward_correct_answer(completions, answer=None, **kwargs):
    """Exact/near-miss numeric answer reward."""
    scores = []
    for i, (comp, gt) in enumerate(zip(completions, answer or [], strict=False)):
        final_text = ""
        if isinstance(comp, list):
            for msg in reversed(comp):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content", ""))
                    break
        else:
            final_text = str(comp)
        # A number the model only mentions while reasoning is not its answer.
        final_text = _strip_thinking(final_text)
        numbers = re.findall(r"-?\d+\.?\d*", final_text)
        try:
            gt_val = float(gt)
        except (ValueError, TypeError):
            want = str(gt).strip()
            score = 1.5 if want and final_text.strip() == want else 0.0
            reason = (
                f"non-numeric gold {gt!r}: exact text match -> 1.5"
                if score
                else f"non-numeric gold {gt!r}: text does not match -> 0.0"
            )
            log_score("sql.reward_correct_answer", score, reasons=[reason], meta={"index": i})
            scores.append(score)
            continue
        score = 0.0
        reason = f"no number within tolerance of gold {gt_val} found in completion -> 0.0"
        for n in numbers:
            try:
                pred = float(n)
                if abs(pred - gt_val) < 1e-3:
                    score = 1.5
                    reason = f"predicted {pred} matches gold {gt_val} exactly -> 1.5"
                    break
                if gt_val != 0 and abs(pred - gt_val) / abs(gt_val) < 0.01:
                    if score < 0.5:
                        score = 0.5
                        reason = f"predicted {pred} within 1% of gold {gt_val} -> 0.5"
            except ValueError:
                continue
        log_score("sql.reward_correct_answer", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


def reward_tool_used(completions, **kwargs):
    """Reward if model called at least one tool."""
    counts = kwargs.get("tool_call_counts")
    convos = kwargs.get("conversations")
    scores = []
    for i, comp in enumerate(completions):
        if counts is not None and i < len(counts) and int(counts[i] or 0) > 0:
            log_score(
                "sql.reward_tool_used",
                1.0,
                reasons=[f"tool_call_counts[{i}]={counts[i]} > 0 -> 1.0"],
                meta={"index": i},
            )
            scores.append(1.0)
            continue
        msgs = comp if isinstance(comp, list) else []
        if not msgs and convos is not None and i < len(convos):
            msgs = convos[i] or []
        used = any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in msgs)
        score = 1.0 if used else 0.0
        reason = (
            "found a message with role='tool' -> 1.0"
            if used
            else "no tool-role message found -> 0.0"
        )
        log_score("sql.reward_tool_used", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


def reward_concise_answer(completions, **kwargs):
    """Small bonus for short final replies (<= 80 words)."""
    scores = []
    for i, comp in enumerate(completions):
        final_text = ""
        if isinstance(comp, list):
            for msg in reversed(comp):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content", ""))
                    break
        else:
            final_text = str(comp)
        n_words = len(final_text.split())
        score = 0.3 if n_words <= 80 else 0.0
        reason = (
            f"final reply has {n_words} words (<=80) -> 0.3"
            if score
            else f"final reply has {n_words} words (>80) -> 0.0"
        )
        log_score("sql.reward_concise_answer", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


def format_reward(completions, **kwargs):
    """Reward well-formed markdown or structured output."""
    scores = []
    for i, comp in enumerate(completions):
        final_text = ""
        if isinstance(comp, list):
            for msg in reversed(comp):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content", ""))
                    break
        else:
            final_text = str(comp)
        has_structure = bool(re.search(r"(\*\*|#{1,3} |\n-\s|\n\d+\.\s|```)", final_text))
        score = 0.2 if has_structure else 0.0
        reason = (
            "markdown/structured formatting detected -> 0.2"
            if has_structure
            else "no markdown/structured formatting -> 0.0"
        )
        log_score("sql.format_reward", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


# ── Registry — add new functions here, nothing else needs to change ───────────
