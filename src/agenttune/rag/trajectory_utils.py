"""
Shared helper for reading tool calls off a Trajectory Step.

Step.action's docstring comment (agentic/trajectory/dataset.py) claims a flat
shape: {"name": "read_file", "arguments": {...}}. The actual rollout engine
(_execute_trajectory) populates OpenAI-style tool_calls instead:
{"tool_calls": [{"type": "function", "function": {"name": ..., "arguments": "<json str>"}}]}.
This helper normalizes both shapes so callers don't have to know which one
they're getting.
"""

import json
from typing import Any


def extract_tool_calls(step) -> list[dict[str, Any]]:
    """Returns a list of {"name": str, "arguments": dict} for this step, [] if none."""
    action = step.action
    if not action:
        return []

    if action.get("name"):
        return [{"name": action["name"], "arguments": action.get("arguments", {})}]

    calls = action.get("tool_calls")
    if not calls:
        return []

    out = []
    for call in calls:
        fn = call.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
        out.append({"name": fn.get("name"), "arguments": args})
    return out


def is_tool_step(step) -> bool:
    return bool(extract_tool_calls(step))


def extract_question_text(task: Any) -> str:
    """Trajectory.task is a plain question string when the rollout is driven
    directly with string prompts (e.g. verify_masking.py's manual questions),
    but during real GRPO training it's the full chat-formatted prompt — a
    list of {"role", "content"} dicts, as produced by
    agenttune.rag.data.hotpotqa.to_grpo_dataset's `prompt` column, since the
    trainer feeds dataset rows straight through to the rollout function.
    This normalizes both to the underlying question text."""
    if isinstance(task, str):
        return task
    if isinstance(task, list):
        for msg in reversed(task):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return str(task)
