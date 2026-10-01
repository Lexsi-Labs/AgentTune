"""
Eval-only LLM-judge groundedness scoring. Not used as a training reward —
kept out of GRPO's hot loop (Groq rate limits / latency), only invoked from
evaluate.py on the small held-out set. Wraps the existing LLMJudge rather
than reimplementing an LLM-call harness.
"""

import os
from typing import Any

from agenttune.agentic.rewards.llm_judge import LLMJudge

GROUNDEDNESS_RUBRIC = """You are grading whether an AI agent's answer to a question is
GROUNDED in the passages it retrieved via its search tool during the conversation
(as opposed to hallucinated), and whether the answer is actually correct given the
gold answer provided. Score from 0.0 (ungrounded/wrong) to 1.0 (fully grounded and
correct). Consider:
  - Did the agent search before answering?
  - Is the final answer supported by the retrieved passages in the transcript?
  - Does the final answer match the gold answer's meaning (not necessarily verbatim)?
Respond with only a single float between 0.0 and 1.0."""


def build_groq_judge(
    model: str = "llama-3.3-70b-versatile",
    api_key: str | None = None,
) -> LLMJudge:
    """Groq is OpenAI-compatible; uses the `--api groq/llama-3.3-70b-versatile`
    convention (model id without the 'groq/' provider prefix here, since
    base_url already scopes it to Groq)."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not set (env var or api_key=...).")
    return LLMJudge(
        model=model,
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        absolute_rubric=GROUNDEDNESS_RUBRIC,
    )


def score_groundedness(judge: LLMJudge, trajectories: list[Any]) -> list[float]:
    """One groundedness score per trajectory, via judge.evaluate_trajectory."""
    return [judge.evaluate_trajectory(t.task, t) for t in trajectories]
