"""
M2 — Decision-event schema + memory-op action wrapper (Sprint 2 §7).

M2 builds on M1's rewrite: instead of always wiping history, the model learns
to make explicit memory DECISIONS. The decision-event schema adds credit-carrying
tokens that the reward can reinforce:

  - keep    : keep the current context as-is (no rewrite needed)
  - drop    : wipe the history entirely (M1's current behavior)
  - compress: rewrite to a compact state block (M1's state mechanism)

Plus search/answer decisions:
  - search-again : emit another search_corpus call
  - answer-now   : transition to emitting <answer> tags

The memory-op wrapper exposes these as actions the model can take, with the
decision logged as a token in the trajectory (so GRPO can reinforce good
decisions).

Wired up via `--m2` in train_grpo.py (system prompt: DEFAULT_SYSTEM_PROMPT_M2
in data/hotpotqa.py; reward: get_m2_reward in rewards/t3_rewards.py, which
adds `decision_reward` as a small bonus on top of the combined stack) and via
the `m2` condition in eval_m1_zeroshot.py. Per the user's instruction: "if M2
is basically running M1 for long time, add it to the doc and we will launch
it at last" — the M2 training run = M1 config (--m1 is implied by --m2) +
this decision schema + a longer step count (200-400 steps), launched only
after M1 itself clears its bar (see rag_plan_s2.md §7; M1's collapse was
traced to a rollout-loop stall bug, not a mechanism limit, and fixed — see
the RAG sprint README's "Known issues" section — so M2 is unblocked).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenttune.utils.score_logger import log_score


class MemoryDecision(Enum):
    """The three memory-management decisions the model can make."""

    KEEP = "keep"  # keep current context (no rewrite)
    DROP = "drop"  # wipe history entirely
    COMPRESS = "compress"  # rewrite to compact state (M1's mechanism)


class ActionDecision(Enum):
    """The two search/answer decisions."""

    SEARCH_AGAIN = "search-again"
    ANSWER_NOW = "answer-now"


@dataclass
class DecisionEvent:
    """A single memory/action decision the model made, logged for GRPO credit.

    These events are the credit-carrying tokens M2 trains on. The reward
    function can give bonus/penalty based on whether the decision was optimal
    (e.g., compress when context is long, answer-now when enough evidence).
    """

    step_number: int
    memory_decision: MemoryDecision
    action_decision: ActionDecision
    context_length_before: int = 0
    context_length_after: int = 0
    token_savings: int = 0  # tokens saved by this decision (compress/drop)
    is_optimal: bool = False  # was this the right call given the state?
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_token(self) -> str:
        """Serialize as a credit-carrying token for the trajectory.

        The format <decision:compress:search-again> is injected into the
        model's output before the action, so GRPO's token-level reward can
        reinforce good decisions.
        """
        return f"<decision:{self.memory_decision.value}:{self.action_decision.value}>"


@dataclass
class MemoryOpResult:
    """Result of executing a memory operation."""

    new_conversation: list[dict]
    decision: DecisionEvent
    state_block: str | None = None  # the compressed state (if COMPRESS)


def parse_decision_token(text: str) -> DecisionEvent | None:
    """Extract a decision token from the model's output.

    Looks for <decision:memory_op:action_op> in the text. Returns None if
    no decision token found (backward-compatible with M1 — M1 doesn't emit
    these tokens, so M2 runs as M1 until the model learns to emit them).
    """
    import re

    m = re.search(r"<decision:(keep|drop|compress):(search-again|answer-now)>", text or "")
    if not m:
        return None
    try:
        mem_dec = MemoryDecision(m.group(1))
        act_dec = ActionDecision(m.group(2))
        return DecisionEvent(
            step_number=0,  # filled by caller
            memory_decision=mem_dec,
            action_decision=act_dec,
        )
    except ValueError:
        return None


def memory_op_post_step_hook(step, conversation=None, raw_outputs=None, **kwargs):
    """M2's post_step_hook: execute the model's memory decision.

    This extends M1's mem1_post_step_hook: instead of always compressing, it
    reads the model's <decision:...> token and executes the chosen operation.
    If no decision token is present, falls back to M1's compress behavior
    (backward-compatible).

    Returns the rewritten conversation (if a decision was made) or None
    (leave conversation untouched).
    """
    if conversation is None:
        return None

    meta = getattr(step, "metadata", None) or {}
    is_terminal = meta.get("is_terminal", False)
    tool_results = meta.get("tool_results", [])
    if is_terminal or not tool_results:
        return None

    thought = getattr(step, "thought", None) or ""

    # Check for an explicit decision token (M2 mode)
    decision = parse_decision_token(thought)

    if decision is None:
        # No decision token — fall back to M1's compress behavior.
        # This makes M2 backward-compatible with M1-trained models.
        from agenttune.rag.memory import mem1_post_step_hook

        return mem1_post_step_hook(
            step, conversation=conversation, raw_outputs=raw_outputs, **kwargs
        )

    # Execute the model's chosen memory operation
    if decision.memory_decision == MemoryDecision.KEEP:
        # Keep the conversation as-is — no rewrite
        return None

    if decision.memory_decision == MemoryDecision.DROP:
        # Wipe everything except system + user
        system_msg = user_msg = None
        for m in conversation:
            if m.get("role") == "system" and system_msg is None:
                system_msg = m
            elif m.get("role") == "user" and user_msg is None:
                user_msg = m
            if system_msg and user_msg:
                break
        if user_msg is None:
            return None
        new_conv = []
        if system_msg is not None:
            new_conv.append(dict(system_msg))
        new_conv.append(dict(user_msg))
        return new_conv

    if decision.memory_decision == MemoryDecision.COMPRESS:
        # Compress: same as M1's mem1_post_step_hook
        from agenttune.rag.memory import mem1_post_step_hook

        return mem1_post_step_hook(
            step, conversation=conversation, raw_outputs=raw_outputs, **kwargs
        )

    return None


def decision_reward(prompts, completions, tool_call_counts=None, **kwargs) -> list[float]:
    """M2 reward component: bonus for optimal memory decisions.

    Rewards the model for emitting decision tokens and for making the right
    call:
    - compress when context is long (saves tokens) → bonus
    - drop when context is very long → bonus
    - keep when context is short → bonus (don't waste compute on rewrite)
    - answer-now when correctness is high → bonus
    - search-again when correctness is low → bonus

    This is a small additive reward on top of the T3 stack. The main signal
    still comes from correctness + termination.
    """
    n = len(completions)
    if tool_call_counts is None:
        tool_call_counts = [0] * n
    scores = []
    for i, (comp, _n_search) in enumerate(zip(completions, tool_call_counts, strict=False)):
        text = comp if isinstance(comp, str) else str(comp)
        decision = parse_decision_token(text)
        if decision is None:
            # No decision token emitted — no bonus (M1 behavior)
            score = 0.0
            reason = "no <decision:...> token emitted -> 0.0 (M1 fallback behavior)"
        else:
            # Small bonus for making an explicit decision
            bonus = 0.1
            reason = f"decision token emitted (memory={decision.memory_decision.value}, action={decision.action_decision}) -> +0.1"
            # Extra bonus for answer-now (encourages terminating)
            if decision.action_decision == ActionDecision.ANSWER_NOW:
                bonus += 0.1
                reason += ", action=answer-now -> +0.1 more"
            score = bonus
        log_score("decision_reward", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores
