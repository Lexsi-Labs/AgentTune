"""M1 + M2 — MEM1-style rewritten running state + M2 memory decisions.

M1: the MEM1 state-rewrite mechanism as a post_step_hook, plus the tool-output
compressor and a recent-k truncation baseline for comparison. See m1_rewrite.py.

M2: decision-event schema + memory-op wrapper. Extends M1 with explicit
keep/drop/compress decisions the model learns to make via RL. See
m2_decisions.py. Backward-compatible with M1 (falls back to compress when no
decision token is emitted).
"""

from .m1_rewrite import (
    build_rewritten_state,
    compress_tool_output,
    extract_internal_state,
    mem1_post_step_hook,
    recent_k_post_step_hook,
)
from .m2_decisions import (
    ActionDecision,
    DecisionEvent,
    MemoryDecision,
    MemoryOpResult,
    decision_reward,
    memory_op_post_step_hook,
    parse_decision_token,
)

__all__ = [
    "extract_internal_state",
    "compress_tool_output",
    "build_rewritten_state",
    "mem1_post_step_hook",
    "recent_k_post_step_hook",
    "MemoryDecision",
    "ActionDecision",
    "DecisionEvent",
    "MemoryOpResult",
    "memory_op_post_step_hook",
    "parse_decision_token",
    "decision_reward",
]
