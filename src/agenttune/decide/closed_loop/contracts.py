from dataclasses import dataclass, field
from typing import Any


@dataclass
class Failure:
    trajectory_id: str
    failure_type: str
    failed_stage_name: str
    context: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    judge_score: float | None = None


@dataclass
class ClassifiedFailure:
    failure: Failure
    root_cause: str  # One of: wrong_tool, wrong_routing, incomplete_reasoning, hallucinated_output, loop_collapse
    confidence: float
    analysis: str


@dataclass
class TrainingExample:
    """A single training example salvaged from a failure.

    Carries two representations so the contract stays a superset across paths
    (frozen handshake, extended — never narrowed):

    - Preference form (``chosen`` / ``rejected``): a single corrected response
      vs the failed one. This is what Path B's adapter-only retrain consumes
      (DPO / BCO — see ``retrain_config``).
    - Multi-completion form (``completions`` / ``rewards``): N scored
      completions, written by Path A's generator for future GRPO-style use.

    Path A's ``TrainingExampleGenerator`` currently populates only the
    multi-completion form. ``derive_preference_from_completions()`` bridges that
    output into a DPO/BCO-usable pair (best-reward = chosen, worst = rejected)
    so the loop runs end-to-end today. Use ``has_preference_pair()`` to check a
    DPO/BCO-usable example.

    NOTE (interim bridge): the derived ``rejected`` is the lowest-reward
    *correction*, not the original failed action. Once Path A's generator sets
    ``chosen``/``rejected`` directly (rejected = the real failed action from the
    failure context), that takes precedence and this derivation is a fallback.
    """

    trajectory_id: str
    original_failure_type: str
    root_cause: str
    prompt: list[dict[str, str]]  # Standard message format [{"role": "user", "content": ...}]
    completions: list[list[dict[str, str]]] | None = None  # For multi-completion RL (e.g. GRPO)
    rewards: list[float] | None = None  # Associated rewards for each completion
    chosen: list[dict[str, str]] | None = None  # For DPO/BCO (highest reward)
    rejected: list[dict[str, str]] | None = None  # For DPO/BCO (lowest reward)
    salvaged_at_attempt: int = 1
    is_negative_only: bool = (
        False  # If true, it means it's just a negative label (no corrected response)
    )

    def has_preference_pair(self) -> bool:
        """True if both a chosen and rejected response are present (DPO-ready)."""
        return bool(self.chosen) and bool(self.rejected)

    def has_completions(self) -> bool:
        """True if scored completions are present (GRPO form / bridge source)."""
        return bool(self.completions) and bool(self.rewards)

    def derive_preference_from_completions(self) -> bool:
        """Bridge multi-completion output into a chosen/rejected pair.

        Best-reward completion → ``chosen``, worst → ``rejected``. Lets a
        generator that only emits ``completions``/``rewards`` (Path A today)
        feed the DPO/BCO retrain path. No-op (returns False) if a preference
        pair already exists, if there are fewer than 2 completions, or if all
        rewards are equal (no meaningful preference). Returns True if it set the
        pair.
        """
        if self.has_preference_pair():
            return False
        if not self.has_completions() or len(self.completions) < 2:
            return False
        best_i = max(range(len(self.rewards)), key=lambda i: self.rewards[i])
        worst_i = min(range(len(self.rewards)), key=lambda i: self.rewards[i])
        if best_i == worst_i:
            return False
        self.chosen = self.completions[best_i]
        self.rejected = self.completions[worst_i]
        return True


@dataclass
class BufferHealth:
    # --- Path A fields (frozen Day 1, do not change) ---
    total_failures_detected: int = 0
    total_examples_attempted: int = 0
    total_examples_accepted: int = 0
    drop_rate: float = 0.0
    # --- Path B fields (Week 1) — all defaulted, backward-compatible ---
    buffer_size: int = 0  # current items in buffer
    dominant_failure_type: str | None = None  # root_cause with the largest share
    dominant_failure_ratio: float = 0.0  # that share as a fraction of buffer
    last_drain_timestamp: str | None = None  # ISO 8601 of last drain()
    last_retrain_timestamp: str | None = None  # ISO 8601 of last retrain finish
    cycle_count: int = 0  # how many should_trigger() cycles ran
    # Reward drift (no LLM — derived from audit episode_reward)
    reward_mean_current: float | None = None  # current rolling-window mean
    reward_mean_baseline: float | None = None  # locked baseline mean
    reward_drift_detected: bool = False
    # Novelty (distribution shift sensing)
    novel_failure_types: list[str] = field(default_factory=list)  # new types this cycle
    total_known_types: int = 0  # distinct root_causes seen historically


@dataclass
class AgenticEvalResult:
    trajectory_id: str
    goal_completion_score: float
    tool_sequence_validity: float
    unnecessary_steps_penalty: float
    error_recovery_score: float
    overall_judge_score: float
    tac_score: float = 0.0  # Tool Argument Correctness
    ter_score: float = 0.0  # Tool Efficacy Reward
    latency_ms: float = 0.0  # Response time in milliseconds
    bleu_score: float = 0.0  # Unigram overlap score
    bert_score: float = 0.0  # Semantic equivalence score

    # Novel Programmatic Metrics
    pas_score: float | None = None  # Plan Adherence Score
    arr_score: float = 0.0  # API Redundancy Ratio
    scsr_score: float = 0.0  # Self-Correction Success Rate
    rad_score: float = 0.0  # Reasoning-to-Action Density
    pmed_score: int | None = None  # Path Minimum Edit Distance
    ase_score: float | None = None  # Action-State Efficiency
    lcf_score: int = 0  # Loop Collapse Frequency

    # Novel LLM-judged Metrics
    iasa_score: float = 0.0  # Intent-Action Semantic Alignment
    scr_score: float | None = None  # Sub-goal Completion Rate
    egs_score: float = 0.0  # Evidence Grounding Score

    rotation_id: int | None = None  # For prompt rotation tracking
