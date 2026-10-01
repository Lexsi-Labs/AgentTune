"""
Retraining Trigger — Path B, Week 1
===================================

Decides *when* to launch a retrain. This module owns three pieces:

- ``TrainingBuffer``      — thread-safe buffer of ``TrainingExample`` objects
                            produced by Path A's ``TrainingExampleGenerator``.
- ``RewardDriftTracker``  — rolling-window episode-reward statistics used to
                            detect when the deployed model is degrading.
- ``RetrainingTrigger``   — evaluates six trigger conditions plus the
                            ``retrain_in_progress`` gate and drains the buffer
                            when a retrain should fire.

Design notes
------------
- The buffer is the single shared mutable state between Path A (producer,
  ``add``) and Path B (consumer, ``drain``).  All access is guarded by one
  reentrant lock; see ``docs``/async_contract for the full contract.
- None of the six triggers issue an LLM call.  They are pure arithmetic /
  set / clock operations so the trigger layer never blocks on network I/O.
- Week 2 scope: TWO active gates now block a retrain — ``retrain_in_progress``
  and the ``drop_rate`` buffer-health guard (skip a retrain on a mostly-bad
  buffer, with a loud warning).  Both are checked before any trigger fires.

Inspirations (see research_and_context):
- SLIME ``RolloutBuffer``     — RLock + Condition producer/consumer pattern.
- Agent Lightning ``Event``   — cooperative ``retrain_in_progress`` flag.
- Agent Lightning store cap   — bounded buffer via FIFO eviction.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean, pstdev

from agenttune.decide.closed_loop.contracts import BufferHealth, TrainingExample

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TriggerConfig:
    """Thresholds for the six trigger conditions and the gates.

    All values are conservative defaults; tune empirically once the loop runs.
    """

    # --- Required triggers (BUILD_PLAN Week 1) ---
    total_failures_threshold: int = 50  # T1: fire when buffer reaches this size
    dominance_ratio: float = 0.70  # T2: one root_cause >= 70% of buffer
    min_examples_ready: int = 30  # T3: accepted-example count to fire

    # --- Production triggers (approved extras) ---
    reward_drift_window: int = 50  # T4: rolling window size for drift
    reward_drift_baseline_size: int = 50  # T4: episodes used to lock the baseline
    reward_drift_sigma: float = 1.5  # T4: fire if mean drops k*std below baseline
    novel_type_min_count: int = 3  # T5: instances of a new type before firing
    staleness_window_hours: float = 6.0  # T6: hours since last retrain to fire
    min_stale_buffer_size: int = 10  # T6: minimum buffer size for staleness

    # --- Gates ---
    max_drop_rate: float = 0.60  # Week 2 gate: skip retrain if drop_rate exceeds this
    min_attempts_for_drop_gate: int = (
        10  # only apply the drop-rate gate once we have this many attempts
    )
    max_buffer_size: int = 500  # FIFO eviction cap


# ---------------------------------------------------------------------------
# Reward drift tracker (T4 support) — zero LLM calls
# ---------------------------------------------------------------------------


class RewardDriftTracker:
    """Tracks rolling episode-reward statistics to detect model degradation.

    No LLM call: reads ``episode_reward`` values that ``AuditWriter`` already
    writes to the audit log.  A baseline mean/std is locked once enough
    episodes have been observed; subsequent windows are compared against it.

    Drift is flagged when the current rolling mean falls more than
    ``sigma * baseline_std`` below the baseline mean — i.e. the model is doing
    measurably worse than it was at baseline.
    """

    def __init__(
        self,
        window_size: int = 50,
        baseline_size: int = 50,
    ) -> None:
        self._window_size = window_size
        self._baseline_size = baseline_size
        self._window: deque[float] = deque(maxlen=window_size)
        self._baseline_samples: list[float] = []
        self._baseline_mean: float | None = None
        self._baseline_std: float | None = None
        self._lock = threading.RLock()

    @property
    def baseline_locked(self) -> bool:
        return self._baseline_mean is not None

    def push(self, reward: float) -> None:
        """Record one episode reward."""
        with self._lock:
            value = float(reward)
            self._window.append(value)
            # Accumulate baseline until we have enough, then lock it.
            if self._baseline_mean is None:
                self._baseline_samples.append(value)
                if len(self._baseline_samples) >= self._baseline_size:
                    self._baseline_mean = mean(self._baseline_samples)
                    # pstdev of a constant series is 0.0 — guarded in is_drifting.
                    self._baseline_std = pstdev(self._baseline_samples)
                    logger.info(
                        "RewardDriftTracker baseline locked: mean=%.4f std=%.4f (n=%d)",
                        self._baseline_mean,
                        self._baseline_std,
                        len(self._baseline_samples),
                    )

    def current_mean(self) -> float | None:
        with self._lock:
            if not self._window:
                return None
            return mean(self._window)

    def is_drifting(self, sigma: float = 1.5) -> bool:
        """True if the current rolling mean is sigma*std below baseline.

        Returns False until the baseline is locked and the rolling window has
        data.  A zero-variance baseline falls back to a strict ``<`` comparison
        so a clear downward shift is still caught.
        """
        with self._lock:
            if self._baseline_mean is None or not self._window:
                return False
            cur = mean(self._window)
            std = self._baseline_std or 0.0
            if std <= 0.0:
                # No baseline variance — any drop below baseline counts.
                return cur < self._baseline_mean
            return cur < (self._baseline_mean - sigma * std)

    def load_from_audit(self, audit_path: str) -> int:
        """Bootstrap the tracker from an existing audit log.

        Reads completion entries (those carrying ``episode_reward``) in order
        and pushes each non-null reward.  Returns the number of rewards loaded.
        """
        if not os.path.exists(audit_path):
            logger.warning("RewardDriftTracker.load_from_audit: %s not found", audit_path)
            return 0

        loaded = 0
        with open(audit_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reward = record.get("episode_reward")
                if reward is not None:
                    self.push(float(reward))
                    loaded += 1
        logger.info("RewardDriftTracker loaded %d rewards from %s", loaded, audit_path)
        return loaded


# ---------------------------------------------------------------------------
# Training buffer — the Path A <-> Path B handshake
# ---------------------------------------------------------------------------


class TrainingBuffer:
    """Thread-safe buffer of ``TrainingExample`` objects.

    Producer (Path A) calls :meth:`add`; consumer (Path B) calls :meth:`drain`.
    Reads (:meth:`health`, ``len``, :meth:`novel_causes`) are also lock-guarded.

    A single reentrant lock protects all state — simple and correct (the
    Agent Lightning ``LightningStoreThreaded`` approach).  ``RLock`` (not
    ``Lock``) because :meth:`health` calls :meth:`__len__` while already
    holding the lock.

    Bounded by ``max_size`` with FIFO eviction: when full, the oldest example
    is dropped on ``add`` (a ``deque(maxlen=...)``).  Stale failures from before
    the last retrain are the least relevant, so dropping oldest-first is safe.
    """

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._examples: deque[TrainingExample] = deque(maxlen=max_size)
        # root_cause -> count currently in buffer (kept in sync with _examples)
        self._type_counts: dict[str, int] = {}
        # root_causes ever observed across all drains (for novelty detection)
        self._seen_root_causes: set[str] = set()
        # lifetime counters (BufferHealth "report card")
        self._total_attempted = 0  # every example offered to the buffer
        self._total_accepted = 0  # examples that were generated successfully
        self._total_evicted = 0  # examples dropped by FIFO cap
        self._last_drain_ts: str | None = None

    # -- mutation -----------------------------------------------------------

    def add(self, example: TrainingExample) -> None:
        """Append an example (thread-safe). Evicts oldest if at capacity."""
        with self._lock:
            self._total_attempted += 1
            # An example with no chosen AND no rejected carries no signal.
            is_accepted = bool(example.chosen) or bool(example.rejected)
            if is_accepted:
                self._total_accepted += 1

            # Detect FIFO eviction: deque at maxlen drops the left item on append.
            if self.max_size and len(self._examples) >= self.max_size:
                evicted = self._examples[0]
                self._decr_type(evicted.root_cause)
                self._total_evicted += 1

            self._examples.append(example)
            self._type_counts[example.root_cause] = self._type_counts.get(example.root_cause, 0) + 1
            self._not_empty.notify_all()

    def drain(self) -> list[TrainingExample]:
        """Atomically remove and return all buffered examples.

        Updates ``_seen_root_causes`` with whatever was in the buffer so that
        the next cycle can tell genuinely new failure types apart.
        """
        with self._lock:
            drained = list(self._examples)
            self._examples.clear()
            self._type_counts.clear()
            self._seen_root_causes.update(ex.root_cause for ex in drained)
            self._last_drain_ts = _utcnow_iso()
            return drained

    def _decr_type(self, root_cause: str) -> None:
        count = self._type_counts.get(root_cause, 0)
        if count <= 1:
            self._type_counts.pop(root_cause, None)
        else:
            self._type_counts[root_cause] = count - 1

    # -- reads --------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._examples)

    def current_root_causes(self) -> set[str]:
        """Distinct root_causes currently in the buffer."""
        with self._lock:
            return set(self._type_counts.keys())

    def novel_causes(self) -> set[str]:
        """root_causes in the buffer that have never been drained before."""
        with self._lock:
            return set(self._type_counts.keys()) - self._seen_root_causes

    def type_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._type_counts)

    def health(self) -> BufferHealth:
        """Snapshot of buffer statistics (the Week 2 "report card")."""
        with self._lock:
            size = len(self._examples)
            drop_rate = (
                (self._total_attempted - self._total_accepted) / self._total_attempted
                if self._total_attempted > 0
                else 0.0
            )
            dominant_type: str | None = None
            dominant_ratio = 0.0
            if size > 0 and self._type_counts:
                dominant_type = max(self._type_counts, key=self._type_counts.get)
                dominant_ratio = self._type_counts[dominant_type] / size

            return BufferHealth(
                total_failures_detected=self._total_attempted,
                total_examples_attempted=self._total_attempted,
                total_examples_accepted=self._total_accepted,
                drop_rate=round(drop_rate, 4),
                buffer_size=size,
                dominant_failure_type=dominant_type,
                dominant_failure_ratio=round(dominant_ratio, 4),
                last_drain_timestamp=self._last_drain_ts,
                total_known_types=len(self._seen_root_causes | set(self._type_counts)),
                novel_failure_types=sorted(set(self._type_counts.keys()) - self._seen_root_causes),
            )


# ---------------------------------------------------------------------------
# Retraining trigger
# ---------------------------------------------------------------------------


class RetrainingTrigger:
    """Decides when to launch a retrain and drains the buffer when it fires.

    Six trigger conditions (any one fires):
        T1 total_failures   — buffer reached ``total_failures_threshold``
        T2 dominance        — one root_cause is >= ``dominance_ratio`` of buffer
        T3 enough_examples  — accepted-example count >= ``min_examples_ready``
        T4 reward_drift     — model degrading vs its own baseline
        T5 novel_failure    — a never-seen failure type emerged (>= min_count)
        T6 staleness        — too long since last retrain, with a minimum buffer

    Gates (block a fire):
        - ``retrain_in_progress`` set  (ACTIVE)
        - ``drop_rate`` over ``max_drop_rate``  (ACTIVE, Week 2 buffer-health guard)

    The ``retrain_in_progress`` flag is a ``threading.Event`` so a future
    background loop can ``wait`` on it rather than busy-poll.
    """

    def __init__(
        self,
        config: TriggerConfig | None = None,
        buffer: TrainingBuffer | None = None,
        reward_tracker: RewardDriftTracker | None = None,
    ) -> None:
        self.config = config or TriggerConfig()
        self.buffer = buffer or TrainingBuffer(max_size=self.config.max_buffer_size)
        self.reward_tracker = reward_tracker or RewardDriftTracker(
            window_size=self.config.reward_drift_window,
            baseline_size=self.config.reward_drift_baseline_size,
        )
        self.retrain_in_progress = threading.Event()  # SET = running, CLEAR = idle
        self._last_retrain_time: datetime | None = None
        self._cycle_count = 0
        self._decision_lock = threading.RLock()

    # -- concurrency flag ---------------------------------------------------

    def mark_retrain_started(self) -> None:
        self.retrain_in_progress.set()
        logger.info("retrain_in_progress -> SET")

    def mark_retrain_finished(self) -> None:
        self.retrain_in_progress.clear()
        self._last_retrain_time = datetime.now(UTC)
        logger.info("retrain_in_progress -> CLEAR (last_retrain=%s)", self._last_retrain_time)

    def is_retrain_running(self) -> bool:
        return self.retrain_in_progress.is_set()

    # -- reward feed --------------------------------------------------------

    def push_reward(self, reward: float) -> None:
        """Forward an episode reward to the drift tracker."""
        self.reward_tracker.push(reward)

    # -- decision -----------------------------------------------------------

    def should_trigger(self) -> tuple[bool, str]:
        """Evaluate gates + six triggers. Returns ``(fire?, reason)``.

        Pure read of buffer/tracker state — no mutation, no LLM, no I/O.

        Order:
          1. ``retrain_in_progress`` gate — never fire while a retrain runs.
          2. Trigger conditions — does anything *want* to fire?
          3. ``drop_rate`` buffer-health guard — if a trigger fired but the
             buffer is mostly bad, veto it with a loud warning (never train on
             a mostly-bad buffer).
        """
        with self._decision_lock:
            self._cycle_count += 1

            # ----- GATE 1: retrain already running -----
            if self.retrain_in_progress.is_set():
                return False, "retrain_already_running"

            # ----- Evaluate the six trigger conditions -----
            fire, reason = self._evaluate_triggers()
            if not fire:
                return False, reason

            # ----- GATE 2: drop-rate buffer-health guard -----
            blocked, drop_reason = self._drop_rate_blocks()
            if blocked:
                return False, drop_reason

            return True, reason

    def _evaluate_triggers(self) -> tuple[bool, str]:
        """Pure trigger evaluation (no gates). Returns ``(fire?, reason)``."""
        cfg = self.config
        size = len(self.buffer)
        health = self.buffer.health()

        # ----- T1: total failure volume -----
        if size >= cfg.total_failures_threshold:
            return True, "total_failures_exceeded"

        # ----- T2: one failure type dominates -----
        if size > 0:
            counts = self.buffer.type_counts()
            for type_name, count in counts.items():
                if count / size >= cfg.dominance_ratio:
                    return True, f"dominance:{type_name}"

        # ----- T3: enough accepted examples -----
        if health.total_examples_accepted >= cfg.min_examples_ready:
            return True, "enough_examples"

        # ----- T4: reward drift -----
        if self.reward_tracker.is_drifting(sigma=cfg.reward_drift_sigma):
            return True, "reward_drift"

        # ----- T5: novel failure type emerged -----
        novel = self.buffer.novel_causes()
        if novel:
            counts = self.buffer.type_counts()
            ready = sorted(t for t in novel if counts.get(t, 0) >= cfg.novel_type_min_count)
            if ready:
                return True, f"novel_failure:{ready[0]}"

        # ----- T6: staleness -----
        if self._last_retrain_time is not None and size >= cfg.min_stale_buffer_size:
            hours_since = (datetime.now(UTC) - self._last_retrain_time).total_seconds() / 3600.0
            if hours_since >= cfg.staleness_window_hours:
                return True, "staleness"

        return False, "below_threshold"

    def _drop_rate_blocks(self) -> tuple[bool, str]:
        """Week 2 buffer-health guard.

        Returns ``(blocked?, reason)``.  Blocks a retrain when the drop rate
        exceeds ``max_drop_rate`` (and we have enough attempts to trust the
        number).  Logs a loud warning — never train on a mostly-bad buffer.
        """
        cfg = self.config
        health = self.buffer.health()
        attempts = health.total_examples_attempted
        if attempts < cfg.min_attempts_for_drop_gate:
            return False, ""  # too few attempts to judge buffer quality
        if health.drop_rate > cfg.max_drop_rate:
            logger.warning(
                "!!! RETRAIN SKIPPED: buffer drop_rate=%.2f exceeds max_drop_rate=%.2f "
                "(%d attempted, %d accepted). Refusing to train on a mostly-bad buffer.",
                health.drop_rate,
                cfg.max_drop_rate,
                attempts,
                health.total_examples_accepted,
            )
            return True, "drop_rate_too_high"
        return False, ""

    def check_and_fire(self) -> list[TrainingExample] | None:
        """Atomically evaluate triggers and drain the buffer if one fires.

        Returns the drained examples when a retrain should start, else ``None``.
        Holding ``_decision_lock`` across the check + drain prevents an
        interleaved decision from a second caller.
        """
        with self._decision_lock:
            fire, reason = self.should_trigger()
            if not fire:
                return None
            examples = self.buffer.drain()
            logger.info(
                "RetrainingTrigger FIRED (reason=%s, drained=%d examples)",
                reason,
                len(examples),
            )
            return examples
