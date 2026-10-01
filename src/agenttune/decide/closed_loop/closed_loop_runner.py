"""
Closed-Loop Runner — Path B, Week 3
===================================

Wires the Path B control path into one non-blocking cycle:

    examples in buffer
        -> RetrainingTrigger.check_and_fire()      (decide + drain)
        -> BackgroundRetrainer.start(retrain job)  (non-blocking)
        -> [loop keeps running; buffer keeps filling]
        -> on retrain done: DeploymentGate.evaluate_decision()
        -> apply_decision(): deploy or keep old

The runner is deliberately *driven* (you call :meth:`tick` from your loop)
rather than owning its own thread, so it composes with Path A's detection loop
and is trivial to test. The retrain itself runs on a background thread via
``BackgroundRetrainer`` — so ``tick`` never blocks, and new examples added
during a retrain accumulate in the buffer for the next cycle.

Week 3 scope: full wiring + tests with injected (stub) retrain/eval/deploy
callables. The real retrain job is ``retrain_config.run_retrain``; the real
gate inputs come from ``DeploymentGate`` + Path A's ``TrajectoryEvaluator``.
None of those are imported here — they are injected, keeping the runner
GPU-free and unit-testable.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agenttune.decide.closed_loop.behavioral_diversity_monitor import BehavioralDiversityMonitor
from agenttune.decide.closed_loop.contracts import TrainingExample
from agenttune.decide.closed_loop.deployment_gate import GateDecision
from agenttune.decide.closed_loop.retrain_runner import BackgroundRetrainer, RetrainResult
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger

logger = logging.getLogger(__name__)


@dataclass
class CycleRecord:
    """One observable outcome of a :meth:`ClosedLoopRunner.tick`."""

    fired: bool  # did the trigger fire this tick?
    reason: str  # trigger reason / why not
    drained: int = 0  # examples drained into the retrain
    retrain_started: bool = False  # did a background retrain launch?
    diversity_alert: bool = False  # BehavioralDiversityMonitor collapse flag
    trajectory_old: float | None = None  # mean trajectory score before retrain
    trajectory_new: float | None = None  # mean trajectory score after retrain
    gate_decision: str | None = None  # "approved" / "blocked: ..." from gate
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ClosedLoopRunner:
    """Drives trigger -> background retrain -> gate -> deploy, non-blocking.

    Parameters
    ----------
    trigger : RetrainingTrigger
        Owns the buffer + the six conditions + the drop-rate gate.
    retrain_job : Callable[[List[TrainingExample]], Any]
        The work that produces a new model from drained examples. Runs on a
        background thread. Typically wraps ``retrain_config.run_retrain``.
        Its return value is handed to ``on_retrain_done`` (e.g. an output dir).
    on_retrain_done : Optional[Callable[[RetrainResult], Optional[GateDecision]]]
        Called when the background retrain finishes (success or failure). This
        is where the caller runs the eval suite + ``DeploymentGate`` and returns
        a :class:`GateDecision` (or None). The decision is stored on the runner.
    diversity_monitor : Optional[BehavioralDiversityMonitor]
        If provided, ``check_diversity()`` is called automatically after every
        retrain. Feed trajectories via :meth:`observe_trajectory`.
    trajectory_eval_fn : Optional[Callable[[RetrainResult], Tuple]]
        Called in the post-retrain callback (before ``on_retrain_done``) to
        compute trajectory scores. Must return ``(old_traj_results, new_traj_results)``
        — lists of ``AgenticEvalResult`` understood by
        ``DeploymentGate.evaluate_decision``. The result is stored as
        ``last_trajectory_scores`` so ``on_retrain_done`` can read it.
    """

    def __init__(
        self,
        trigger: RetrainingTrigger,
        retrain_job: Callable[[list[TrainingExample]], Any],
        on_retrain_done: Callable[[RetrainResult], GateDecision | None] | None = None,
        diversity_monitor: BehavioralDiversityMonitor | None = None,
        trajectory_eval_fn: Callable[[RetrainResult], tuple] | None = None,
    ) -> None:
        self.trigger = trigger
        self.retrain_job = retrain_job
        self._on_retrain_done = on_retrain_done
        self._diversity_monitor = diversity_monitor
        self._trajectory_eval_fn = trajectory_eval_fn
        self._lock = threading.RLock()
        self._runner = BackgroundRetrainer(trigger=trigger, on_done=self._handle_done)
        self.cycles: list[CycleRecord] = []
        self.last_result: RetrainResult | None = None
        self.last_decision: GateDecision | None = None
        self.last_diversity_alert: bool = False
        self.last_trajectory_scores: tuple[Any | None, Any | None] = (None, None)

    # -- producer side ------------------------------------------------------

    def submit(self, example: TrainingExample) -> None:
        """Add one example to the buffer (Path A producer side).

        Safe to call at ANY time, including while a retrain is running — that
        is exactly the "buffer keeps filling during retrain" guarantee.
        """
        self.trigger.buffer.add(example)

    def observe_trajectory(self, trajectory: dict[str, Any]) -> None:
        """Feed a trajectory into the diversity monitor (no-op if none set)."""
        if self._diversity_monitor is not None:
            self._diversity_monitor.observe_trajectory(trajectory)

    # -- driven cycle -------------------------------------------------------

    def tick(self) -> CycleRecord:
        """Run one control cycle. Non-blocking.

        Checks the trigger; if it fires (and no retrain is already running),
        drains the buffer and launches the retrain in the background, returning
        immediately. If a retrain is in progress, the trigger's own gate makes
        this a no-op so the buffer simply keeps accumulating.
        """
        with self._lock:
            # check_and_fire() already honours the retrain_in_progress gate and
            # the drop-rate guard, and drains atomically on a positive decision.
            fire, reason = self.trigger.should_trigger()
            if not fire:
                rec = CycleRecord(fired=False, reason=reason)
                self.cycles.append(rec)
                return rec

            examples = self.trigger.check_and_fire()
            if not examples:
                # Race: state changed between should_trigger and check_and_fire.
                rec = CycleRecord(fired=False, reason="no_examples_after_check")
                self.cycles.append(rec)
                return rec

            started = self._runner.start(self.retrain_job, examples)
            rec = CycleRecord(
                fired=True,
                reason=reason,
                drained=len(examples),
                retrain_started=started,
            )
            self.cycles.append(rec)
            logger.info(
                "ClosedLoopRunner.tick: fired(%s) drained=%d retrain_started=%s",
                reason,
                len(examples),
                started,
            )
            return rec

    def _handle_done(self, result: RetrainResult) -> None:
        """BackgroundRetrainer callback: retrain finished → diversity → traj eval → gate."""
        self.last_result = result

        # trajectory eval — injected fn, no GPU dep in runner itself
        if self._trajectory_eval_fn is not None:
            try:
                self.last_trajectory_scores = self._trajectory_eval_fn(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("ClosedLoopRunner trajectory_eval_fn failed: %s", exc)
                self.last_trajectory_scores = (None, None)

        # diversity check — post-retrain, per plan
        if self._diversity_monitor is not None:
            alert = self._diversity_monitor.check_diversity()
            self.last_diversity_alert = alert
            if alert:
                logger.warning(
                    "ClosedLoopRunner: BehavioralDiversityMonitor raised collapse alert post-retrain."
                )

        if self._on_retrain_done is not None:
            try:
                self.last_decision = self._on_retrain_done(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("ClosedLoopRunner on_retrain_done failed: %s", exc)
                self.last_decision = None

    # -- Week 4: full async loop --------------------------------------------

    async def run(
        self,
        n_ticks: int,
        detect_fn: Callable[[], Any],
        classify_fn: Callable[[Any], Any],
        generate_fn: Callable[[Any], Any],
        replay_validator: Any | None = None,
        on_tick: Callable[[CycleRecord], None] | None = None,
    ) -> list[CycleRecord]:
        """Week 4 full async loop driver.

        Each tick:
          1. ``detect_fn()``       → list of failures
          2. ``classify_fn(failures)`` → list of ClassifiedFailure (parallel internally)
          3. ``generate_fn(cf)``   → TrainingExample or None, for each CF in parallel
          4. ``replay_validator.validate_batch()`` → filter to valid examples (parallel)
          5. ``observe_trajectory`` + ``submit`` each valid example to buffer
          6. ``tick()``            → trigger check / background retrain launch

        All callables may be sync or async — awaitable return values are
        awaited automatically. The retrain itself runs on a background thread
        (non-blocking), so the loop keeps accumulating examples while it runs.

        Parameters
        ----------
        n_ticks : int
            Number of detect→submit→tick iterations to run.
        detect_fn : Callable[[], List]
            Returns the current batch of new failures (or empty list).
        classify_fn : Callable[[List], List]
            Classifies a batch of failures. May call LLM judges concurrently
            internally.
        generate_fn : Callable[[ClassifiedFailure], Optional[TrainingExample]]
            Generates one training example from one classified failure. Called
            concurrently across all failures in the batch via ``asyncio.gather``.
        replay_validator : Optional[ReplayValidator]
            If provided, ``validate_batch`` is called on generated examples
            before they enter the buffer.
        on_tick : Optional[Callable[[CycleRecord], None]]
            Optional callback after each tick for logging / progress.

        Returns
        -------
        List[CycleRecord]
            One record per tick.
        """

        async def _call(fn: Callable, *args: Any) -> Any:
            result = fn(*args)
            if inspect.isawaitable(result):
                return await result
            return result

        records: list[CycleRecord] = []

        for _ in range(n_ticks):
            # 1. detect
            failures = await _call(detect_fn)

            if not failures:
                rec = self.tick()
                records.append(rec)
                if on_tick:
                    on_tick(rec)
                continue

            # 2. classify (may be parallel internally)
            classified = await _call(classify_fn, failures)
            if not classified:
                rec = self.tick()
                records.append(rec)
                if on_tick:
                    on_tick(rec)
                continue

            # 3. generate in parallel
            gen_results = await asyncio.gather(
                *[_call(generate_fn, cf) for cf in classified],
            )
            examples = [e for e in gen_results if e is not None]

            # 4. replay-validate in parallel
            if examples and replay_validator is not None:
                completion_texts = [
                    str(e.chosen or e.completions or e.prompt or "") for e in examples
                ]
                validated = await replay_validator.validate_batch(examples, completion_texts)
                examples = [e for e, (ok, _) in zip(examples, validated, strict=False) if ok]

            # 5. observe + submit
            for ex in examples:
                traj = {"tool_calls": [s.get("content", "") for s in (ex.prompt or [])]}
                self.observe_trajectory(traj)
                self.submit(ex)

            # 6. tick
            rec = self.tick()
            records.append(rec)
            if on_tick:
                on_tick(rec)

        return records

    # -- status / sync ------------------------------------------------------

    def is_retraining(self) -> bool:
        return self._runner.is_running()

    def wait_for_retrain(self, timeout: float | None = None) -> RetrainResult | None:
        """Block until the in-flight retrain finishes (tests / sync callers)."""
        return self._runner.wait(timeout=timeout)
