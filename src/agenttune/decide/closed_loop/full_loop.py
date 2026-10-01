"""
Full Closed Loop — Path B, Week 4 (the wiring)
==============================================

W4 connects every piece into one continuously running, non-blocking loop:

    Production audit log (continuous)
      │
      ▼  Path A (detect & learn)
    FailureDetector.scan  →  FailureClassifier.classify_batch (parallel)
                          →  TrainingExampleGenerator.generate_batch (parallel,
                             with ReplayValidator)  →  TrainingExample(s)
      │
      ▼  shared thread-safe TrainingBuffer
    ClosedLoopRunner.submit(example)        ← buffer keeps filling
      │
      ▼  Path B (decide & act)
    ClosedLoopRunner.tick()  →  RetrainingTrigger (6 conditions + 2 gates)
      │ fires
      ▼  BackgroundRetrainer (daemon thread; loop keeps running)
    retrain_job(examples)  →  RetrainResult
      │ on done
      ▼
    BehavioralDiversityMonitor.check  +  TrajectoryEvaluator.evaluate_batch
      │
      ▼
    DeploymentGate.evaluate_decision (task A/B + trajectory mean)
      │
      ▼
    DeploymentGate.apply_decision  →  deploy  OR  keep old

This module owns ONLY the orchestration. Every model-dependent boundary — the
retrain job, the old/new model verdict runners, and (optionally) the trajectory
collection — is an injected callable. That keeps the loop:

- **GPU-free and unit-testable** (inject stubs; see ``tests/.../test_w4_full_loop.py``), and
- **LLM- and API-compatible** at both ends: Path A's classifier / generator /
  evaluator run through ``litellm`` (works with hosted APIs like ``groq/...`` /
  ``openai/...`` AND a local OpenAI-compatible server via ``api_base``); the
  retrain writes a local LoRA adapter; deployment supports the
  ``transformers`` / ``vllm`` / ``api`` backends of the Decide bridge.

The runner is *driven* — you call :meth:`ingest_once` (Path A) and :meth:`tick`
(Path B) from your own loop, or use :meth:`run_until` / :meth:`run_forever` for
a turnkey daemon. Driving it yourself keeps it composable and trivial to test.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agenttune.decide.closed_loop.behavioral_diversity_monitor import (
    BehavioralDiversityMonitor,
)
from agenttune.decide.closed_loop.closed_loop_runner import ClosedLoopRunner, CycleRecord
from agenttune.decide.closed_loop.contracts import AgenticEvalResult, TrainingExample
from agenttune.decide.closed_loop.deployment_gate import DeploymentGate, GateDecision
from agenttune.decide.closed_loop.failure_classifier import FailureClassifier
from agenttune.decide.closed_loop.failure_detector import FailureDetector
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig
from agenttune.decide.closed_loop.training_example_generator import (
    TrainingExampleGenerator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases for the injected boundaries
# ---------------------------------------------------------------------------

# Produces a new model artifact from drained examples; returns a dict that MUST
# carry a "path" key pointing at the trained adapter/checkpoint.
RetrainJob = Callable[[list[TrainingExample]], dict[str, Any]]

# input_text -> verdict (string). Used for task-accuracy A/B scoring in the gate.
ModelRunner = Callable[[Any], Any]

# Given a model path/id, build a ModelRunner for it. Called once for the freshly
# trained model after a retrain finishes.
BuildModelRunner = Callable[[str], ModelRunner]

# Optional: given a ModelRunner, produce trajectories (list of dicts) to feed the
# TrajectoryEvaluator and the diversity monitor. None -> skip the trajectory
# signal (the gate then uses task accuracy only).
CollectTrajectories = Callable[[ModelRunner], list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PathAConfig:
    """Settings for the Path A signal path (detect → classify → generate)."""

    audit_log_path: str
    # Models for the LLM-backed Path A components. litellm-style names; an
    # api_base makes them point at any OpenAI-compatible server (local or hosted).
    classifier_model: str = "gpt-4o-mini"
    generator_model: str = "groq/llama-3.3-70b-versatile"
    api_base: str | None = None
    # FailureDetector knobs.
    judge_threshold: float = 0.6
    max_revisits: int = 3
    # ReplayValidator command (validates a corrected completion off-process).
    validation_script: str = "python -c 'import sys; sys.exit(0)'"
    batch_size: int = 10


@dataclass
class GateConfig:
    """Settings for the deployment decision + applying it."""

    config_path: str = "config.yaml"
    backend: str = "transformers"  # transformers | vllm | api
    task_regression_tol: float = 0.0
    trajectory_regression_tol: float = 0.05
    min_test_samples: int = 1


@dataclass
class LoopCycle:
    """One observable retrain→decide outcome (recorded on the loop)."""

    tick: CycleRecord
    retrain_success: bool | None = None
    decision: GateDecision | None = None
    applied: str | None = None  # "deployed" | "kept_old" | None
    diversity_collapsed: bool | None = None
    trajectory_mean_new: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ---------------------------------------------------------------------------
# The full loop
# ---------------------------------------------------------------------------


class FullClosedLoop:
    """End-to-end Path A ⇄ Path B closed loop (W4).

    Parameters
    ----------
    path_a : PathAConfig
        Where production failures come from and which models score/synthesize.
    retrain_job : RetrainJob
        Produces a new model from drained examples (returns ``{"path": ...}``).
        Real: a thin wrapper over ``retrain_config.run_retrain``. Tests: a stub.
    build_model_runner : BuildModelRunner
        Builds an ``input_text -> verdict`` runner for a model path. Called for
        the freshly trained model; the *old* runner is supplied separately
        (``old_model_runner``) since it is the currently-deployed model.
    old_model_runner : ModelRunner
        Verdict runner for the currently-deployed model (the A/B baseline).
    trigger / gate :
        Optional pre-built Path B components; sensible defaults otherwise.
    gate_cfg : GateConfig
        Deployment thresholds + bridge target.
    collect_trajectories : Optional[CollectTrajectories]
        Optional hook to gather trajectories for the trajectory-quality gate
        and the diversity monitor. Omit to gate on task accuracy alone.
    deploy_fn / rollback_fn : Optional callables
        Injectable deployment bridge (defaults to the Decide
        ``model_deployment`` bridge, adapting its ``trained_model_path`` kwarg).
    diversity_monitor : Optional[BehavioralDiversityMonitor]
        Post-retrain behavioral-collapse detector.
    """

    def __init__(
        self,
        path_a: PathAConfig,
        retrain_job: RetrainJob,
        build_model_runner: BuildModelRunner,
        old_model_runner: ModelRunner,
        trigger: RetrainingTrigger | None = None,
        gate: DeploymentGate | None = None,
        gate_cfg: GateConfig | None = None,
        collect_trajectories: CollectTrajectories | None = None,
        deploy_fn: Callable[..., Any] | None = None,
        rollback_fn: Callable[..., Any] | None = None,
        diversity_monitor: BehavioralDiversityMonitor | None = None,
    ) -> None:
        self.path_a = path_a
        self.gate_cfg = gate_cfg or GateConfig()
        self._build_model_runner = build_model_runner
        self._old_model_runner = old_model_runner
        self._collect_trajectories = collect_trajectories
        self._deploy_fn = deploy_fn
        self._rollback_fn = rollback_fn

        # --- Path A components (detect → classify → generate) ---
        self.detector = FailureDetector(
            judge_threshold=path_a.judge_threshold,
            max_revisits=path_a.max_revisits,
        )
        self.classifier = FailureClassifier(
            model_name=path_a.classifier_model, api_base=path_a.api_base
        )
        self.validator = ReplayValidator(validation_script=path_a.validation_script)
        self.generator = TrainingExampleGenerator(
            validator=self.validator,
            model_name=path_a.generator_model,
            api_base=path_a.api_base,
        )
        self.evaluator = self.generator.evaluator  # shared TrajectoryEvaluator
        self.diversity = diversity_monitor or BehavioralDiversityMonitor()

        # --- Path B components (buffer → trigger → retrain → gate) ---
        self.trigger = trigger or RetrainingTrigger(TriggerConfig())
        self.gate = gate or DeploymentGate(seed=0)
        self.retrain_job = retrain_job

        # Build the test set once from successful past runs (the gate yardstick).
        self.test_set: list[dict[str, Any]] = self.gate.build_test_set(
            path_a.audit_log_path, min_samples=self.gate_cfg.min_test_samples
        )
        if not self.test_set:
            logger.warning(
                "FullClosedLoop: empty gate test set from %s — the gate will "
                "approve by default until enough successful runs accumulate.",
                path_a.audit_log_path,
            )

        # The driven Path B runner; on retrain completion it calls our gate.
        self.runner = ClosedLoopRunner(
            self.trigger,
            retrain_job=self.retrain_job,
            on_retrain_done=self._on_retrain_done,
        )

        self.cycles: list[LoopCycle] = []
        self._last_tick: CycleRecord | None = None
        self._lock = threading.RLock()

    # -- Path A: detect → classify → generate → buffer ----------------------

    async def ingest_once(self) -> int:
        """Run one Path A pass over new audit-log lines and buffer the result.

        Detects new failures since the last offset, classifies them in parallel,
        generates training examples in parallel (each validated by the replay
        validator), pushes accepted examples into the shared buffer, and feeds
        any ``episode_reward`` to the drift tracker. Returns the number of
        examples submitted.

        Safe to call while a retrain runs — that is the "buffer keeps filling"
        guarantee; submission goes straight to the thread-safe buffer.
        """
        failures = list(self.detector.scan_audit_log(self.path_a.audit_log_path))
        if not failures:
            return 0

        submitted = 0
        for i in range(0, len(failures), self.path_a.batch_size):
            batch = failures[i : i + self.path_a.batch_size]
            classified = await self.classifier.classify_batch(batch)
            examples = await self.generator.generate_batch(classified)
            for ex in examples:
                self.runner.submit(ex)
                submitted += 1

        logger.info("FullClosedLoop.ingest_once: submitted %d example(s).", submitted)
        return submitted

    def push_reward(self, reward: float) -> None:
        """Forward an episode reward into the drift tracker (T4 signal)."""
        self.trigger.push_reward(reward)

    # -- Path B: one driven control cycle -----------------------------------

    def tick(self) -> CycleRecord:
        """Run one non-blocking Path B control cycle (delegates to the runner).

        Fires the trigger + launches the background retrain when conditions are
        met; otherwise a no-op. The gate runs automatically on retrain
        completion via :meth:`_on_retrain_done`.
        """
        rec = self.runner.tick()
        self._last_tick = rec
        return rec

    # -- retrain completion: diversity + eval + gate + deploy ---------------

    def _on_retrain_done(self, result) -> GateDecision | None:
        """BackgroundRetrainer callback: score the new model and decide."""
        cycle = LoopCycle(tick=self._last_tick, retrain_success=result.success)

        if not result.success:
            logger.error("FullClosedLoop: retrain failed: %s", result.error)
            with self._lock:
                self.cycles.append(cycle)
            return None

        trained_path = (result.result or {}).get("path")
        if not trained_path:
            logger.error("FullClosedLoop: retrain result missing 'path'; cannot gate.")
            with self._lock:
                self.cycles.append(cycle)
            return None

        # Build a verdict runner for the freshly trained model.
        new_runner = self._build_model_runner(trained_path)

        # --- optional trajectory signal: diversity + trajectory-quality gate ---
        old_traj_results: list[AgenticEvalResult] | None = None
        new_traj_results: list[AgenticEvalResult] | None = None
        if self._collect_trajectories is not None:
            try:
                old_trajs = self._collect_trajectories(self._old_model_runner)
                new_trajs = self._collect_trajectories(new_runner)
                old_traj_results = self._evaluate(old_trajs)
                new_traj_results = self._evaluate(new_trajs)
                # Behavioral diversity on the NEW model's trajectories.
                for t in new_trajs:
                    self.diversity.observe_trajectory(t)
                cycle.diversity_collapsed = self.diversity.check_diversity()
                cycle.trajectory_mean_new = DeploymentGate._mean_trajectory_score(new_traj_results)
            except Exception as exc:  # noqa: BLE001 — trajectory signal is best-effort
                logger.warning("FullClosedLoop: trajectory eval skipped: %s", exc)

        # --- the gate: task accuracy AND trajectory quality ---
        decision = self.gate.evaluate_decision(
            self.test_set,
            old_model_fn=self._old_model_runner,
            new_model_fn=new_runner,
            old_trajectory_results=old_traj_results,
            new_trajectory_results=new_traj_results,
            task_regression_tol=self.gate_cfg.task_regression_tol,
            trajectory_regression_tol=self.gate_cfg.trajectory_regression_tol,
        )
        cycle.decision = decision

        # --- apply: deploy the new model or keep the old one ---
        applied = self.gate.apply_decision(
            decision,
            config_path=self.gate_cfg.config_path,
            trained_path=trained_path,
            deploy_fn=self._resolve_deploy_fn(),
            rollback_fn=self._rollback_fn,
        )
        cycle.applied = applied

        # On a real deploy, the new model becomes the baseline for next time.
        if applied == "deployed":
            self._old_model_runner = new_runner

        with self._lock:
            self.cycles.append(cycle)
        return decision

    def _evaluate(self, trajectories: list[dict[str, Any]]) -> list[AgenticEvalResult]:
        """Run the trajectory evaluator (handles sync or running event loop)."""
        if not trajectories:
            return []
        coro = self.evaluator.evaluate_batch(trajectories)
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Already inside an event loop (e.g. driven from async code) — run
            # the coroutine on a dedicated loop in a worker thread.
            out: dict[str, Any] = {}

            def _runner() -> None:
                out["r"] = asyncio.new_event_loop().run_until_complete(coro)

            th = threading.Thread(target=_runner)
            th.start()
            th.join()
            return out.get("r", [])

    def _resolve_deploy_fn(self) -> Callable[..., Any]:
        """Return a deploy fn that matches ``apply_decision``'s call signature.

        ``apply_decision`` calls ``deploy_fn(config_path=..., trained_path=...)``,
        but the real Decide bridge expects ``trained_model_path`` + ``backend``.
        When no ``deploy_fn`` is injected, adapt the bridge here.
        """
        if self._deploy_fn is not None:
            return self._deploy_fn

        backend = self.gate_cfg.backend

        def _bridge_deploy(config_path: str, trained_path: str) -> None:
            from agenttune.decide.model_deployment import deploy_trained_model

            deploy_trained_model(
                trained_model_path=trained_path,
                config_path=config_path,
                backend=backend,
            )

        return _bridge_deploy

    # -- turnkey driving loops ----------------------------------------------

    async def run_until(
        self,
        stop: Callable[[], bool],
        poll_interval_s: float = 1.0,
        wait_for_retrain: bool = True,
    ) -> list[LoopCycle]:
        """Drive the loop until ``stop()`` returns True.

        Each iteration: Path A ``ingest_once`` → Path B ``tick`` → sleep. The
        retrain runs in the background; the buffer keeps filling throughout. If
        ``wait_for_retrain`` is set, blocks for the final in-flight retrain
        before returning so callers see the last decision.
        """
        while not stop():
            await self.ingest_once()
            self.tick()
            await asyncio.sleep(poll_interval_s)
        if wait_for_retrain and self.runner.is_retraining():
            self.runner.wait_for_retrain(timeout=None)
        return self.cycles

    async def run_forever(self, poll_interval_s: float = 1.0) -> None:
        """Drive the loop indefinitely (production daemon)."""
        await self.run_until(
            stop=lambda: False, poll_interval_s=poll_interval_s, wait_for_retrain=False
        )

    # -- status -------------------------------------------------------------

    def wait_for_retrain(self, timeout: float | None = None):
        return self.runner.wait_for_retrain(timeout=timeout)

    @property
    def last_decision(self) -> GateDecision | None:
        return self.runner.last_decision
