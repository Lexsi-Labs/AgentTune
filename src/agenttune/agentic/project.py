"""Project — the integration spine's public lifecycle object (Phase 5).

Threads one artifact (tasks on a strategy+harness) through infer/collect/evaluate,
accumulating trajectories (Phase-1 EventLogs) and emitting a typed lifecycle event
stream. train/heal/distill wrap the existing runners in a later phase (6/7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from agenttune.agentic.events import EventKind, EventLog
from agenttune.agentic.harness import Harness
from agenttune.agentic.strategy import AgentStrategy, run_episode


@dataclass
class LifecycleEvent:
    stage: str
    kind: str
    data: dict = field(default_factory=dict)


# ── Frozen return contracts (v1.0). These name the dict keys callers depend on. ──
class AgenticMetrics(TypedDict):
    """The programmatic trajectory metrics, all in [0, 1]."""

    tac: float  # tool-argument correctness
    ter: float  # tool-error rate
    arr: float  # action-repetition rate
    scsr: float  # self-consistency success rate
    rad: float  # redundant-action detection
    lcf: float  # loop-collapse fraction


class EvalReport(TypedDict):
    n: int
    mean_score: float
    scores: list[float]


class AgenticEvalReport(TypedDict):
    n: int
    metrics: AgenticMetrics
    per_trajectory: list[AgenticMetrics]


# The existing TrajectoryEvaluator's metrics computable from a bare trajectory trace
# ({tool_calls, tool_outputs}) — no model, no network, no reference. The reference-
# dependent metrics (pas/pmed/ase need a plan or golden trajectory) and the judge/
# semantic metrics (need litellm) are wired when those inputs are supplied, in a later phase.
_PROGRAMMATIC_METRICS = ("tac", "ter", "arr", "scsr", "rad", "lcf")


def agentic_metrics(log: EventLog, evaluator=None) -> AgenticMetrics:
    """Score an EventLog with the EXISTING ``TrajectoryEvaluator``'s programmatic
    metrics. Wraps — reuses the evaluator's real ``_calculate_*`` logic via
    ``EventLog.to_eval_dict()``; nothing is reimplemented. GPU/network-free."""
    from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

    ev = evaluator or TrajectoryEvaluator(model_name="none", api_base=None)
    d = log.to_eval_dict()
    return {m: getattr(ev, f"_calculate_{m}")(d) for m in _PROGRAMMATIC_METRICS}


def answer_match(log: EventLog, expected) -> float:
    if expected is None:
        return 0.0
    needle = str(expected)
    for e in log:
        if e.kind in (EventKind.TOOL_RESULT, EventKind.TEXT):
            hay = str(e.payload.get("output", e.payload.get("text", "")))
            if needle in hay:
                return 1.0
    return 0.0


class Project:
    """The public lifecycle object: one artifact threaded through
    build → collect → evaluate → train → distill → heal, all via one ``EventLog`` schema.

    All parameters are keyword-only and optional, so ``Project()`` is valid and stages that
    need a piece (e.g. ``infer`` needs ``strategy`` + ``harness``) raise a clear error if it
    is missing.

    Args:
        strategy: the agent design (``AgentStrategy``) ``infer``/``collect`` run.
        harness:  the environment (``Harness``) the strategy acts on.
        workflow: an optional pre-built DECIDE workflow this project wraps (``None`` for the
                  pure agentic path). Reserved for DECIDE integration; unused by the core spine.
        data:     optional free-form project metadata (``dict``). Not interpreted by the spine.
    """

    def __init__(
        self,
        *,
        strategy: AgentStrategy | None = None,
        harness: Harness | None = None,
        workflow=None,
        data=None,
    ):
        self.workflow = workflow
        self.data = data
        self.strategy = strategy
        self.harness = harness
        self._events: list[LifecycleEvent] = []
        self._trajectories: list[EventLog] = []
        self._native_trajectories: list = []  # RL substrate (Trajectory.to_trl_format)

    def _emit(self, stage: str, kind: str, **data) -> None:
        self._events.append(LifecycleEvent(stage, kind, data))

    def events(self) -> list[LifecycleEvent]:
        return list(self._events)

    @property
    def trajectories(self) -> list[EventLog]:
        return list(self._trajectories)

    def add_trajectory(self, log: EventLog) -> None:
        """Register a teacher/rollout trajectory (typically full-tier) for
        training/distillation — e.g. ``add_trajectory(EventLog.from_trajectory(t))``."""
        self._trajectories.append(log)

    def sft_dataset(self, fmt: str = "sft") -> list[dict]:
        """The SFT rows the spine would train/distill on — assembled from full-tier
        trajectories (the same rows train/distill feed the real trainer). GPU-free preview."""
        full = [t for t in self._trajectories if getattr(t, "tier", None) == "full"]
        return [row for t in full for row in t.as_dataset_rows(fmt)]

    @property
    def native_trajectories(self) -> list:
        """The native ``agentic.Trajectory`` objects from rollouts — the RL substrate
        (``to_trl_format``). Full-tier ``EventLog`` projections live in ``trajectories``."""
        return list(self._native_trajectories)

    def collect_rollout(
        self,
        engine,
        tasks,
        *,
        tools=None,
        max_steps: int = 8,
        reward_fn=None,
        system_prompt=None,
        **gen_kwargs,
    ) -> list[EventLog]:
        """Phase 2b — drive a real ``RolloutEngine`` over ``tasks`` to produce FULL-tier
        trajectories (carrying logprobs). Wraps the existing ``create_rollout_fn`` (the same
        producer GRPO uses): one call yields both the native ``Trajectory`` objects (retained
        as the RL substrate) and the GRPO on-policy batch. Each native trajectory is projected
        to a full-tier ``EventLog`` (feeding SFT/distill/eval). Emits lifecycle events.

        This is the gate that makes ``train(fmt='sft')`` and distillation real off rollouts."""
        from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

        collected: list = []
        fn = create_rollout_fn(
            rollout_engine=engine,
            tools=tools or [],
            max_steps=max_steps,
            reward_fn=reward_fn,
            system_prompt=system_prompt,
            on_trajectory_end=collected.append,
        )
        self._emit("collect_rollout", "started", n=len(list(tasks)))
        fn(list(tasks))  # also returns the GRPO batch
        logs: list[EventLog] = []
        for traj in collected:
            self._native_trajectories.append(traj)
            log = EventLog.from_trajectory(traj)
            self._trajectories.append(log)
            logs.append(log)
        self._emit("collect_rollout", "done", n=len(logs))
        return logs

    def infer(self, task: str) -> EventLog:
        """Run one episode of ``strategy`` on ``harness`` for ``task`` and return its
        (light-tier) ``EventLog``. Requires both ``strategy`` and ``harness`` to be set."""
        if self.strategy is None or self.harness is None:
            raise ValueError("Project.infer requires both `strategy` and `harness`.")
        self._emit("infer", "started", task=task)
        log = run_episode(self.strategy, self.harness, task)
        self._trajectories.append(log)
        self._emit("infer", "episode", trajectory_id=log.id, n_events=len(log))
        return log

    def collect(self, tasks: list[str]) -> list[EventLog]:
        """Run ``infer`` over many ``tasks`` (light-tier episodes, no model). For trainable
        full-tier trajectories use ``collect_rollout`` with a ``RolloutEngine`` instead."""
        self._emit("collect", "started", n=len(tasks))
        logs = [self.infer(t) for t in tasks]
        self._emit("collect", "done", n=len(logs))
        return logs

    def evaluate(self, dataset: list[dict], scorer=answer_match) -> EvalReport:
        self._emit("evaluate", "started", n=len(dataset))
        scores = [scorer(self.infer(item["task"]), item.get("expected")) for item in dataset]
        report = {
            "n": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "scores": scores,
        }
        self._emit("evaluate", "eval_done", **report)
        return report

    def evaluate_agentic(self, tasks: list[str]) -> AgenticEvalReport:
        """Run each task and score its trajectory with the existing evaluator's
        programmatic metrics (reuses TrajectoryEvaluator, no model call). Emits a
        lifecycle event. This is the spine reaching a REAL consumer end-to-end."""
        from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

        self._emit("evaluate_agentic", "started", n=len(tasks))
        ev = TrajectoryEvaluator(model_name="none", api_base=None)
        rows = [agentic_metrics(self.infer(t), ev) for t in tasks]
        metrics = {
            m: (sum(r[m] for r in rows) / len(rows) if rows else 0.0) for m in _PROGRAMMATIC_METRICS
        }
        report = {"n": len(rows), "metrics": metrics, "per_trajectory": rows}
        self._emit("evaluate_agentic", "eval_done", n=len(rows))
        return report

    def train(
        self,
        trainer_factory=None,
        *,
        fmt: str = "sft",
        rollout_engine=None,
        tools=None,
        max_steps: int = 8,
        reward_fn=None,
        system_prompt=None,
        **trainer_kwargs,
    ):
        """Run the EXISTING trainer through the spine, emitting lifecycle events.

        - ``fmt='sft'`` (default): build a dataset from the spine's OWN full-tier
          trajectories (``EventLog.as_dataset_rows``, whose ``messages`` schema matches the
          real SFT trainer — verified in test_train_wiring.py) and delegate to
          ``trainer_factory``. This is also the distillation dataset path.
        - ``fmt='grpo'``: on-policy RL. Wires the real ``create_rollout_fn(rollout_engine=…)``
          as the trainer's ``rollout_func`` (the GRPO on-policy contract — env_mask/logprobs/
          prompt_ids). Rollout runs inside the real trainer with a real model; Project wires
          the engine in. Requires ``rollout_engine`` (Phase 2b).

        ``trainer_factory`` returns an object with ``.train() -> dict`` (e.g. the existing
        ``TRLSFTTrainer`` / ``TrlAgenticGrpo``), so Project stays GPU/dep-free."""
        if trainer_factory is None:
            raise ValueError(
                "Project.train requires a `trainer_factory` returning an object with "
                ".train() (e.g. agenttune's TRLSFTTrainer / TrlAgenticGrpo)."
            )
        if fmt == "grpo":
            if rollout_engine is None:
                raise ValueError(
                    "Project.train(fmt='grpo') is on-policy — it requires a `rollout_engine` "
                    "(Phase 2b) to generate rollouts during training, not a precollected buffer."
                )
            from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

            rollout_func = create_rollout_fn(
                rollout_engine=rollout_engine,
                tools=tools or [],
                max_steps=max_steps,
                reward_fn=reward_fn,
                system_prompt=system_prompt,
            )
            self._emit("train", "started", fmt=fmt)
            trainer = trainer_factory(rollout_func=rollout_func, **trainer_kwargs)
            result = trainer.train()
            self._emit("train", "done", result=result)
            return result
        if fmt != "sft":
            raise NotImplementedError(
                f"Project.train supports fmt='sft' and fmt='grpo'; fmt={fmt!r} is not wired."
            )
        full = [t for t in self._trajectories if getattr(t, "tier", None) == "full"]
        if not full:
            raise ValueError(
                "Project.train needs full-tier trajectories (teacher/rollout, carrying "
                "logprobs). Add them via add_trajectory(EventLog.from_trajectory(...)); "
                "light-tier observational logs (e.g. DictToolHarness) cannot be trained on."
            )
        rows = [row for t in full for row in t.as_dataset_rows(fmt)]
        self._emit("train", "started", n_rows=len(rows), fmt=fmt, n_trajectories=len(full))
        trainer = trainer_factory(train_dataset=rows, **trainer_kwargs)
        result = trainer.train()
        self._emit("train", "done", result=result)
        return result

    def heal(self, *, detector=None, max_revisits: int = 3):
        """Detect failures in the spine's OWN collected trajectories using the existing
        closed-loop ``FailureDetector`` (loop_collapse / tool_crash), reusing its real logic.

        Each trajectory is projected to the detector's audit-record schema
        (``EventLog.to_audit_records``), written to a JSONL, and scanned by the REAL detector.
        Returns the ``Failure`` objects and emits a lifecycle event. Full self-healing
        (classify → regenerate → retrain) needs litellm and rides on top of this detection."""
        import json
        import os
        import tempfile

        from agenttune.decide.closed_loop.failure_detector import FailureDetector

        detector = detector or FailureDetector(max_revisits=max_revisits)
        records = [r for log in self._trajectories for r in log.to_audit_records()]
        self._emit("heal", "started", n_records=len(records))

        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="agenttune_heal_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            failures = list(detector.scan_audit_log(path))
        finally:
            for p in (path, path + ".offset"):
                if os.path.exists(p):
                    os.remove(p)

        self._emit(
            "heal",
            "detected",
            n_failures=len(failures),
            types=sorted({f.failure_type for f in failures}),
        )
        return failures

    def distill(
        self,
        student,
        *,
        trainer_factory=None,
        teacher_engine=None,
        tasks=None,
        fmt: str = "sft",
        **trainer_kwargs,
    ):
        """Agentic distillation — compress an expensive agent DESIGN into a small ``student``
        model by SFT on the TEACHER's own trajectories (behavior cloning, not weight-KD).

        Rides the same rails as ``train(fmt='sft')`` (dataset fits the real SFT ``messages``
        schema). If ``teacher_engine`` + ``tasks`` are given, teacher rollouts are collected
        first (``collect_rollout``); otherwise it distills from already-collected full-tier
        trajectories (``collect_rollout``/``add_trajectory``). ``trainer_factory`` builds the
        student trainer, so Project stays GPU/dep-free while delegating to the real runner."""
        if trainer_factory is None:
            raise ValueError(
                "Project.distill requires a `trainer_factory` for the student "
                "(e.g. agenttune's TRLSFTTrainer)."
            )
        if teacher_engine is not None:
            if not tasks:
                raise ValueError(
                    "Project.distill(teacher_engine=…) needs `tasks` to roll the teacher out on."
                )
            self.collect_rollout(teacher_engine, tasks, **trainer_kwargs.pop("rollout_kwargs", {}))
        full = [t for t in self._trajectories if getattr(t, "tier", None) == "full"]
        if not full:
            raise ValueError(
                "Project.distill needs teacher trajectories. Provide teacher_engine+tasks, or "
                "collect_rollout(...) / add_trajectory(...) full-tier teacher trajectories first."
            )
        rows = [row for t in full for row in t.as_dataset_rows(fmt)]
        self._emit(
            "distill",
            "started",
            student=student,
            n_rows=len(rows),
            n_teacher_trajectories=len(full),
        )
        trainer = trainer_factory(model=student, train_dataset=rows, **trainer_kwargs)
        result = trainer.train()
        self._emit("distill", "done", student=student, result=result)
        return result
