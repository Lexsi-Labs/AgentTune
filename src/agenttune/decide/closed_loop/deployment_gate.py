"""
Deployment Gate — Path B, Weeks 1–3
===================================

Gates the deployment of a retrained model.

- Week 1: ``build_test_set()`` — reconstruct ground-truth test cases from past
  *successful* pipeline runs in the Decide audit log.
- Week 2: ``score_model()`` / ``compare_models()`` — task-accuracy A/B scoring.
- Week 3: ``evaluate_decision()`` — the full gate. Blocks a deploy if the new
  model is worse on the task test set OR worse on Path A's trajectory eval
  scores (conservative secondary signal). Wraps the existing deployment bridge
  (``deploy`` / ``rollback``).

A "successful run" is a completion entry with ``is_complete == True``,
``error is None``, and a passing verdict (see ``PASS_VERDICTS``).  For each
such run we recover its per-stage entries (matched by ``pipeline_id``) so the
test case carries the original input and the stage outputs the model is
expected to reproduce.

The audit log has two entry shapes, both written by ``AuditWriter``:
  1. per-stage : {pipeline_id, stage_id, stage_type, input, output, reward, ...}
  2. completion: {pipeline_id, template_id, verdict, is_complete, error,
                  step_count, episode_reward, ...}
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agenttune.decide.state import PASS_VERDICTS

logger = logging.getLogger(__name__)


@dataclass
class ABComparison:
    """Result of A/B scoring an old model vs a new (retrained) model."""

    old_score: float
    new_score: float
    delta: float  # new_score - old_score
    num_cases: int
    new_is_better: bool  # new_score > old_score (by >= min_delta)
    recommendation: str  # "deploy" | "keep_old" (advisory in Week 2)


@dataclass
class GateDecision:
    """Final gate decision for a retrained model (Week 3).

    A deploy is allowed only if the new model is NOT worse on task accuracy AND
    NOT worse on trajectory quality (the conservative secondary signal). Either
    regression blocks the deploy.
    """

    approved: bool  # True → deploy, False → keep old (rollback/no-op)
    reason: str  # human-readable explanation
    task_delta: float = 0.0  # new task pass-rate − old
    trajectory_delta: float = 0.0  # new mean trajectory score − old
    task_old: float = 0.0
    task_new: float = 0.0
    trajectory_old: float | None = None
    trajectory_new: float | None = None
    num_cases: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class DeploymentGate:
    """Builds a ground-truth test set and gates deploy of a retrained model.

    Week 1: ``build_test_set``. Week 2: ``score_model`` / ``compare_models``.
    Week 3: ``evaluate_decision`` (task + trajectory gate) and ``apply_decision``
    (wrap the deployment bridge).
    """

    def __init__(self, seed: int | None = None) -> None:
        # Deterministic sampling when a seed is provided (reproducible test sets).
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Test-set builder
    # ------------------------------------------------------------------

    def build_test_set(
        self,
        audit_path: str,
        min_samples: int = 20,
        max_samples: int = 200,
    ) -> list[dict[str, Any]]:
        """Reconstruct test cases from successful pipeline runs.

        Args:
            audit_path: Path to the Decide ``audit.jsonl`` file.
            min_samples: If fewer successful runs than this are found, return
                an empty list (not enough signal for a meaningful gate).
            max_samples: Cap on the returned set; randomly sampled if exceeded.

        Returns:
            A list of test cases, each::

                {
                    "pipeline_id": str,
                    "template_id": Optional[str],
                    "input_text": Any,           # input to the first stage
                    "expected_verdict": str,
                    "expected_stage_outputs": {stage_id: output, ...},
                    "episode_reward": Optional[float],
                }
        """
        if not os.path.exists(audit_path):
            logger.warning("DeploymentGate.build_test_set: %s not found", audit_path)
            return []

        # Pass 1: collect per-stage entries grouped by pipeline, and the set of
        # successful completion entries.
        stages_by_pipeline: dict[str, list[dict[str, Any]]] = {}
        completions: dict[str, dict[str, Any]] = {}

        with open(audit_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                pid = record.get("pipeline_id")
                if not pid:
                    continue

                if "stage_type" in record or "stage_id" in record:
                    # per-stage entry
                    stages_by_pipeline.setdefault(pid, []).append(record)
                elif self._is_successful_completion(record):
                    # successful completion entry (last writer wins per pipeline)
                    completions[pid] = record

        # Pass 2: build a test case per successful pipeline.
        test_cases: list[dict[str, Any]] = []
        for pid, completion in completions.items():
            stages = stages_by_pipeline.get(pid, [])
            if not stages:
                # No stage detail to reconstruct an input — skip.
                continue

            input_text = self._first_stage_input(stages)
            if input_text is None:
                continue

            expected_outputs = {
                s.get("stage_id"): s.get("output") for s in stages if s.get("stage_id") is not None
            }

            test_cases.append(
                {
                    "pipeline_id": pid,
                    "template_id": completion.get("template_id"),
                    "input_text": input_text,
                    "expected_verdict": completion.get("verdict"),
                    "expected_stage_outputs": expected_outputs,
                    "episode_reward": completion.get("episode_reward"),
                }
            )

        if len(test_cases) < min_samples:
            logger.warning(
                "DeploymentGate.build_test_set: only %d successful runs found "
                "(min_samples=%d) — returning empty test set.",
                len(test_cases),
                min_samples,
            )
            return []

        if len(test_cases) > max_samples:
            test_cases = self._rng.sample(test_cases, max_samples)

        logger.info(
            "DeploymentGate.build_test_set: built %d test cases from %s",
            len(test_cases),
            audit_path,
        )
        return test_cases

    # ------------------------------------------------------------------
    # Scoring & A/B comparison
    # ------------------------------------------------------------------

    def score_model(
        self,
        test_set: list[dict[str, Any]],
        model_runner_fn: Callable[[Any], Any],
    ) -> float | None:
        """Score a model on the test set: fraction of verdicts that match.

        ``model_runner_fn(input_text) -> verdict`` is supplied by the caller
        (it runs the model / pipeline and returns the produced verdict). This
        keeps the gate model-agnostic and testable without a GPU.

        The returned verdict is compared case-insensitively against each test
        case's ``expected_verdict``.  Returns the pass rate in ``[0, 1]``, or
        ``None`` if the test set is empty.
        """
        if not test_set:
            logger.warning("DeploymentGate.score_model: empty test set — nothing to score.")
            return None

        passed = 0
        scored = 0
        for case in test_set:
            expected = str(case.get("expected_verdict") or "").upper()
            try:
                produced = model_runner_fn(case.get("input_text"))
            except Exception as exc:  # noqa: BLE001 — a crash is a failed case
                logger.warning("score_model: runner raised on %s: %s", case.get("pipeline_id"), exc)
                scored += 1
                continue
            scored += 1
            if str(produced or "").upper() == expected:
                passed += 1

        pass_rate = passed / scored if scored else 0.0
        logger.info("DeploymentGate.score_model: %d/%d passed (%.3f).", passed, scored, pass_rate)
        return round(pass_rate, 4)

    def compare_models(
        self,
        test_set: list[dict[str, Any]],
        old_model_fn: Callable[[Any], Any],
        new_model_fn: Callable[[Any], Any],
        min_delta: float = 0.0,
    ) -> ABComparison:
        """A/B score old vs new model on the same test set.

        ``new_is_better`` requires the new model to beat the old by at least
        ``min_delta`` (default 0.0 → strictly better).  The ``recommendation``
        is advisory in Week 2 — actually gating deploy/rollback is Week 3.
        """
        old_score = self.score_model(test_set, old_model_fn) or 0.0
        new_score = self.score_model(test_set, new_model_fn) or 0.0
        delta = round(new_score - old_score, 4)
        new_is_better = delta >= min_delta and new_score > old_score
        recommendation = "deploy" if new_is_better else "keep_old"
        logger.info(
            "DeploymentGate.compare_models: old=%.3f new=%.3f delta=%+.3f → %s",
            old_score,
            new_score,
            delta,
            recommendation,
        )
        return ABComparison(
            old_score=old_score,
            new_score=new_score,
            delta=delta,
            num_cases=len(test_set),
            new_is_better=new_is_better,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Week 3 — full gate: task accuracy AND trajectory quality
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_trajectory_score(results: list[Any]) -> float | None:
        """Mean composite trajectory score over a list of AgenticEvalResult.

        Incorporates base LLM judge score with the novel task-agnostic metrics
        (TAC, SCSR, IASA, EGS) and applies penalties for redundancy (ARR) and
        loop collapses (LCF).
        """
        scores = []
        for r in results or []:
            if hasattr(r, "overall_judge_score"):
                base = getattr(r, "overall_judge_score", 0.0)

                # Positive deterministic & specific LLM signals
                tac = getattr(r, "tac_score", 0.0)
                scsr = getattr(r, "scsr_score", 0.0)
                iasa = getattr(r, "iasa_score", 0.0)
                egs = getattr(r, "egs_score", 0.0)

                # Average the positive components
                composite = (base + tac + scsr + iasa + egs) / 5.0

                # Penalties for failure modes
                arr = getattr(r, "arr_score", 0.0)
                lcf = getattr(r, "lcf_score", 0)

                composite -= arr * 0.2  # Penalty for API redundancy
                if lcf > 0:
                    composite -= 0.1 * lcf  # Penalty for loop collapses

                scores.append(max(0.0, min(1.0, composite)))

        if not scores:
            return None
        return round(sum(scores) / len(scores), 4)

    def evaluate_decision(
        self,
        test_set: list[dict[str, Any]],
        old_model_fn: Callable[[Any], Any],
        new_model_fn: Callable[[Any], Any],
        old_trajectory_results: list[Any] | None = None,
        new_trajectory_results: list[Any] | None = None,
        task_regression_tol: float = 0.0,
        trajectory_regression_tol: float = 0.05,
    ) -> GateDecision:
        """Decide whether to deploy the new model.

        Blocks the deploy if EITHER signal regresses:
          - Task accuracy: ``new_task < old_task - task_regression_tol``.
          - Trajectory quality: ``new_traj < old_traj - trajectory_regression_tol``
            (conservative secondary signal; only applied when both trajectory
            scores are supplied — see Path A's ``TrajectoryEvaluator``).

        ``*_trajectory_results`` are lists of ``AgenticEvalResult`` (from
        ``TrajectoryEvaluator.evaluate_batch``); pass ``None`` to skip the
        trajectory check (e.g. when the eval suite hasn't run).
        """
        task_old = self.score_model(test_set, old_model_fn) or 0.0
        task_new = self.score_model(test_set, new_model_fn) or 0.0
        task_delta = round(task_new - task_old, 4)

        traj_old = self._mean_trajectory_score(old_trajectory_results)
        traj_new = self._mean_trajectory_score(new_trajectory_results)
        traj_delta = (
            round((traj_new or 0.0) - (traj_old or 0.0), 4)
            if traj_old is not None and traj_new is not None
            else 0.0
        )

        # --- regression checks ---
        task_regressed = task_new < (task_old - task_regression_tol)
        trajectory_regressed = (
            traj_old is not None
            and traj_new is not None
            and traj_new < (traj_old - trajectory_regression_tol)
        )

        if task_regressed and trajectory_regressed:
            approved, reason = False, "blocked: task AND trajectory regressed"
        elif task_regressed:
            approved, reason = False, "blocked: task accuracy regressed"
        elif trajectory_regressed:
            approved, reason = False, "blocked: trajectory quality regressed"
        else:
            approved, reason = True, "approved: no regression on task or trajectory"

        decision = GateDecision(
            approved=approved,
            reason=reason,
            task_delta=task_delta,
            trajectory_delta=traj_delta,
            task_old=task_old,
            task_new=task_new,
            trajectory_old=traj_old,
            trajectory_new=traj_new,
            num_cases=len(test_set),
            details={
                "task_regressed": task_regressed,
                "trajectory_regressed": trajectory_regressed,
                "task_regression_tol": task_regression_tol,
                "trajectory_regression_tol": trajectory_regression_tol,
                "trajectory_checked": traj_old is not None and traj_new is not None,
            },
        )
        logger.info(
            "GateDecision: approved=%s (%s) task %.3f→%.3f traj %s→%s",
            approved,
            reason,
            task_old,
            task_new,
            traj_old,
            traj_new,
        )
        return decision

    def apply_decision(
        self,
        decision: GateDecision,
        config_path: str,
        trained_path: str | None = None,
        deploy_fn: Callable[..., Any] | None = None,
        rollback_fn: Callable[..., Any] | None = None,
    ) -> str:
        """Act on a :class:`GateDecision` via the deployment bridge.

        On ``approved`` → deploy the new model; otherwise → keep the old model
        (rollback hook, a no-op by default). ``deploy_fn`` / ``rollback_fn`` are
        injectable so this is testable without touching real deployments; when
        omitted they default to the Decide ``model_deployment`` bridge.

        Returns one of ``"deployed"`` / ``"kept_old"``.
        """
        if decision.approved:
            if deploy_fn is None:
                from agenttune.decide.model_deployment import (
                    deploy_trained_model as deploy_fn,  # type: ignore
                )
            if trained_path is None:
                raise ValueError("apply_decision(approved): trained_path is required to deploy.")
            deploy_fn(config_path=config_path, trained_path=trained_path)
            logger.info("apply_decision: DEPLOYED new model (%s).", decision.reason)
            return "deployed"

        # Not approved → keep old model.
        if rollback_fn is not None:
            rollback_fn(config_path=config_path)
        logger.warning("apply_decision: KEPT OLD model (%s).", decision.reason)
        return "kept_old"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_successful_completion(record: dict[str, Any]) -> bool:
        """A completion entry that finished cleanly with a passing verdict."""
        if "is_complete" not in record and "verdict" not in record:
            return False
        if not record.get("is_complete", False):
            return False
        if record.get("error") is not None:
            return False
        verdict = str(record.get("verdict") or "").upper()
        return verdict in PASS_VERDICTS

    @staticmethod
    def _first_stage_input(stages: list[dict[str, Any]]) -> Any | None:
        """Return the input recorded for the earliest stage of a pipeline.

        Falls back across stages because not every stage records an ``input``.
        """
        for stage in stages:
            value = stage.get("input")
            if value is not None:
                return value
        return None
