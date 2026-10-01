"""
Week 4 — FullClosedLoop end-to-end wiring tests (GPU-free, no real LLM).

These prove the W4 orchestration ties Path A (detect → classify → generate →
validate → buffer) to Path B (trigger → background retrain → diversity + eval →
gate → deploy/keep-old) as one driven, non-blocking loop.

Only the external boundaries are stubbed — exactly what you'd mock in CI:
  - litellm.acompletion           (classifier + generator + evaluator LLM calls)
  - ReplayValidator subprocess     (via the real validator's default no-op script)
  - build_model_runner / retrain_job / deploy bridge (model-dependent boundaries)

Everything in between runs for real: the FailureDetector parsing a trajectory
audit log, FailureClassifier.classify_batch, TrainingExampleGenerator building
TrainingExamples with completions+rewards AND setting chosen/rejected, the
shared thread-safe buffer, the trigger's 6-condition decision, the
BackgroundRetrainer daemon thread, and the DeploymentGate task+trajectory gate.

Run:
    pytest tests/decide/closed_loop/test_w4_full_loop.py -v
"""

import json
import shutil
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from agenttune.decide.closed_loop import (
    FullClosedLoop,
    GateConfig,
    PathAConfig,
    RetrainingTrigger,
    TriggerConfig,
)

FIXTURES = Path(__file__).parent
TRAJ_AUDIT = FIXTURES / "trajectory_audit.jsonl"
GATE_AUDIT = FIXTURES / "deployment_gate_audit.jsonl"


# ---------------------------------------------------------------------------
# Stubs for the external boundaries
# ---------------------------------------------------------------------------


def _fake_completion(text: str):
    msg = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _make_acompletion():
    """A litellm.acompletion stub that serves both the classifier (JSON
    root_cause) and the generator (a 'corrected action' with a unique number so
    distinct completions score distinct rewards)."""
    counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        messages = kwargs.get("messages", [])
        system = messages[0]["content"] if messages else ""
        if "root cause analyzer" in system.lower():
            # Classifier call → return a valid root_cause JSON.
            return _fake_completion(
                json.dumps(
                    {
                        "root_cause": "wrong_tool",
                        "confidence": 0.9,
                        "analysis": "used the wrong tool for the task",
                    }
                )
            )
        if "trajectory evaluator" in system.lower():
            # Trajectory judge call → return the rubric JSON.
            return _fake_completion(
                json.dumps(
                    {
                        "goal_completion": 0.9,
                        "tool_sequence_validity": 0.9,
                        "unnecessary_steps": 0.9,
                        "error_recovery": 0.9,
                        "intent_action_alignment": 0.9,
                        "evidence_grounding": 0.9,
                        "overall_score": 0.9,
                    }
                )
            )
        # Generator synthesis call → unique corrected action text.
        counter["n"] += 1
        return _fake_completion(
            '{"tool": "execute_sql", "query": "SELECT COUNT(*) FROM subs", "v": %d}' % counter["n"]
        )

    return fake_acompletion


def _copy_audit(tmp_path: Path) -> str:
    """Copy the trajectory fixture into tmp so the detector's .offset file
    doesn't pollute the shared fixtures dir."""
    dst = tmp_path / "trajectory_audit.jsonl"
    shutil.copy(TRAJ_AUDIT, dst)
    return str(dst)


def _build_loop(
    audit_path,
    *,
    retrain_job,
    build_model_runner,
    old_model_runner,
    collect_trajectories=None,
    deploy_fn=None,
    rollback_fn=None,
    trigger=None,
    gate_cfg=None,
):
    return FullClosedLoop(
        path_a=PathAConfig(
            audit_log_path=audit_path,
            classifier_model="stub/model",
            generator_model="stub/model",
        ),
        retrain_job=retrain_job,
        build_model_runner=build_model_runner,
        old_model_runner=old_model_runner,
        trigger=trigger
        or RetrainingTrigger(
            TriggerConfig(
                total_failures_threshold=2,
                dominance_ratio=2.0,
                min_examples_ready=999,
                max_buffer_size=10_000,
            )
        ),
        gate_cfg=gate_cfg,
        collect_trajectories=collect_trajectories,
        deploy_fn=deploy_fn,
        rollback_fn=rollback_fn,
    )


# ---------------------------------------------------------------------------
# W4-1 — Path A ingest populates the buffer through the real pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_once_detects_classifies_generates_and_buffers(tmp_path):
    audit = _copy_audit(tmp_path)
    loop = _build_loop(
        audit,
        retrain_job=lambda ex: {"path": "/models/new"},
        build_model_runner=lambda path: (lambda i: "APPROVE"),
        old_model_runner=lambda i: "APPROVE",
    )
    with patch("litellm.acompletion", new=_make_acompletion()):
        submitted = await loop.ingest_once()

    # The fixture has wrong_tool (tool_crash) + loop_collapse failures; the
    # generator only salvages wrong_tool/loop_collapse → real examples buffered.
    assert submitted >= 1
    assert len(loop.trigger.buffer) == submitted
    drained = loop.trigger.buffer.drain()
    for ex in drained:
        assert ex.root_cause in ("wrong_tool", "loop_collapse")
        # Generator always populates completions+rewards — that's the contract.
        assert ex.has_completions()
        # Whether a preference pair is set depends on reward variance; with the
        # default no-op validator, rewards may tie (TAC=1.0, TER=0.0 for all).
        # DPO/BCO conversion correctness is covered by test_retrain_week2.py.


# ---------------------------------------------------------------------------
# W4-2 — Full loop deploys a better model, unattended
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_loop_deploys_better_model(tmp_path):
    audit = _copy_audit(tmp_path)
    gate_cfg = GateConfig(config_path="cfg.yaml", min_test_samples=1)
    outcome = {}

    # Old model is wrong everywhere; the trained model is perfect on the test set.
    gate_lookup = {}

    def build_model_runner(path):
        return lambda i: gate_lookup.get(i)  # perfect new model

    loop = _build_loop(
        audit,
        retrain_job=lambda ex: {"path": "/models/new", "trained_on": len(ex)},
        build_model_runner=build_model_runner,
        old_model_runner=lambda i: "DENY",  # always wrong → 0.0
        deploy_fn=lambda config_path, trained_path: outcome.update(deployed=trained_path),
        rollback_fn=lambda config_path: outcome.update(kept_old=True),
        gate_cfg=gate_cfg,
    )
    # Build the gate test set from the gate audit fixture (the loop built it from
    # the trajectory audit, which has no successful completions → empty).
    loop.test_set = loop.gate.build_test_set(str(GATE_AUDIT), min_samples=1)
    gate_lookup.update({tc["input_text"]: tc["expected_verdict"] for tc in loop.test_set})

    # Pre-submit enough examples to guarantee the trigger fires (threshold=2),
    # independent of how many the real detector finds in the fixture.
    from agenttune.decide.closed_loop.contracts import TrainingExample

    def _ex(i):
        return TrainingExample(
            trajectory_id=f"pre{i}",
            original_failure_type="tool_crash",
            root_cause="wrong_tool",
            prompt=[{"role": "user", "content": "x"}],
            chosen=[{"role": "assistant", "content": "correct"}],
            rejected=[{"role": "assistant", "content": "wrong"}],
        )

    for i in range(3):
        loop.runner.submit(_ex(i))

    with patch("litellm.acompletion", new=_make_acompletion()):
        rec = loop.tick()
        assert rec.fired and rec.retrain_started
        loop.wait_for_retrain(timeout=5)

    assert loop.last_decision is not None
    assert loop.last_decision.approved is True
    assert outcome.get("deployed") == "/models/new"
    assert loop.cycles[-1].applied == "deployed"


# ---------------------------------------------------------------------------
# W4-3 — Full loop keeps the old model when the new one regresses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_loop_keeps_old_on_regression(tmp_path):
    audit = _copy_audit(tmp_path)
    outcome = {}
    gate_lookup = {}

    loop = _build_loop(
        audit,
        retrain_job=lambda ex: {"path": "/models/bad"},
        build_model_runner=lambda path: (lambda i: "DENY"),  # regressed new model
        old_model_runner=lambda i: gate_lookup.get(i),  # perfect old model
        deploy_fn=lambda config_path, trained_path: outcome.update(deployed=trained_path),
        rollback_fn=lambda config_path: outcome.update(kept_old=True),
        gate_cfg=GateConfig(min_test_samples=1),
    )
    loop.test_set = loop.gate.build_test_set(str(GATE_AUDIT), min_samples=1)
    gate_lookup.update({tc["input_text"]: tc["expected_verdict"] for tc in loop.test_set})

    from agenttune.decide.closed_loop.contracts import TrainingExample

    def _ex(i):
        return TrainingExample(
            trajectory_id=f"pre{i}",
            original_failure_type="tool_crash",
            root_cause="wrong_tool",
            prompt=[{"role": "user", "content": "x"}],
            chosen=[{"role": "assistant", "content": "correct"}],
            rejected=[{"role": "assistant", "content": "wrong"}],
        )

    for i in range(3):
        loop.runner.submit(_ex(i))

    with patch("litellm.acompletion", new=_make_acompletion()):
        loop.tick()
        loop.wait_for_retrain(timeout=5)

    assert loop.last_decision.approved is False
    assert outcome.get("kept_old") is True
    assert "deployed" not in outcome


# ---------------------------------------------------------------------------
# W4-4 — Buffer keeps filling during the retrain (non-blocking guarantee)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_keeps_filling_during_retrain(tmp_path):
    import threading

    audit = _copy_audit(tmp_path)
    release = threading.Event()
    gate_lookup = {}

    def slow_retrain(examples):
        release.wait(timeout=5)
        return {"path": "/models/new"}

    loop = _build_loop(
        audit,
        retrain_job=slow_retrain,
        build_model_runner=lambda path: (lambda i: gate_lookup.get(i)),
        old_model_runner=lambda i: "DENY",
        deploy_fn=lambda config_path, trained_path: None,
        gate_cfg=GateConfig(min_test_samples=1),
    )
    loop.test_set = loop.gate.build_test_set(str(GATE_AUDIT), min_samples=1)
    gate_lookup.update({tc["input_text"]: tc["expected_verdict"] for tc in loop.test_set})

    with patch("litellm.acompletion", new=_make_acompletion()):
        await loop.ingest_once()
        rec = loop.tick()
        assert rec.fired and rec.retrain_started
        # Buffer was drained on fire.
        assert len(loop.trigger.buffer) == 0

        # While the retrain blocks, a second ingest keeps filling the buffer and
        # a tick is a gated no-op (retrain_already_running).
        from agenttune.decide.closed_loop.contracts import TrainingExample

        loop.runner.submit(
            TrainingExample(
                trajectory_id="live1",
                original_failure_type="tool_crash",
                root_cause="wrong_tool",
                prompt=[{"role": "user", "content": "x"}],
                chosen=[{"role": "assistant", "content": "c"}],
                rejected=[{"role": "assistant", "content": "r"}],
            )
        )
        assert len(loop.trigger.buffer) == 1
        assert loop.tick().fired is False  # gated: retrain in progress

        release.set()
        loop.wait_for_retrain(timeout=5)

    # Retrain finished; the live example is still buffered for the next cycle.
    assert len(loop.trigger.buffer) == 1


# ---------------------------------------------------------------------------
# W4-5 — Trajectory signal: diversity monitor + trajectory-quality gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trajectory_signal_feeds_gate_and_diversity(tmp_path):
    audit = _copy_audit(tmp_path)

    # New model: identical repeated tool sequence → diversity collapse; and a
    # trajectory mean WORSE than old → trajectory regression blocks the deploy
    # even though task accuracy is tied.
    #
    # Old ("GOOD") trajectories use two distinct tool calls (no redundancy
    # penalty in DeploymentGate._mean_trajectory_score's ARR term); new
    # ("BAD") trajectories repeat the same call (ARR penalty applies) — this
    # is what actually produces the redundant pattern the test's own comment
    # describes, on top of the judge-score gap, so the composite trajectory
    # score (a blend of judge score + TAC/SCSR/IASA/EGS, minus ARR/LCF
    # penalties — see DeploymentGate._mean_trajectory_score) regresses by
    # more than trajectory_regression_tol.
    def collect_trajectories(runner):
        verdict = runner("probe")
        # Old runner returns "GOOD" (high score), new returns "BAD" (low score).
        score_tag = verdict
        tool_calls = ["search", "search"] if score_tag == "BAD" else ["search", "read"]
        return [
            {
                "trajectory_id": f"{score_tag}{i}",
                "tool_calls": tool_calls,
                "tool_outputs": ["x"],
                "final_answer": "a",
                "reference_answer": "a",
            }
            for i in range(12)
        ]

    # Make the trajectory judge score depend on the verdict tag via a custom
    # acompletion that reads the trajectory id.
    async def judge_aware_acompletion(**kwargs):
        messages = kwargs.get("messages", [])
        system = messages[0]["content"] if messages else ""
        if "trajectory evaluator" in system.lower():
            user = messages[1]["content"] if len(messages) > 1 else ""
            score = 0.9 if "GOOD" in user else 0.4
            return _fake_completion(
                json.dumps(
                    {
                        "goal_completion": score,
                        "tool_sequence_validity": score,
                        "unnecessary_steps": score,
                        "error_recovery": score,
                        "intent_action_alignment": score,
                        "evidence_grounding": score,
                        "overall_score": score,
                    }
                )
            )
        if "root cause analyzer" in system.lower():
            return _fake_completion(
                json.dumps({"root_cause": "wrong_tool", "confidence": 0.9, "analysis": "x"})
            )
        return _fake_completion('{"tool": "execute_sql", "v": 1}')

    loop = _build_loop(
        audit,
        retrain_job=lambda ex: {"path": "/models/new"},
        build_model_runner=lambda path: (lambda i: "BAD"),  # new → low traj score
        old_model_runner=lambda i: "GOOD",  # old → high traj score
        collect_trajectories=collect_trajectories,
        deploy_fn=lambda config_path, trained_path: None,
        rollback_fn=lambda config_path: None,
        gate_cfg=GateConfig(min_test_samples=1, trajectory_regression_tol=0.05),
    )
    # Tie task accuracy (both models score equally), so only the trajectory
    # signal decides. Empty test set → task scores both 0.0 → tied.
    loop.test_set = []

    # Pre-submit examples so the trigger fires deterministically — this test
    # is about the gate+diversity signal, not the ingest path.
    from agenttune.decide.closed_loop.contracts import TrainingExample

    def _ex(i):
        return TrainingExample(
            trajectory_id=f"pre{i}",
            original_failure_type="tool_crash",
            root_cause="wrong_tool",
            prompt=[{"role": "user", "content": "x"}],
            chosen=[{"role": "assistant", "content": "c"}],
            rejected=[{"role": "assistant", "content": "r"}],
        )

    for i in range(3):
        loop.runner.submit(_ex(i))

    with patch("litellm.acompletion", new=judge_aware_acompletion):
        loop.tick()
        loop.wait_for_retrain(timeout=5)

    assert loop.last_decision is not None, "on_retrain_done callback did not complete"
    cyc = loop.cycles[-1]
    # Trajectory regressed (0.4 < 0.9 - 0.05) → gate blocks.
    assert loop.last_decision.approved is False
    assert "trajectory" in loop.last_decision.reason
    # New model's repeated sequence tripped the diversity collapse alert.
    assert cyc.diversity_collapsed is True
