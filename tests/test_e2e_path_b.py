"""
End-to-End (E2E) tests for the Week-1 Path B slice.

These exercise the full Path A -> Path B handshake with realistic data,
ending at the Week-1 finish line (trigger decision / test-set artifact) — not
at a trained model (Weeks 2-3) or the continuous daemon (Week 4).

See .claude/plans/e2e_test_cases.md for the design narrative.
"""

import threading
from pathlib import Path

from agenttune.decide.closed_loop import (
    BackgroundRetrainer,
    DeploymentGate,
    RetrainingTrigger,
    RewardDriftTracker,
    TriggerConfig,
)
from agenttune.decide.closed_loop.contracts import TrainingExample

FIXTURES = Path(__file__).parent
GATE_AUDIT = FIXTURES / "deployment_gate_audit.jsonl"
DRIFT_AUDIT = FIXTURES / "drift_audit.jsonl"


def _example(root_cause, tid, accepted=True):
    """A TrainingExample shaped exactly as Path A's generator emits."""
    return TrainingExample(
        trajectory_id=tid,
        original_failure_type="tool_crash",
        root_cause=root_cause,
        prompt=[{"role": "user", "content": f"task for {tid}"}],
        chosen=[{"role": "assistant", "content": "corrected action"}] if accepted else None,
        rejected=[{"role": "assistant", "content": "failed action"}] if accepted else None,
        salvaged_at_attempt=1,
    )


# ---------------------------------------------------------------------------
# E2E-1 — Path A -> Buffer -> Trigger fires (the core handshake)
# ---------------------------------------------------------------------------


def test_e2e_handshake_fires():
    trig = RetrainingTrigger(
        TriggerConfig(total_failures_threshold=20, dominance_ratio=2.0, min_examples_ready=999)
    )
    causes = [
        "wrong_tool",
        "wrong_routing",
        "incomplete_reasoning",
        "hallucinated_output",
        "loop_collapse",
    ]
    for i in range(20):
        trig.buffer.add(_example(causes[i % len(causes)], f"t{i}"))

    examples = trig.check_and_fire()
    assert examples is not None
    assert len(examples) == 20
    assert len(trig.buffer) == 0
    assert all(isinstance(e, TrainingExample) for e in examples)
    assert all(e.chosen or e.rejected for e in examples)


# ---------------------------------------------------------------------------
# E2E-2 — Real audit log -> RewardDriftTracker -> drift trigger
# ---------------------------------------------------------------------------


def test_e2e_drift_from_audit():
    tracker = RewardDriftTracker(window_size=10, baseline_size=20)
    loaded = tracker.load_from_audit(str(DRIFT_AUDIT))
    assert loaded == 30  # 20 healthy + 10 degraded
    assert tracker.baseline_locked is True

    trig = RetrainingTrigger(
        config=TriggerConfig(
            total_failures_threshold=999,
            min_examples_ready=999,
            dominance_ratio=2.0,
            reward_drift_sigma=1.5,
        ),
        reward_tracker=tracker,
    )
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "reward_drift"


# ---------------------------------------------------------------------------
# E2E-3 — Novel failure type emerges after a retrain cycle
# ---------------------------------------------------------------------------


def test_e2e_novel_failure_after_retrain():
    trig = RetrainingTrigger(
        TriggerConfig(
            total_failures_threshold=999,
            min_examples_ready=999,
            dominance_ratio=2.0,
            novel_type_min_count=3,
        )
    )
    # Production has been seeing wrong_tool; it fires and drains.
    for i in range(10):
        trig.buffer.add(_example("wrong_tool", f"w{i}"))
    drained = trig.buffer.drain()
    assert len(drained) == 10

    # Simulate the retrain cycle that just happened.
    trig.mark_retrain_started()
    trig.mark_retrain_finished()

    # A never-seen failure type now appears 3 times.
    for i in range(3):
        trig.buffer.add(_example("hallucinated_output", f"h{i}"))

    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "novel_failure:hallucinated_output"


# ---------------------------------------------------------------------------
# E2E-4 — No double-retrain under concurrency
# ---------------------------------------------------------------------------


def test_e2e_no_double_retrain():
    trig = RetrainingTrigger(
        TriggerConfig(
            total_failures_threshold=5,
            dominance_ratio=2.0,
            min_examples_ready=999,
            max_buffer_size=10_000,
        )
    )
    for i in range(5):
        trig.buffer.add(_example("wrong_tool", f"seed{i}"))

    # First fire drains the buffer, then we mark a retrain in progress.
    first = trig.check_and_fire()
    assert first is not None and len(first) == 5
    trig.mark_retrain_started()

    fire_results = []
    added = []

    def producer(tid):
        for i in range(50):
            ex = _example("wrong_tool", f"{tid}_{i}")
            trig.buffer.add(ex)
            added.append(ex)

    def consumer():
        for _ in range(50):
            fire_results.append(trig.check_and_fire())

    threads = [threading.Thread(target=producer, args=(t,)) for t in range(4)]
    threads.append(threading.Thread(target=consumer))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # During the in-progress window, NOTHING fired (no second drain).
    assert all(r is None for r in fire_results)
    # Every example added during the window is still buffered (none lost).
    assert len(trig.buffer) == len(added) == 200
    # After finishing, a fresh check can fire again.
    trig.mark_retrain_finished()
    again = trig.check_and_fire()
    assert again is not None
    assert len(again) == 200


# ---------------------------------------------------------------------------
# E2E-5 — Decide audit log -> DeploymentGate test set
# ---------------------------------------------------------------------------


def test_e2e_deployment_test_set():
    gate = DeploymentGate(seed=0)
    test_set = gate.build_test_set(str(GATE_AUDIT), min_samples=1, max_samples=200)

    pids = {tc["pipeline_id"] for tc in test_set}
    assert pids == {"pipe_001", "pipe_002", "pipe_005"}  # only successful
    for tc in test_set:
        assert tc["input_text"]
        assert tc["expected_verdict"] in {"APPROVE", "COMPLETE", "PASS"}
        assert tc["expected_stage_outputs"]
    # Week 2: a perfect runner scores 1.0 on its own expected verdicts.
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in test_set}
    assert gate.score_model(test_set, lambda i: lookup.get(i)) == 1.0


# ---------------------------------------------------------------------------
# E2E-6 — Full Week-1 Path B slice (the demo narrative)
# ---------------------------------------------------------------------------


def test_e2e_full_week1_slice():
    # 1. Bootstrap drift baseline from a real audit log (healthy history).
    tracker = RewardDriftTracker(window_size=10, baseline_size=20)
    tracker.load_from_audit(str(DRIFT_AUDIT))

    trig = RetrainingTrigger(
        config=TriggerConfig(
            total_failures_threshold=999,
            min_examples_ready=999,
            dominance_ratio=0.70,
            reward_drift_sigma=1.5,
        ),
        reward_tracker=tracker,
    )

    # 2. Path A pushes a burst of classified failures; wrong_tool dominates ~75%.
    for i in range(15):
        trig.buffer.add(_example("wrong_tool", f"wt{i}"))
    for i in range(5):
        trig.buffer.add(_example("loop_collapse", f"lc{i}"))

    # 3. Health report card reflects the skew + a clean drop_rate.
    health = trig.buffer.health()
    assert health.buffer_size == 20
    assert health.dominant_failure_type == "wrong_tool"
    assert health.dominant_failure_ratio >= 0.70
    assert health.drop_rate == 0.0  # all examples had chosen+rejected

    # 4. Trigger fires on dominance and drains.
    examples = trig.check_and_fire()
    assert examples is not None and len(examples) == 20

    # 5. Retrain "starts" — Week 2 would launch training here.
    trig.mark_retrain_started()
    assert trig.is_retrain_running() is True

    # 6. Buffer keeps filling during the retrain; no second fire.
    for i in range(5):
        trig.buffer.add(_example("wrong_routing", f"wr{i}"))
    assert trig.check_and_fire() is None
    assert len(trig.buffer) == 5

    # 7. Retrain finishes.
    trig.mark_retrain_finished()
    assert trig.is_retrain_running() is False

    # 8. DeploymentGate builds the yardstick the Week-3 gate will score against.
    gate = DeploymentGate(seed=0)
    test_set = gate.build_test_set(str(GATE_AUDIT), min_samples=1)
    assert len(test_set) == 3


# ===========================================================================
# Week 2 E2E — drop-rate guard, background retrain, A/B compare
# ===========================================================================


def test_e2e_drop_rate_guard_skips_retrain():
    """A mostly-bad buffer trips a trigger but the drop-rate guard vetoes it."""
    trig = RetrainingTrigger(
        config=TriggerConfig(
            total_failures_threshold=5,
            dominance_ratio=2.0,
            min_examples_ready=999,
            max_drop_rate=0.50,
            min_attempts_for_drop_gate=4,
        )
    )
    # 2 usable + 4 no-signal → drop_rate ≈ 0.67, volume trigger would fire at 5.
    for i in range(2):
        trig.buffer.add(_example("wrong_tool", f"g{i}"))
    for i in range(4):
        trig.buffer.add(_example("wrong_tool", f"b{i}", accepted=False))

    fire, reason = trig.should_trigger()
    assert fire is False
    assert reason == "drop_rate_too_high"
    # Buffer is NOT drained when the guard blocks.
    assert trig.check_and_fire() is None
    assert len(trig.buffer) == 6


def test_e2e_background_retrain_buffer_keeps_filling():
    """Trigger fires -> background retrain (stub) runs -> buffer keeps filling,
    no double-retrain -> done signal clears the flag."""
    trig = RetrainingTrigger(
        config=TriggerConfig(
            total_failures_threshold=10,
            dominance_ratio=2.0,
            min_examples_ready=999,
            max_buffer_size=10_000,
        )
    )
    causes = ["wrong_tool", "wrong_routing", "loop_collapse"]
    for i in range(10):
        trig.buffer.add(_example(causes[i % 3], f"t{i}"))

    drained = trig.check_and_fire()
    assert drained is not None and len(drained) == 10

    # Launch a stub "retrain" in the background, driven via the trigger flag.
    release = threading.Event()

    def stub_retrain(examples):
        release.wait(timeout=5)
        return {"trained_on": len(examples)}

    runner = BackgroundRetrainer(trigger=trig)
    assert runner.start(stub_retrain, drained) is True

    # During the retrain: detection keeps pushing; no second fire.
    for i in range(7):
        trig.buffer.add(_example("wrong_tool", f"live{i}"))
    assert trig.check_and_fire() is None  # blocked by retrain_in_progress
    assert len(trig.buffer) == 7  # buffer kept filling

    # Finish the retrain; flag clears; result delivered.
    release.set()
    result = runner.wait(timeout=5)
    assert result.success is True
    assert result.result == {"trained_on": 10}
    assert trig.is_retrain_running() is False


def test_e2e_ab_compare_new_better_recommends_deploy():
    """Full slice: build test set -> A/B score old vs new -> recommendation."""
    gate = DeploymentGate(seed=0)
    test_set = gate.build_test_set(str(GATE_AUDIT), min_samples=1)
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in test_set}

    def old_model(_i):
        return "DENY"  # always wrong → 0.0

    def new_model(i):
        return lookup.get(i)  # perfect → 1.0

    cmp = gate.compare_models(test_set, old_model, new_model)
    assert cmp.old_score == 0.0
    assert cmp.new_score == 1.0
    assert cmp.new_is_better is True
    assert cmp.recommendation == "deploy"


# ===========================================================================
# Week 3 E2E — full closed loop: detect -> trigger -> background retrain ->
# trajectory eval + gate -> deploy/keep, with the buffer filling throughout.
# ===========================================================================

from agenttune.agentic.tools.base import BaseTool, ToolResult
from agenttune.decide.closed_loop import (
    ClosedLoopRunner,
    IsolatedTool,
)
from agenttune.decide.closed_loop.contracts import AgenticEvalResult


class _SandboxTool(BaseTool):
    name = "sandbox_echo"
    description = "echoes input; used to prove off-process isolation"

    def _parameters(self):
        return {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}

    def execute(self, **kwargs) -> ToolResult:
        import os

        # pid proves which process executed the tool
        return ToolResult(success=True, output={"echo": kwargs.get("msg"), "pid": os.getpid()})


def _eval_batch(score, n=3):
    return [
        AgenticEvalResult(
            trajectory_id=f"e{i}",
            goal_completion_score=score,
            tool_sequence_validity=score,
            unnecessary_steps_penalty=score,
            error_recovery_score=score,
            overall_judge_score=score,
        )
        for i in range(n)
    ]


def test_e2e_full_loop_deploys_better_model():
    """The whole Week-3 Path B control path, unattended, with a stub retrain
    and mocked trajectory eval (no GPU / no LLM)."""

    trig = RetrainingTrigger(
        TriggerConfig(
            total_failures_threshold=6,
            dominance_ratio=2.0,
            min_examples_ready=999,
            max_buffer_size=10_000,
        )
    )
    gate = DeploymentGate(seed=0)
    test_set = gate.build_test_set(str(GATE_AUDIT), min_samples=1)
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in test_set}

    release = threading.Event()
    outcome = {}

    def retrain_job(examples):
        release.wait(timeout=5)
        return {"trained_on": len(examples), "path": "/models/new"}

    def on_done(result):
        # New model is perfect on task + better on trajectory → gate approves.
        decision = gate.evaluate_decision(
            test_set,
            old_model_fn=lambda _i: "DENY",  # 0.0
            new_model_fn=lambda i: lookup.get(i),  # 1.0
            old_trajectory_results=_eval_batch(0.5),
            new_trajectory_results=_eval_batch(0.85),
        )
        applied = gate.apply_decision(
            decision,
            config_path="cfg.yaml",
            trained_path=result.result["path"],
            deploy_fn=lambda config_path, trained_path: outcome.update(deployed=trained_path),
            rollback_fn=lambda config_path: outcome.update(kept_old=True),
        )
        outcome["applied"] = applied
        return decision

    runner = ClosedLoopRunner(trig, retrain_job=retrain_job, on_retrain_done=on_done)

    # 1. Detection feeds the buffer until the trigger fires.
    for i in range(6):
        runner.submit(_example(["wrong_tool", "loop_collapse", "wrong_routing"][i % 3], f"t{i}"))
    rec = runner.tick()
    assert rec.fired and rec.retrain_started and rec.drained == 6

    # 2. Buffer keeps filling during the retrain.
    for i in range(4):
        runner.submit(_example("wrong_tool", f"live{i}"))
    assert runner.tick().fired is False
    assert len(trig.buffer) == 4

    # 3. Retrain completes → gate evaluates → deploy.
    release.set()
    runner.wait_for_retrain(timeout=5)
    assert runner.last_decision.approved is True
    assert outcome["applied"] == "deployed"
    assert outcome["deployed"] == "/models/new"


def test_e2e_full_loop_keeps_old_on_regression():
    """Same loop, but the new model regresses → gate blocks → old model kept."""
    trig = RetrainingTrigger(
        TriggerConfig(total_failures_threshold=3, dominance_ratio=2.0, min_examples_ready=999)
    )
    gate = DeploymentGate(seed=0)
    test_set = gate.build_test_set(str(GATE_AUDIT), min_samples=1)
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in test_set}
    outcome = {}

    def on_done(result):
        decision = gate.evaluate_decision(
            test_set,
            old_model_fn=lambda i: lookup.get(i),  # 1.0 (old is good)
            new_model_fn=lambda _i: "DENY",  # 0.0 (new regressed)
        )
        gate.apply_decision(
            decision,
            config_path="cfg.yaml",
            rollback_fn=lambda config_path: outcome.update(kept_old=True),
        )
        return decision

    runner = ClosedLoopRunner(
        trig, retrain_job=lambda ex: {"path": "/models/bad"}, on_retrain_done=on_done
    )
    for i in range(3):
        runner.submit(_example(["a", "b", "c"][i], f"t{i}"))
    runner.tick()
    runner.wait_for_retrain(timeout=5)
    assert runner.last_decision.approved is False
    assert outcome.get("kept_old") is True


def test_e2e_tool_isolation_runs_off_process():
    """A sandbox tool runs in a separate process (different pid), proving L1
    isolation; the rollout loop interface is unchanged."""
    import os

    isolated = IsolatedTool(_SandboxTool(), timeout_s=30)
    res = isolated.execute(msg="hello from loop")
    assert res.success is True
    assert res.output["echo"] == "hello from loop"
    assert res.metadata["isolation"] == "process"
    # The tool ran in a DIFFERENT process than the test.
    assert res.output["pid"] != os.getpid()
