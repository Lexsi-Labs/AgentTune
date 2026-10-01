"""
Week 3 Path B unit tests:
- DeploymentGate.evaluate_decision (task + trajectory regression gate)
- DeploymentGate.apply_decision (deploy/keep-old via injected bridge fns)
- IsolatedTool (off-process execution + in-process fallback)
- ClosedLoopRunner (trigger -> background retrain -> gate, buffer keeps filling)

No GPU / no real model: retrain + eval + deploy are injected stub callables;
the isolated tool is a tiny picklable BaseTool.
"""

import threading

import pytest

from agenttune.agentic.tools.base import BaseTool, ToolResult
from agenttune.decide.closed_loop.closed_loop_runner import ClosedLoopRunner
from agenttune.decide.closed_loop.contracts import AgenticEvalResult, TrainingExample
from agenttune.decide.closed_loop.deployment_gate import DeploymentGate, GateDecision
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig
from agenttune.decide.closed_loop.tool_isolation import IsolatedTool, isolate_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _example(root_cause="wrong_tool", tid="t"):
    return TrainingExample(
        trajectory_id=tid,
        original_failure_type="tool_crash",
        root_cause=root_cause,
        prompt=[{"role": "user", "content": "do a task"}],
        chosen=[{"role": "assistant", "content": "correct"}],
        rejected=[{"role": "assistant", "content": "wrong"}],
    )


def _test_set():
    return [
        {
            "pipeline_id": "p1",
            "input_text": "a",
            "expected_verdict": "APPROVE",
            "expected_stage_outputs": {},
            "episode_reward": 0.9,
        },
        {
            "pipeline_id": "p2",
            "input_text": "b",
            "expected_verdict": "PASS",
            "expected_stage_outputs": {},
            "episode_reward": 0.8,
        },
    ]


def _eval_results(score):
    # DeploymentGate._mean_trajectory_score composites overall_judge_score with
    # tac/scsr/iasa/egs (see deployment_gate.py's "eval-metric integration" commit) --
    # set them all to `score` too so the composite equals `score` exactly, matching
    # this helper's original intent (uniform trajectory quality), not diluted 5x by
    # defaulted-to-0 component scores.
    return [
        AgenticEvalResult(
            trajectory_id=f"e{i}",
            goal_completion_score=score,
            tool_sequence_validity=score,
            unnecessary_steps_penalty=score,
            error_recovery_score=score,
            overall_judge_score=score,
            tac_score=score,
            scsr_score=score,
            iasa_score=score,
            egs_score=score,
        )
        for i in range(3)
    ]


# Module-level tools (must be top-level to be picklable for process isolation).
class EchoTool(BaseTool):
    name = "echo"
    description = "echoes input"

    def _parameters(self):
        return {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=kwargs.get("msg", ""))


class CrashTool(BaseTool):
    name = "crash"
    description = "always raises"

    def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("tool blew up")


class UnpicklableTool(BaseTool):
    name = "unpicklable"
    description = "holds a lock (not picklable)"

    def __init__(self):
        self._lock = threading.Lock()  # locks can't be pickled

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="ran locally")


# ---------------------------------------------------------------------------
# DeploymentGate.evaluate_decision — task + trajectory gate
# ---------------------------------------------------------------------------


def test_gate_approves_when_no_regression():
    gate = DeploymentGate(seed=0)
    ts = _test_set()
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in ts}

    def old(_i):
        return "DENY"  # 0.0

    def new(i):
        return lookup.get(i)  # 1.0

    dec = gate.evaluate_decision(
        ts,
        old,
        new,
        old_trajectory_results=_eval_results(0.5),
        new_trajectory_results=_eval_results(0.8),
    )
    assert dec.approved is True
    assert dec.task_delta == 1.0
    assert dec.trajectory_delta == pytest.approx(0.3)


def test_gate_blocks_on_task_regression():
    gate = DeploymentGate(seed=0)
    ts = _test_set()
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in ts}

    def old(i):
        return lookup.get(i)  # 1.0

    def new(_i):
        return "DENY"  # 0.0

    dec = gate.evaluate_decision(ts, old, new)
    assert dec.approved is False
    assert "task" in dec.reason


def test_gate_blocks_on_trajectory_regression():
    gate = DeploymentGate(seed=0)
    ts = _test_set()
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in ts}

    def runner(i):
        return lookup.get(i)  # both task-perfect → task tie, not a regression

    dec = gate.evaluate_decision(
        ts,
        runner,
        runner,
        old_trajectory_results=_eval_results(0.9),
        new_trajectory_results=_eval_results(0.5),
    )  # big drop
    assert dec.approved is False
    assert "trajectory" in dec.reason


def test_gate_skips_trajectory_check_when_absent():
    gate = DeploymentGate(seed=0)
    ts = _test_set()
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in ts}

    def runner(i):
        return lookup.get(i)

    dec = gate.evaluate_decision(ts, runner, runner)  # no trajectory results
    assert dec.approved is True
    assert dec.details["trajectory_checked"] is False


def test_gate_trajectory_tolerance_allows_small_dip():
    gate = DeploymentGate(seed=0)
    ts = _test_set()
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in ts}

    def runner(i):
        return lookup.get(i)

    # 0.02 dip < default 0.05 tolerance → still approved.
    dec = gate.evaluate_decision(
        ts,
        runner,
        runner,
        old_trajectory_results=_eval_results(0.80),
        new_trajectory_results=_eval_results(0.78),
    )
    assert dec.approved is True


# ---------------------------------------------------------------------------
# DeploymentGate.apply_decision — deploy / keep-old via injected fns
# ---------------------------------------------------------------------------


def test_apply_decision_deploys_when_approved():
    gate = DeploymentGate(seed=0)
    calls = {}
    dec = GateDecision(approved=True, reason="ok")
    out = gate.apply_decision(
        dec,
        config_path="cfg.yaml",
        trained_path="/models/new",
        deploy_fn=lambda config_path, trained_path: calls.update(
            deployed=(config_path, trained_path)
        ),
    )
    assert out == "deployed"
    assert calls["deployed"] == ("cfg.yaml", "/models/new")


def test_apply_decision_keeps_old_when_blocked():
    gate = DeploymentGate(seed=0)
    calls = {}
    dec = GateDecision(approved=False, reason="blocked: task regressed")
    out = gate.apply_decision(
        dec,
        config_path="cfg.yaml",
        rollback_fn=lambda config_path: calls.update(rolled_back=config_path),
    )
    assert out == "kept_old"
    assert calls["rolled_back"] == "cfg.yaml"


def test_apply_decision_approved_requires_trained_path():
    gate = DeploymentGate(seed=0)
    dec = GateDecision(approved=True, reason="ok")
    with pytest.raises(ValueError):
        gate.apply_decision(dec, config_path="cfg.yaml", deploy_fn=lambda **k: None)


# ---------------------------------------------------------------------------
# IsolatedTool — off-process execution + fallback
# ---------------------------------------------------------------------------


def test_isolated_tool_runs_off_process():
    tool = IsolatedTool(EchoTool(), timeout_s=30)
    res = tool.execute(msg="hello")
    assert res.success is True
    assert res.output == "hello"
    assert res.metadata["isolation"] == "process"


def test_isolated_tool_preserves_schema_and_name():
    tool = IsolatedTool(EchoTool())
    assert tool.name == "echo"
    assert tool.to_schema()["function"]["name"] == "echo"
    assert "msg" in tool.to_schema()["function"]["parameters"]["properties"]


def test_isolated_tool_unpicklable_falls_back_in_process():
    tool = IsolatedTool(UnpicklableTool())
    res = tool.execute()
    assert res.success is True
    assert res.output == "ran locally"
    assert res.metadata["isolation"] == "in_process_fallback"
    assert res.metadata["fallback_reason"] == "not_picklable"


def test_isolated_tool_require_isolation_errors_when_unavailable():
    tool = IsolatedTool(UnpicklableTool(), require_isolation=True)
    res = tool.execute()
    assert res.success is False
    assert res.error == "isolation_unavailable"


def test_isolate_tools_helper_wraps_all():
    tools = isolate_tools([EchoTool(), EchoTool()], timeout_s=10)
    assert all(isinstance(t, IsolatedTool) for t in tools)
    assert len(tools) == 2


# ---------------------------------------------------------------------------
# ClosedLoopRunner — trigger -> background retrain -> gate
# ---------------------------------------------------------------------------


def test_runner_tick_no_fire_below_threshold():
    trig = RetrainingTrigger(
        TriggerConfig(total_failures_threshold=99, dominance_ratio=2.0, min_examples_ready=99)
    )
    runner = ClosedLoopRunner(trig, retrain_job=lambda ex: "model")
    runner.submit(_example())
    rec = runner.tick()
    assert rec.fired is False
    assert runner.is_retraining() is False


def test_runner_fires_and_buffer_keeps_filling_during_retrain():
    trig = RetrainingTrigger(
        TriggerConfig(
            total_failures_threshold=5,
            dominance_ratio=2.0,
            min_examples_ready=999,
            max_buffer_size=10_000,
        )
    )
    release = threading.Event()
    gate_calls = {}

    def slow_retrain(examples):
        release.wait(timeout=5)
        return {"trained_on": len(examples), "path": "/models/new"}

    def on_done(result):
        gate_calls["result"] = result
        return GateDecision(approved=True, reason="ok")

    runner = ClosedLoopRunner(trig, retrain_job=slow_retrain, on_retrain_done=on_done)

    for i in range(5):
        runner.submit(_example(["wrong_tool", "loop_collapse"][i % 2], f"t{i}"))

    rec = runner.tick()
    assert rec.fired is True
    assert rec.drained == 5
    assert rec.retrain_started is True
    assert runner.is_retraining() is True

    # Buffer keeps filling while the retrain runs; tick is a no-op (gated).
    for i in range(7):
        runner.submit(_example("wrong_tool", f"live{i}"))
    rec2 = runner.tick()
    assert rec2.fired is False
    assert rec2.reason == "retrain_already_running"
    assert len(trig.buffer) == 7

    # Finish retrain → gate runs via callback.
    release.set()
    result = runner.wait_for_retrain(timeout=5)
    assert result.success is True
    assert gate_calls["result"].result["trained_on"] == 5
    assert runner.last_decision.approved is True
    assert trig.is_retrain_running() is False


def test_runner_gate_blocks_bad_model():
    trig = RetrainingTrigger(
        TriggerConfig(total_failures_threshold=3, dominance_ratio=2.0, min_examples_ready=999)
    )

    def on_done(result):
        return GateDecision(approved=False, reason="blocked: task accuracy regressed")

    runner = ClosedLoopRunner(trig, retrain_job=lambda ex: "bad_model", on_retrain_done=on_done)
    for i in range(3):
        runner.submit(_example(["a", "b", "c"][i], f"t{i}"))
    runner.tick()
    runner.wait_for_retrain(timeout=5)
    assert runner.last_decision.approved is False
    assert "regressed" in runner.last_decision.reason
