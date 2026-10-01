"""
Tests for Path B DeploymentGate.

Week 1: test-set builder from a realistic fixture audit log (APPROVE/COMPLETE/
PASS successes, DENY failure, incomplete-with-error).
Week 2: score_model pass-rate + compare_models A/B scoring (model-agnostic via
an injected runner fn — no GPU needed).
"""

from pathlib import Path

import pytest

from agenttune.decide.closed_loop.deployment_gate import ABComparison, DeploymentGate

FIXTURE = Path(__file__).parent / "deployment_gate_audit.jsonl"


def test_build_test_set_from_successful_runs():
    gate = DeploymentGate(seed=0)
    # 3 successful runs in the fixture: pipe_001 (APPROVE), pipe_002 (COMPLETE),
    # pipe_005 (PASS). pipe_003 (DENY) and pipe_004 (error/incomplete) excluded.
    test_set = gate.build_test_set(str(FIXTURE), min_samples=1, max_samples=200)
    pids = {tc["pipeline_id"] for tc in test_set}
    assert pids == {"pipe_001", "pipe_002", "pipe_005"}
    verdicts = {tc["expected_verdict"] for tc in test_set}
    assert verdicts == {"APPROVE", "COMPLETE", "PASS"}


def test_build_test_set_correlates_stages():
    gate = DeploymentGate(seed=0)
    test_set = gate.build_test_set(str(FIXTURE), min_samples=1, max_samples=200)
    by_pid = {tc["pipeline_id"]: tc for tc in test_set}

    case = by_pid["pipe_001"]
    assert case["input_text"] == "Review KYC for Alice"
    assert case["template_id"] == "bfsi/kyc_triage"
    assert case["episode_reward"] == pytest.approx(0.95)
    # Both stages of the pipeline are recovered.
    assert set(case["expected_stage_outputs"].keys()) == {"extract", "decide"}
    assert case["expected_stage_outputs"]["decide"] == {"verdict": "APPROVE"}


def test_build_test_set_min_samples():
    gate = DeploymentGate(seed=0)
    # Only 3 successful runs exist; requiring 10 yields an empty set.
    test_set = gate.build_test_set(str(FIXTURE), min_samples=10, max_samples=200)
    assert test_set == []


def test_build_test_set_max_samples():
    gate = DeploymentGate(seed=0)
    test_set = gate.build_test_set(str(FIXTURE), min_samples=1, max_samples=2)
    assert len(test_set) == 2  # capped


def test_build_test_set_missing_file():
    gate = DeploymentGate(seed=0)
    assert gate.build_test_set("/nonexistent/audit.jsonl") == []


# ---------------------------------------------------------------------------
# Week 2 — score_model pass rate + compare_models A/B
# ---------------------------------------------------------------------------


def _perfect_runner(test_set):
    """A runner that returns the exact expected verdict for each input."""
    lookup = {tc["input_text"]: tc["expected_verdict"] for tc in test_set}
    return lambda input_text: lookup.get(input_text)


def test_score_model_perfect():
    gate = DeploymentGate(seed=0)
    ts = gate.build_test_set(str(FIXTURE), min_samples=1)
    score = gate.score_model(ts, _perfect_runner(ts))
    assert score == 1.0


def test_score_model_all_wrong():
    gate = DeploymentGate(seed=0)
    ts = gate.build_test_set(str(FIXTURE), min_samples=1)
    score = gate.score_model(ts, lambda _i: "DENY")  # never matches a passing verdict
    assert score == 0.0


def test_score_model_partial_and_runner_crash():
    gate = DeploymentGate(seed=0)
    ts = gate.build_test_set(str(FIXTURE), min_samples=1)  # 3 cases

    # Correct on the first case, crash on the rest → 1/3.
    correct_input = ts[0]["input_text"]
    correct_verdict = ts[0]["expected_verdict"]

    def flaky(input_text):
        if input_text == correct_input:
            return correct_verdict
        raise RuntimeError("model crashed")

    score = gate.score_model(ts, flaky)
    assert score == pytest.approx(1 / 3, abs=1e-3)


def test_score_model_empty_returns_none():
    gate = DeploymentGate(seed=0)
    assert gate.score_model([], lambda _i: "APPROVE") is None


def test_compare_models_new_better():
    gate = DeploymentGate(seed=0)
    ts = gate.build_test_set(str(FIXTURE), min_samples=1)

    def old_fn(_i):
        return "DENY"  # 0.0

    new_fn = _perfect_runner(ts)  # 1.0
    cmp = gate.compare_models(ts, old_fn, new_fn)
    assert isinstance(cmp, ABComparison)
    assert cmp.old_score == 0.0
    assert cmp.new_score == 1.0
    assert cmp.delta == 1.0
    assert cmp.new_is_better is True
    assert cmp.recommendation == "deploy"


def test_compare_models_new_worse_keeps_old():
    gate = DeploymentGate(seed=0)
    ts = gate.build_test_set(str(FIXTURE), min_samples=1)
    old_fn = _perfect_runner(ts)  # 1.0

    def new_fn(_i):
        return "DENY"  # 0.0

    cmp = gate.compare_models(ts, old_fn, new_fn)
    assert cmp.new_is_better is False
    assert cmp.recommendation == "keep_old"
    assert cmp.delta == -1.0


def test_compare_models_tie_keeps_old():
    gate = DeploymentGate(seed=0)
    ts = gate.build_test_set(str(FIXTURE), min_samples=1)
    runner = _perfect_runner(ts)
    cmp = gate.compare_models(ts, runner, runner)  # identical → tie
    assert cmp.delta == 0.0
    assert cmp.new_is_better is False
    assert cmp.recommendation == "keep_old"
