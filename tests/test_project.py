import pytest

from agenttune.agentic.events import EventKind
from agenttune.agentic.harness import DictToolHarness
from agenttune.agentic.project import LifecycleEvent, Project, answer_match
from agenttune.agentic.strategy import ReActStrategy


def _finish_policy(answer):
    return lambda state: {"name": "finish", "arguments": {"answer": answer}}


def _project(answer="42"):
    return Project(
        strategy=ReActStrategy(policy=_finish_policy(answer)),
        harness=DictToolHarness({"echo": lambda text="": text}),
    )


# ---------- Task 1 ----------


def test_infer_requires_strategy_and_harness():
    with pytest.raises(ValueError):
        Project().infer("t")


def test_infer_runs_episode_and_records():
    p = _project()
    log = p.infer("what is 6x7")
    assert any(e.kind is EventKind.TOOL_RESULT for e in log)
    assert len(p.trajectories) == 1
    stages = [ev.stage for ev in p.events()]
    assert "infer" in stages


# ---------- Task 2 ----------


def test_collect_runs_all_tasks():
    p = _project()
    logs = p.collect(["a", "b", "c"])
    assert len(logs) == 3 and len(p.trajectories) == 3
    kinds = [(ev.stage, ev.kind) for ev in p.events()]
    assert ("collect", "done") in kinds


# ---------- Task 3 ----------


def test_answer_match_scorer():
    p = _project(answer="the answer is 42")
    log = p.infer("q")
    assert answer_match(log, "42") == 1.0
    assert answer_match(log, "99") == 0.0


def test_evaluate_reports_mean_and_emits_event():
    p = _project(answer="42")
    report = p.evaluate([{"task": "q1", "expected": "42"}, {"task": "q2", "expected": "99"}])
    assert report["n"] == 2
    assert report["mean_score"] == 0.5
    eval_events = [ev for ev in p.events() if ev.kind == "eval_done"]
    assert eval_events and eval_events[0].data["mean_score"] == 0.5


# ---------- Task 4 ----------


def test_heal_on_empty_project_finds_nothing():
    p = _project()
    assert p.heal() == []


def test_exports():
    from agenttune.agentic import LifecycleEvent as L
    from agenttune.agentic import Project as P

    assert P is Project and L is LifecycleEvent
