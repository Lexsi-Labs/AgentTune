import pytest

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.harness import (
    DictToolHarness,
    Harness,
    HarnessCapabilities,
    Observation,
    replay,
    run_conformance,
)


def _harness():
    return DictToolHarness({"echo": lambda text="": f"echo:{text}"}, max_steps=3)


# ---------- Task 1 ----------


def test_observation_and_capabilities_defaults():
    o = Observation(text="hi")
    assert o.text == "hi" and o.metadata == {}
    c = HarnessCapabilities()
    assert c.supports_snapshot is False and c.max_steps is None


def test_harness_is_abstract():
    with pytest.raises(TypeError):
        Harness()


# ---------- Task 2 ----------


def test_reset_and_step_records_events():
    h = _harness()
    obs = h.reset("do the thing")
    assert obs.text == "do the thing"
    obs, reward, done, info = h.step({"name": "echo", "arguments": {"text": "hi"}})
    assert obs.text == "echo:hi" and reward == 0.0 and done is False
    kinds = [e.kind for e in h.event_log]
    assert EventKind.TOOL_CALL in kinds and EventKind.TOOL_RESULT in kinds


def test_unknown_tool_penalized():
    h = _harness()
    h.reset("t")
    obs, reward, done, info = h.step({"name": "nope", "arguments": {}})
    assert reward == -1.0 and "unknown tool" in obs.text


def test_done_tool_ends_episode():
    h = _harness()
    h.reset("t")
    obs, reward, done, info = h.step({"name": "finish", "arguments": {"answer": "42"}})
    assert done is True and obs.text == "42"


def test_max_steps_terminates():
    h = _harness()
    h.reset("t")
    for _ in range(3):
        obs, reward, done, info = h.step({"name": "echo", "arguments": {"text": "x"}})
    assert done is True


# ---------- Task 3 ----------


def test_snapshot_restore_round_trips():
    h = _harness()
    h.reset("t")
    blob = h.snapshot()
    before = len(h.event_log)
    h.step({"name": "echo", "arguments": {"text": "x"}})
    assert len(h.event_log) > before
    h.restore(blob)
    assert len(h.event_log) == before


# ---------- Task 4 ----------


def test_conformance_passes_for_reference_harness():
    rep = run_conformance(_harness())
    assert rep.passed is True and rep.drift == []


def test_conformance_flags_drift_when_snapshot_lies():
    h = _harness()
    h.capabilities.supports_snapshot = True
    h.snapshot = lambda: b""  # broken snapshot that won't revert
    h.restore = lambda blob: None
    rep = run_conformance(h)
    assert rep.passed is False and any("snapshot" in d for d in rep.drift)


# ---------- Task 5 ----------


def test_replay_reexecutes_tool_calls():
    src = EventLog(
        events=[
            Event(EventKind.OBSERVATION, {"text": "start"}),
            Event(EventKind.TOOL_CALL, {"action": {"name": "echo", "arguments": {"text": "a"}}}),
        ]
    )
    out = replay(_harness(), src)
    outs = [e.payload["output"] for e in out if e.kind is EventKind.TOOL_RESULT]
    assert "echo:a" in outs


# ---------- Task 6 ----------


def test_exports_and_end_to_end():
    from agenttune.agentic import DictToolHarness as DTH
    from agenttune.agentic import run_conformance as rc

    h = DTH({"echo": lambda text="": f"echo:{text}"})
    assert rc(h).passed is True
    log = h.to_eventlog()
    assert log.tier == "light"
