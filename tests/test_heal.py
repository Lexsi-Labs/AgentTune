"""Phase 6 (heal) — detect failures in the spine's OWN trajectories using the EXISTING
closed-loop FailureDetector, reusing its real logic (loop_collapse / tool_crash), not a rewrite.

Discriminating (GPU/network-free): a spine trajectory that loops on one tool must, once
projected to the detector's audit-record schema, trigger the REAL FailureDetector to yield a
`loop_collapse` Failure — proving the spine's traces reach the real self-healing entry point.

Full self-healing (classify -> regenerate -> retrain) needs litellm and rides on top of this.
"""

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.project import Project


def _looping_log(tool="search", n=4):
    """A trajectory that calls the same tool n times — an agentic loop."""
    log = EventLog(tier="light")
    for _ in range(n):
        log.append(Event(EventKind.TOOL_CALL, {"action": {"name": tool, "arguments": {}}}))
        log.append(Event(EventKind.TOOL_RESULT, {"output": "same"}))
    return log


# ---- to_audit_records projection fits the detector schema ----


def test_to_audit_records_shape():
    recs = _looping_log(n=2).to_audit_records()
    assert len(recs) == 2
    for r in recs:
        # exact keys the FailureDetector reads
        assert r["trajectory_id"] and r["stage_name"] == "search"
        assert r["stage_type"] == "tool_call"
        assert isinstance(r["state_snapshot"], dict)


# ---- the REAL FailureDetector flags the spine's loop ----


def test_heal_detects_loop_collapse_via_real_detector():
    p = Project()
    p.add_trajectory(_looping_log(tool="search", n=4))  # 4 > max_revisits=3
    failures = p.heal(max_revisits=3)
    assert any(
        f.failure_type == "loop_collapse" and f.failed_stage_name == "search" for f in failures
    )
    # lifecycle event emitted
    assert any(ev.stage == "heal" and ev.kind == "detected" for ev in p.events())


def test_heal_clean_trajectory_yields_no_failures():
    p = Project()
    p.add_trajectory(_looping_log(tool="search", n=2))  # 2 <= max_revisits=3
    assert p.heal(max_revisits=3) == []


def test_heal_does_not_flag_distinct_arg_calls_as_loop():
    """Same tool, DIFFERENT args ×4 = legitimate exploration, NOT a loop. The action
    identity (tool+args) must keep this from firing loop_collapse."""
    log = EventLog(tier="light")
    for i in range(4):
        log.append(
            Event(
                EventKind.TOOL_CALL,
                {"action": {"name": "search", "arguments": {"q": f"query-{i}"}}},
            )
        )
        log.append(Event(EventKind.TOOL_RESULT, {"output": f"doc-{i}"}))
    p = Project()
    p.add_trajectory(log)
    assert not any(f.failure_type == "loop_collapse" for f in p.heal(max_revisits=3))


def test_heal_detects_tool_crash():
    """A tool result carrying an error marks its call crashed — detector flags tool_crash."""
    log = EventLog(tier="light")
    log.append(Event(EventKind.TOOL_CALL, {"action": {"name": "fetch", "arguments": {}}}))
    log.append(Event(EventKind.TOOL_RESULT, {"output": None, "error": "boom"}))
    p = Project()
    p.add_trajectory(log)
    failures = p.heal()
    assert any(f.failure_type == "tool_crash" and f.error_message == "boom" for f in failures)
