"""`agenttune.agentic.events` is one package: the event types and the judgment hook both import,
and the hook ships in the wheel."""

from pathlib import Path


def test_event_names_still_import():
    from agenttune.agentic import Event as E2
    from agenttune.agentic.events import Event, EventKind, EventLog

    assert E2 is Event and EventKind.TOOL_RESULT and EventLog


def test_judgment_hook_imports():
    from agenttune.agentic.events.judgment_hook import JudgmentHook

    assert JudgmentHook.__module__ == "agenttune.agentic.events.judgment_hook"


def test_events_package_is_packaged():
    from setuptools import find_packages

    src = Path(__file__).resolve().parents[1] / "src"
    assert "agenttune.agentic.events" in find_packages(where=str(src))
