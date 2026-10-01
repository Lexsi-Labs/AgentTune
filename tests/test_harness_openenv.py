"""Phase 2b — OpenEnvHarness: the OpenEnv env driver behind the Harness contract.

These tests run with NO GPU and NO network: the harness is driven by an injected fake env
(dependency injection). The whole point is that the module imports and the harness works
regardless of whether the openenv extra is installed — so, unlike test_openenv_tool.py,
these tests DO NOT skip. One extra check (``test_openenv_absent_in_base_install``) asserts
the extra is genuinely absent, and is itself skipped when the extra happens to be present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.harness import (
    Harness,
    HarnessCapabilities,
    Observation,
    replay,
    run_conformance,
)
from agenttune.agentic.harness_openenv import OpenEnvHarness, _extract, _obs_text
from agenttune.utils.optional import OPENENV_AVAILABLE

# ---------------------------------------------------------------------------
# Fakes — stand-ins for OpenEnv's Observation / StepResult / Environment
# ---------------------------------------------------------------------------


@dataclass
class FakeObs:
    """Stand-in for an OpenEnv observation object (has a .text field)."""

    text: str = ""


@dataclass
class FakeStepResult:
    """Stand-in for OpenEnv's StepResult (observation / reward / done)."""

    observation: Any
    reward: float | None = None
    done: bool = False


class FakeEnv:
    """Minimal gym-like OpenEnv env: reset() + step(action) → StepResult.

    Echoes the action's ``arguments['text']`` back as the observation, rewards +1.0,
    and ends after ``finish`` or after ``max_env_steps`` internal steps.
    """

    def __init__(self, max_env_steps: int = 100, reset_text: str = ""):
        self.max_env_steps = max_env_steps
        self.reset_text = reset_text
        self.calls: list[Any] = []
        self._n = 0

    def reset(self) -> FakeStepResult:
        self._n = 0
        self.calls = []
        return FakeStepResult(observation=FakeObs(self.reset_text))

    def step(self, action: Any) -> FakeStepResult:
        self.calls.append(action)
        self._n += 1
        name = action.get("name") if isinstance(action, dict) else None
        args = (action.get("arguments") or {}) if isinstance(action, dict) else {}
        done = name == "finish" or self._n >= self.max_env_steps
        text = args.get("text", f"observed:{name}")
        return FakeStepResult(observation=FakeObs(text), reward=1.0, done=done)


def _space():
    return [{"name": "echo", "arguments": {"text": ""}}, {"name": "finish", "arguments": {}}]


def _harness(**kw) -> OpenEnvHarness:
    return OpenEnvHarness(FakeEnv(**kw.pop("env_kwargs", {})), action_space=_space(), **kw)


# ---------------------------------------------------------------------------
# DI / import-safety: no openenv required
# ---------------------------------------------------------------------------


def test_di_seam_works_off_injected_env():
    """The DI seam: OpenEnvHarness works purely off the injected env — no openenv import
    is needed to construct or reset it. Holds whether or not the openenv extra is installed."""
    h = _harness()
    assert isinstance(h.reset("t"), Observation)


@pytest.mark.skipif(
    OPENENV_AVAILABLE,
    reason="the base-install path — only reproducible when the openenv extra is absent",
)
def test_openenv_absent_in_base_install():
    """In a base install the openenv extra is genuinely absent, yet the module above still
    imports and the harness still runs — that is the DI seam being real, not incidental."""
    assert OPENENV_AVAILABLE is False


def test_is_a_harness():
    assert isinstance(_harness(), Harness)


# ---------------------------------------------------------------------------
# _extract tolerance (StepResult-shaped AND bare-observation)
# ---------------------------------------------------------------------------


def test_extract_from_step_result():
    obs, reward, done = _extract(FakeStepResult(observation=FakeObs("hi"), reward=2.0, done=True))
    assert isinstance(obs, FakeObs) and reward == 2.0 and done is True


def test_extract_from_bare_observation():
    bare = FakeObs("bare")
    obs, reward, done = _extract(bare)
    assert obs is bare and reward == 0.0 and done is False


def test_obs_text_prefers_text_field_then_str():
    assert _obs_text(FakeObs("x")) == "x"
    assert _obs_text(123) == "123"
    assert _obs_text(None) == ""


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_are_set_appropriately():
    caps = _harness().capabilities
    assert isinstance(caps, HarnessCapabilities)
    assert caps.supports_stepwise_turns is True
    assert caps.supports_snapshot is False
    assert caps.supports_streaming is False
    assert caps.supports_tool_boundary_interrupt is False


# ---------------------------------------------------------------------------
# reset / step / action_space delegation
# ---------------------------------------------------------------------------


def test_reset_seeds_observation_with_task():
    h = _harness()
    obs = h.reset("do the thing")
    assert obs.text == "do the thing"
    assert h.event_log[0].kind is EventKind.OBSERVATION
    assert h.event_log[0].payload["text"] == "do the thing"


def test_reset_uses_env_observation_text_when_present():
    h = OpenEnvHarness(FakeEnv(reset_text="welcome"), action_space=_space())
    obs = h.reset("ignored-because-env-has-text")
    assert obs.text == "welcome"


def test_step_delegates_to_env_and_records_events():
    h = _harness()
    h.reset("t")
    obs, reward, done, info = h.step({"name": "echo", "arguments": {"text": "hi"}})
    assert obs.text == "hi" and reward == 1.0 and done is False
    assert info["steps"] == 1
    kinds = [e.kind for e in h.event_log]
    assert EventKind.TOOL_CALL in kinds and EventKind.TOOL_RESULT in kinds
    assert EventKind.TURN_COMPLETE in kinds


def test_step_records_original_dict_action():
    h = _harness()
    h.reset("t")
    action = {"name": "echo", "arguments": {"text": "hi"}}
    h.step(action)
    call = next(e for e in h.event_log if e.kind is EventKind.TOOL_CALL)
    assert call.payload["action"] == action


def test_action_adapter_converts_before_env_call():
    env = FakeEnv()
    sentinel = object()
    h = OpenEnvHarness(env, action_space=_space(), action_adapter=lambda a: sentinel)
    h.reset("t")
    h.step({"name": "echo", "arguments": {}})
    # env saw the adapted action, not the raw dict
    assert env.calls == [sentinel]
    # but the event log kept the original dict
    call = next(e for e in h.event_log if e.kind is EventKind.TOOL_CALL)
    assert call.payload["action"] == {"name": "echo", "arguments": {}}


def test_env_done_propagates():
    h = _harness()
    h.reset("t")
    obs, reward, done, info = h.step({"name": "finish", "arguments": {}})
    assert done is True and obs.metadata["done"] is True


def test_max_steps_forces_done():
    h = _harness(max_steps=2)
    h.reset("t")
    _, _, done1, _ = h.step({"name": "echo", "arguments": {"text": "a"}})
    _, _, done2, _ = h.step({"name": "echo", "arguments": {"text": "b"}})
    assert done1 is False and done2 is True


def test_action_space_returns_injected_specs():
    assert _harness().action_space() == _space()


def test_action_space_prefers_env_method_when_present():
    class EnvWithSpace(FakeEnv):
        def action_space(self):
            return [{"name": "native", "arguments": {}}]

    h = OpenEnvHarness(EnvWithSpace(), action_space=_space())
    assert h.action_space() == [{"name": "native", "arguments": {}}]


# ---------------------------------------------------------------------------
# Conformance — snapshot=False path must report sensibly (passes cleanly)
# ---------------------------------------------------------------------------


def test_conformance_passes_and_reports_no_drift():
    rep = run_conformance(_harness())
    assert rep.passed is True
    assert rep.drift == []


# ---------------------------------------------------------------------------
# EventLog projection + replay round-trip
# ---------------------------------------------------------------------------


def test_to_eventlog_is_light_tier():
    h = _harness()
    h.reset("t")
    h.step({"name": "echo", "arguments": {"text": "x"}})
    log = h.to_eventlog()
    assert isinstance(log, EventLog)
    assert log.tier == "light"


def test_replay_reexecutes_tool_calls_through_env():
    src = EventLog(
        events=[
            Event(EventKind.OBSERVATION, {"text": "start"}),
            Event(EventKind.TOOL_CALL, {"action": {"name": "echo", "arguments": {"text": "a"}}}),
        ]
    )
    out = replay(_harness(), src)
    outs = [e.payload["output"] for e in out if e.kind is EventKind.TOOL_RESULT]
    assert "a" in outs


def test_export():
    from agenttune.agentic import OpenEnvHarness as OEH

    assert OEH is OpenEnvHarness
