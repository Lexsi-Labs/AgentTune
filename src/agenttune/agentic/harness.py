"""Harness abstraction — the agent's environment / RL env (Phase 2 of the spine).

Gym-like: reset/step/action_space, capability flags with a DRIFT conformance bench,
and replay(). Produces Phase-1 EventLogs. Pure-Python reference impl; OpenEnvHarness
(wrapping the OpenEnv adapter) lands in Phase 2b.
"""

from __future__ import annotations

import copy
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agenttune.agentic.events import Event, EventKind, EventLog


@dataclass
class Observation:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class HarnessCapabilities:
    supports_snapshot: bool = False
    supports_streaming: bool = False
    supports_tool_boundary_interrupt: bool = False
    supports_stepwise_turns: bool = False
    max_steps: int | None = None


class Harness(ABC):
    capabilities: HarnessCapabilities
    event_log: list[Event]

    @abstractmethod
    def reset(self, task: str) -> Observation: ...

    @abstractmethod
    def step(self, action: dict) -> tuple[Observation, float, bool, dict]: ...

    @abstractmethod
    def action_space(self) -> list[dict]: ...

    def render(self) -> str:
        parts: list[str] = []
        for e in self.event_log:
            if e.kind in (EventKind.TEXT, EventKind.REASONING, EventKind.OBSERVATION):
                parts.append(str(e.payload.get("text", "")))
            elif e.kind is EventKind.TOOL_CALL:
                parts.append(f"CALL {e.payload.get('action', e.payload)}")
            elif e.kind is EventKind.TOOL_RESULT:
                parts.append(f"RESULT {e.payload.get('output')}")
        return "\n".join(p for p in parts if p)

    def to_eventlog(self) -> EventLog:
        tier = "full" if any(e.token_span is not None for e in self.event_log) else "light"
        return EventLog(events=list(self.event_log), tier=tier)

    def snapshot(self) -> bytes:
        raise NotImplementedError

    def restore(self, blob: bytes) -> None:
        raise NotImplementedError


class DictToolHarness(Harness):
    def __init__(
        self,
        tools: dict[str, Callable[..., Any]],
        *,
        max_steps: int = 10,
        done_tool: str = "finish",
    ):
        self.tools = tools
        self.done_tool = done_tool
        self.capabilities = HarnessCapabilities(
            supports_snapshot=True, supports_stepwise_turns=True, max_steps=max_steps
        )
        self.event_log: list[Event] = []
        self._task = ""
        self._steps = 0

    def reset(self, task: str) -> Observation:
        self.event_log = []
        self._task = task
        self._steps = 0
        self.event_log.append(Event(EventKind.OBSERVATION, {"text": task}))
        return Observation(text=task)

    def step(self, action: dict) -> tuple[Observation, float, bool, dict]:
        name = action.get("name")
        args = action.get("arguments", {}) or {}
        self.event_log.append(Event(EventKind.TOOL_CALL, {"action": action}))
        self._steps += 1
        if name == self.done_tool:
            out = args.get("answer", "")
            self.event_log.append(Event(EventKind.TOOL_RESULT, {"output": out}))
            self.event_log.append(Event(EventKind.TURN_COMPLETE, {"step": self._steps}))
            return (
                Observation(text=str(out), metadata={"done": True}),
                0.0,
                True,
                {"steps": self._steps},
            )
        fn = self.tools.get(name)
        if fn is None:
            out: Any = f"error: unknown tool {name!r}"
            reward = -1.0
        else:
            try:
                out = fn(**args)
                reward = 0.0
            except Exception as exc:  # noqa: BLE001 — surface tool errors as observations
                out = f"error: {exc}"
                reward = -1.0
        self.event_log.append(Event(EventKind.TOOL_RESULT, {"output": out}))
        self.event_log.append(Event(EventKind.TURN_COMPLETE, {"step": self._steps}))
        done = (
            self.capabilities.max_steps is not None and self._steps >= self.capabilities.max_steps
        )
        return Observation(text=str(out)), reward, done, {"steps": self._steps}

    def action_space(self) -> list[dict]:
        specs = [{"name": n, "arguments": {}} for n in self.tools]
        specs.append({"name": self.done_tool, "arguments": {"answer": ""}})
        return specs

    # snapshot/restore are for in-process tree-search backtracking: the blob is produced
    # by snapshot() and consumed by restore() on the same harness within one run. It is
    # never persisted to disk or loaded from an untrusted/external source, so pickle is safe
    # here. (A schema-validated serializer would be required if snapshots ever crossed a
    # trust boundary.)
    def snapshot(self) -> bytes:
        return pickle.dumps((copy.deepcopy(self.event_log), self._task, self._steps))

    def restore(self, blob: bytes) -> None:
        self.event_log, self._task, self._steps = pickle.loads(
            blob
        )  # noqa: S301 — trusted in-process blob


@dataclass
class ConformanceReport:
    passed: bool
    drift: list[str]


def run_conformance(harness: Harness, task: str = "conformance-probe") -> ConformanceReport:
    drift: list[str] = []
    harness.reset(task)
    caps = harness.capabilities
    if caps.supports_snapshot:
        try:
            blob = harness.snapshot()
            before = len(harness.event_log)
            space = harness.action_space()
            probe = space[0] if space else {"name": "noop", "arguments": {}}
            harness.step(probe)
            harness.restore(blob)
            if len(harness.event_log) != before:
                drift.append("declared supports_snapshot but restore() did not revert state")
        except NotImplementedError:
            drift.append("declared supports_snapshot but snapshot()/restore() is NotImplemented")
    else:
        try:
            harness.snapshot()
            drift.append("supports_snapshot=False but snapshot() succeeded (undeclared capability)")
        except NotImplementedError:
            pass
    return ConformanceReport(passed=not drift, drift=drift)


def replay(harness: Harness, log: EventLog) -> EventLog:
    """Re-execute the TOOL_CALL events of `log` through `harness`; return a fresh EventLog."""
    start = next((e.payload.get("text", "") for e in log if e.kind is EventKind.OBSERVATION), "")
    harness.reset(start)
    for e in log:
        if e.kind is EventKind.TOOL_CALL and isinstance(e.payload.get("action"), dict):
            harness.step(e.payload["action"])
    return harness.to_eventlog()
