"""Agent strategies — the policy layer (Phase 3 of the spine).

Every design reduces to init -> propose -> observe -> is_done. ReActStrategy is the
reference; Plan-Execute / Reflexion / Tree-Search follow the same interface (later
phases). Model access is an injected `policy` callable so strategies are testable
without a model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.harness import Harness, Observation


@dataclass
class AgentState:
    task: str
    events: list[Event]
    step: int = 0
    done: bool = False
    scratch: dict = field(default_factory=dict)


class AgentStrategy(ABC):
    @abstractmethod
    def init(self, task: str, tools) -> AgentState: ...

    @abstractmethod
    def propose(self, state: AgentState):  # -> dict | list[dict]
        ...

    @abstractmethod
    def observe(self, state: AgentState, observation: Observation) -> AgentState: ...

    def is_done(self, state: AgentState) -> bool:
        return state.done


class ReActStrategy(AgentStrategy):
    def __init__(self, policy: Callable[[AgentState], dict], max_steps: int = 10):
        self.policy = policy
        self.max_steps = max_steps

    def init(self, task: str, tools) -> AgentState:
        return AgentState(
            task=task,
            events=[Event(EventKind.OBSERVATION, {"text": task})],
            scratch={"tools": tools},
        )

    def propose(self, state: AgentState):
        return self.policy(state)

    def observe(self, state: AgentState, observation: Observation) -> AgentState:
        state.events.append(Event(EventKind.OBSERVATION, {"text": observation.text}))
        state.step += 1
        if state.step >= self.max_steps:
            state.done = True
        return state


def run_episode(strategy: AgentStrategy, harness: Harness, task: str) -> EventLog:
    tools = {s["name"] for s in harness.action_space()}
    state = strategy.init(task, tools)
    harness.reset(task)
    while not strategy.is_done(state):
        action = strategy.propose(state)
        if isinstance(action, list):
            action = action[0] if action else {"name": "finish", "arguments": {}}
        thought = action.get("thought")
        if thought:
            harness.event_log.append(Event(EventKind.REASONING, {"text": thought}))
        observation, reward, done, info = harness.step(action)
        state = strategy.observe(state, observation)
        if done:
            state.done = True
    return harness.to_eventlog()
