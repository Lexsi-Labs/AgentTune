"""Advanced agent strategies — Plan-Execute and Reflexion (later-phase policies).

Both follow the same ``AgentStrategy`` interface as ``ReActStrategy`` (init ->
propose -> observe -> is_done) and keep model access behind an injected ``policy``
callable, so they are testable without a model. Additive: nothing existing is edited.

  - ``PlanExecuteStrategy`` — Plan-and-Solve (arXiv:2305.04091): the policy first
    emits a plan (a list of step actions), which is then executed in order. The plan
    lives in ``AgentState.scratch``.
  - ``ReflexionStrategy`` — Reflexion (arXiv:2303.11366): after a failed/low-reward
    attempt, a verbal self-reflection is generated (via the injected ``reflect``
    callable) and carried into the next attempt's context, up to N attempts.
"""

from __future__ import annotations

from collections.abc import Callable

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.harness import Harness, Observation
from agenttune.agentic.strategy import AgentState, AgentStrategy, run_episode


class PlanExecuteStrategy(AgentStrategy):
    """Plan-and-Solve: plan once up front, then execute the steps in order.

    ``policy`` is called a single time in ``init`` and returns the plan as a list of
    action dicts; ``propose`` hands back the step under the plan cursor and ``observe``
    advances it, marking the episode done once the plan is exhausted.
    """

    def __init__(self, policy: Callable[[AgentState], list], max_steps: int = 10):
        self.policy = policy
        self.max_steps = max_steps

    def init(self, task: str, tools) -> AgentState:
        state = AgentState(
            task=task,
            events=[Event(EventKind.OBSERVATION, {"text": task})],
            scratch={"tools": tools},
        )
        plan = list(self.policy(state) or [])
        state.scratch["plan"] = plan
        state.scratch["cursor"] = 0
        if not plan:
            state.done = True
        return state

    def propose(self, state: AgentState):
        plan = state.scratch.get("plan", [])
        cursor = state.scratch.get("cursor", 0)
        if cursor >= len(plan):
            return {"name": "finish", "arguments": {}}
        return plan[cursor]

    def observe(self, state: AgentState, observation: Observation) -> AgentState:
        state.events.append(Event(EventKind.OBSERVATION, {"text": observation.text}))
        state.scratch["cursor"] = state.scratch.get("cursor", 0) + 1
        state.step += 1
        if (
            state.scratch["cursor"] >= len(state.scratch.get("plan", []))
            or state.step >= self.max_steps
        ):
            state.done = True
        return state


class ReflexionStrategy(AgentStrategy):
    """Reflexion: retry a task up to ``max_attempts`` times, each attempt seeded with
    the verbal self-reflections accumulated from prior failed attempts.

    ``policy`` is the actor (state -> action, as in ReAct); ``reflect`` turns a failed
    trajectory (an ``EventLog``) into a short verbal reflection string. Reflections
    persist on ``self.memory`` and are injected into every fresh ``init`` — both as
    OBSERVATION events and under ``state.scratch['reflections']`` so the actor can
    condition on them.
    """

    def __init__(
        self,
        policy: Callable[[AgentState], dict],
        reflect: Callable[[EventLog], str],
        max_attempts: int = 3,
        max_steps: int = 10,
        reward_threshold: float = 1.0,
    ):
        self.policy = policy
        self.reflect = reflect
        self.max_attempts = max_attempts
        self.max_steps = max_steps
        self.reward_threshold = reward_threshold
        self.memory: list[str] = []

    def init(self, task: str, tools) -> AgentState:
        events = [Event(EventKind.OBSERVATION, {"text": task})]
        for r in self.memory:
            events.append(Event(EventKind.OBSERVATION, {"text": f"Reflection: {r}"}))
        return AgentState(
            task=task, events=events, scratch={"tools": tools, "reflections": list(self.memory)}
        )

    def propose(self, state: AgentState):
        return self.policy(state)

    def observe(self, state: AgentState, observation: Observation) -> AgentState:
        state.events.append(Event(EventKind.OBSERVATION, {"text": observation.text}))
        state.step += 1
        if state.step >= self.max_steps:
            state.done = True
        return state


def run_reflexion(
    strategy: ReflexionStrategy, harness: Harness, task: str, reward_fn: Callable[[EventLog], float]
) -> list[EventLog]:
    """Drive a Reflexion trial loop: run an episode, score it with ``reward_fn``, and on
    a below-threshold result append a verbal reflection before retrying. Returns the
    per-attempt ``EventLog``s; stops early once an attempt clears ``reward_threshold``.
    """
    attempts: list[EventLog] = []
    for _ in range(strategy.max_attempts):
        log = run_episode(strategy, harness, task)
        attempts.append(log)
        if reward_fn(log) >= strategy.reward_threshold:
            break
        strategy.memory.append(strategy.reflect(log))
    return attempts
