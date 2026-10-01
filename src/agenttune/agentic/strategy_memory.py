"""Memory-backed agent strategy — wires the BaseMemory pillar into the agent loop.

``MemoryReActStrategy`` is a ReAct-style policy (same init -> propose -> observe ->
is_done interface as ``ReActStrategy``, so it drops into ``run_episode`` unchanged)
that reads and writes an INJECTED ``BaseMemory`` across an episode:

  - ``init`` READS memory relevant to the task and seeds the recalled items into the
    fresh ``AgentState`` — both as OBSERVATION events (for context building) and under
    ``state.scratch['recalled']`` (a plain list of item contents) so the injected
    ``policy`` can condition on recalled memory directly.
  - ``observe`` WRITES each new observation into memory as an EPISODIC ``MemoryItem``,
    so a run's experience persists and is recallable by later episodes.

Model access stays behind the injected ``policy`` callable and the memory driver is
injected, so the strategy is testable with ``InContextMemory`` and a scripted policy —
no model / GPU / network. Additive: nothing existing is edited.
"""

from __future__ import annotations

from collections.abc import Callable

from agenttune.agentic.events import Event, EventKind
from agenttune.agentic.harness import Observation
from agenttune.agentic.memory import BaseMemory, MemoryItem, MemoryKind, Scope
from agenttune.agentic.strategy import AgentState, AgentStrategy


class MemoryReActStrategy(AgentStrategy):
    """ReAct with a persistent, injected memory across the episode.

    Args:
        policy: actor callable ``state -> action`` (as in ``ReActStrategy``).
        memory: injected ``BaseMemory`` driver read on ``init`` and written on ``observe``.
        scope: memory scope used for both reads and writes (per-agent / shared pool).
        recall_k: number of most-relevant items to recall on ``init``.
        recall_kind: optional ``MemoryKind`` filter for recall (``None`` = any kind).
        max_steps: step cap, matching ``ReActStrategy``.
    """

    def __init__(
        self,
        policy: Callable[[AgentState], dict],
        memory: BaseMemory,
        *,
        scope: Scope | None = None,
        recall_k: int = 5,
        recall_kind: MemoryKind | None = None,
        max_steps: int = 10,
    ):
        self.policy = policy
        self.memory = memory
        self.scope = scope or Scope()
        self.recall_k = recall_k
        self.recall_kind = recall_kind
        self.max_steps = max_steps

    def init(self, task: str, tools) -> AgentState:
        recalled = self.memory.read(task, k=self.recall_k, scope=self.scope, kind=self.recall_kind)
        events = [Event(EventKind.OBSERVATION, {"text": task})]
        for item in recalled:
            events.append(Event(EventKind.OBSERVATION, {"text": f"Recalled: {item.content}"}))
        return AgentState(
            task=task,
            events=events,
            scratch={"tools": tools, "recalled": [item.content for item in recalled]},
        )

    def propose(self, state: AgentState):
        return self.policy(state)

    def observe(self, state: AgentState, observation: Observation) -> AgentState:
        state.events.append(Event(EventKind.OBSERVATION, {"text": observation.text}))
        item_id = self.memory.write(
            MemoryItem(
                content=observation.text,
                kind=MemoryKind.EPISODIC,
                scope=self.scope,
                metadata={"step": state.step},
            ),
            scope=self.scope,
        )
        state.events.append(
            Event(
                EventKind.MEMORY_OP, {"op": "ADD", "id": item_id, "kind": MemoryKind.EPISODIC.value}
            )
        )
        state.step += 1
        if state.step >= self.max_steps:
            state.done = True
        return state
