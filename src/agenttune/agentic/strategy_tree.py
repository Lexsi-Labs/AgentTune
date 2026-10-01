"""Search-based agent strategy — Tree-of-Thoughts / LATS (a later-phase policy).

Follows the same ``AgentStrategy`` interface as ``ReActStrategy`` (init -> propose ->
observe -> is_done) and keeps model access behind injected callables, so it is testable
without a model. Additive: nothing existing is edited.

  - ``TreeOfThoughtsStrategy`` — Tree-of-Thoughts (arXiv:2305.10601) / LATS
    (arXiv:2310.04406): at each step the agent enumerates several candidate
    thoughts/actions, scores each, and expands the best via a beam/greedy search over
    the thought tree. Two callables are injected:

      * ``propose_candidates(state) -> list[dict]`` — several candidate action dicts;
      * ``score(state, candidate) -> float`` — a value/heuristic for one candidate.

    ``propose`` scores every candidate, retains the top ``beam_width`` in
    ``AgentState.scratch['frontier']`` (each ``{"candidate": dict, "score": float}``,
    sorted descending), records the full scored expansion under ``scratch['tree']`` for
    traceability, and commits the single top candidate as the step's action. Because
    ``run_episode`` executes exactly one action per step, ``beam_width`` governs how much
    of the frontier is remembered, not which action fires (that is always the best one).
"""

from __future__ import annotations

import copy
import heapq
import itertools
from collections.abc import Callable

from agenttune.agentic.events import Event, EventKind
from agenttune.agentic.harness import Observation
from agenttune.agentic.strategy import AgentState, AgentStrategy


def _default_transition(state: AgentState, action: dict) -> AgentState:
    """Apply an action -> a fresh, independent next state (never mutate the parent).

    Deep-copies so sibling branches on the search frontier cannot alias each other;
    records the applied action under ``scratch['last_action']`` for the default goal
    test and advances ``step``.
    """
    nxt = copy.deepcopy(state)
    nxt.events.append(Event(EventKind.TOOL_CALL, dict(action)))
    nxt.scratch["last_action"] = action
    nxt.step = state.step + 1
    return nxt


def _default_is_goal(state: AgentState) -> bool:
    """Terminal test: the most recently applied action was ``finish``."""
    return (state.scratch.get("last_action") or {}).get("name") == "finish"


class TreeOfThoughtsStrategy(AgentStrategy):
    """Tree-of-Thoughts / LATS: propose many candidate actions per step, score them,
    and greedily expand the best (keeping the top ``beam_width`` on the frontier).

    ``propose_candidates`` and ``score`` are injected so the search is exercised without
    a model. ``beam_width`` defaults to 1 (greedy-best); a wider beam keeps more scored
    candidates in ``scratch['frontier']`` for inspection/backtracking, but the committed
    action is always the top-scored one.
    """

    def __init__(
        self,
        propose_candidates: Callable[[AgentState], list],
        score: Callable[[AgentState, dict], float],
        beam_width: int = 1,
        max_steps: int = 10,
        transition: Callable[[AgentState, dict], AgentState] | None = None,
        is_goal: Callable[[AgentState], bool] | None = None,
        max_depth: int = 10,
        max_nodes: int = 256,
    ):
        self.propose_candidates = propose_candidates
        self.score = score
        self.beam_width = beam_width
        self.max_steps = max_steps
        # Search-only knobs (additive; unused by the greedy propose/observe path).
        self.transition = transition or _default_transition
        self.is_goal = is_goal or _default_is_goal
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def init(self, task: str, tools) -> AgentState:
        return AgentState(
            task=task,
            events=[Event(EventKind.OBSERVATION, {"text": task})],
            scratch={"tools": tools, "frontier": [], "tree": []},
        )

    def propose(self, state: AgentState):
        candidates = list(self.propose_candidates(state) or [])
        scored = [{"candidate": c, "score": self.score(state, c)} for c in candidates]
        scored.sort(key=lambda e: e["score"], reverse=True)
        frontier = scored[: self.beam_width]
        state.scratch["frontier"] = frontier
        state.scratch.setdefault("tree", []).append(scored)
        if not frontier:
            return {"name": "finish", "arguments": {}}
        return frontier[0]["candidate"]

    def observe(self, state: AgentState, observation: Observation) -> AgentState:
        state.events.append(Event(EventKind.OBSERVATION, {"text": observation.text}))
        state.step += 1
        if state.step >= self.max_steps:
            state.done = True
        return state

    # ------------------------------------------------------------------ search
    def search(self, task: str, tools=None) -> dict:
        """Real best-first tree search over the thought tree, WITH backtracking.

        Additive and self-contained: unlike ``propose`` (which commits the single
        top-scored candidate per ``run_episode`` step and so is effectively greedy),
        ``search`` explores the whole tree. It keeps a priority frontier of *every*
        scored branch — not just the current best — so when the greedy-best path
        dead-ends or scores poorly, the next ``heappop`` naturally BACKTRACKS to a
        lower-scored sibling from an earlier expansion.

        Uses the injected ``propose_candidates``/``score``/``transition``/``is_goal``
        (no model). A node's search priority is its own candidate ``score`` (matching
        the ``{"candidate", "score"}`` frontier shape used by ``propose``).

        Bounded by ``max_depth`` (path length) and ``max_nodes`` (expansion budget).

        Returns a dict::

            {"found": bool,            # goal reached within budget
             "path": [action, ...],    # best action path found (to goal if found)
             "goal_state": AgentState | None,
             "visited": int,           # nodes expanded
             "tree": [ ... ]}          # per-expansion scored candidate records
        """
        root = self.init(task, tools)
        counter = itertools.count()
        # Frontier entries: (-priority, tie, state, path). Max-priority pops first.
        frontier: list = [(0.0, next(counter), root, [])]
        tree: list = []
        visited = 0
        # Best partial path seen (deepest / highest-priority), returned if no goal.
        best_partial: tuple = (float("-inf"), 0, [])

        while frontier and visited < self.max_nodes:
            neg_prio, _, state, path = heapq.heappop(frontier)
            priority = -neg_prio

            if self.is_goal(state):
                return {
                    "found": True,
                    "path": path,
                    "goal_state": state,
                    "visited": visited,
                    "tree": tree,
                }

            visited += 1
            if (priority, len(path)) > (best_partial[0], best_partial[1]):
                best_partial = (priority, len(path), path)

            if len(path) >= self.max_depth:
                continue  # depth budget hit on this branch — backtrack via frontier

            candidates = list(self.propose_candidates(state) or [])
            scored = [{"candidate": c, "score": self.score(state, c)} for c in candidates]
            scored.sort(key=lambda e: e["score"], reverse=True)
            tree.append({"node_path": path, "expansion": scored})
            for entry in scored:
                child = self.transition(state, entry["candidate"])
                heapq.heappush(
                    frontier, (-entry["score"], next(counter), child, path + [entry["candidate"]])
                )

        return {
            "found": False,
            "path": best_partial[2],
            "goal_state": None,
            "visited": visited,
            "tree": tree,
        }
