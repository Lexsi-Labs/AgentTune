from agenttune.agentic.events import EventKind
from agenttune.agentic.harness import DictToolHarness, Observation
from agenttune.agentic.strategy import AgentState, run_episode
from agenttune.agentic.strategy_tree import TreeOfThoughtsStrategy

# ---------- candidate scoring / selection ----------


def _candidates():
    return [
        {"name": "echo", "arguments": {"text": "a"}},
        {"name": "echo", "arguments": {"text": "b"}},
        {"name": "echo", "arguments": {"text": "c"}},
    ]


def _score_by_text(scores):
    """Score a candidate by its arguments['text'] via a lookup table."""
    return lambda state, cand: scores[cand["arguments"]["text"]]


def test_propose_selects_highest_scored_candidate():
    cands = _candidates()
    scores = {"a": 0.1, "b": 0.9, "c": 0.4}  # 'b' is best
    strat = TreeOfThoughtsStrategy(propose_candidates=lambda s: cands, score=_score_by_text(scores))
    st = strat.init("task", tools={"echo"})
    action = strat.propose(st)
    assert action == {"name": "echo", "arguments": {"text": "b"}}


def test_default_beam_is_greedy_best_single_frontier_entry():
    cands = _candidates()
    scores = {"a": 0.1, "b": 0.9, "c": 0.4}
    strat = TreeOfThoughtsStrategy(
        propose_candidates=lambda s: cands, score=_score_by_text(scores)
    )  # beam_width=1 default
    st = strat.init("task", tools={"echo"})
    strat.propose(st)
    frontier = st.scratch["frontier"]
    assert len(frontier) == 1
    assert frontier[0]["candidate"]["arguments"]["text"] == "b"
    assert frontier[0]["score"] == 0.9


def test_wider_beam_keeps_top_k_in_scratch_sorted():
    cands = _candidates()
    scores = {"a": 0.1, "b": 0.9, "c": 0.4}
    strat = TreeOfThoughtsStrategy(
        propose_candidates=lambda s: cands, score=_score_by_text(scores), beam_width=2
    )
    st = strat.init("task", tools={"echo"})
    strat.propose(st)
    frontier = st.scratch["frontier"]
    assert len(frontier) == 2
    # sorted descending by score: b (0.9) then c (0.4)
    assert [f["candidate"]["arguments"]["text"] for f in frontier] == ["b", "c"]
    assert [f["score"] for f in frontier] == [0.9, 0.4]


def test_wider_beam_retains_more_frontier_entries_than_narrow():
    cands = _candidates()
    scores = {"a": 0.1, "b": 0.9, "c": 0.4}
    narrow = TreeOfThoughtsStrategy(
        propose_candidates=lambda s: cands, score=_score_by_text(scores), beam_width=1
    )
    wide = TreeOfThoughtsStrategy(
        propose_candidates=lambda s: cands, score=_score_by_text(scores), beam_width=3
    )
    sn = narrow.init("task", tools={"echo"})
    sw = wide.init("task", tools={"echo"})
    narrow.propose(sn)
    wide.propose(sw)
    assert len(sw.scratch["frontier"]) > len(sn.scratch["frontier"])
    assert len(sn.scratch["frontier"]) == 1
    assert len(sw.scratch["frontier"]) == 3


def test_beam_wider_than_candidate_count_keeps_all():
    cands = _candidates()  # 3 candidates
    scores = {"a": 0.1, "b": 0.9, "c": 0.4}
    strat = TreeOfThoughtsStrategy(
        propose_candidates=lambda s: cands, score=_score_by_text(scores), beam_width=10
    )
    st = strat.init("task", tools={"echo"})
    strat.propose(st)
    assert len(st.scratch["frontier"]) == 3


def test_empty_candidates_falls_back_to_finish():
    strat = TreeOfThoughtsStrategy(propose_candidates=lambda s: [], score=lambda s, c: 0.0)
    st = strat.init("task", tools=set())
    action = strat.propose(st)
    assert action == {"name": "finish", "arguments": {}}
    assert st.scratch["frontier"] == []


# ---------- observe / termination ----------


def test_observe_advances_step_and_caps_at_max_steps():
    strat = TreeOfThoughtsStrategy(
        propose_candidates=lambda s: _candidates(), score=lambda s, c: 0.0, max_steps=2
    )
    st = strat.init("task", tools={"echo"})
    st = strat.observe(st, Observation(text="o1"))
    assert st.done is False and st.step == 1
    st = strat.observe(st, Observation(text="o2"))
    assert st.done is True and st.step == 2


# ---------- end-to-end via run_episode ----------


def test_run_episode_selects_best_and_terminates():
    # Each step: a low-scored echo vs a high-scored finish -> finish wins, episode ends.
    def propose_candidates(state):
        return [
            {"name": "echo", "arguments": {"text": "noise"}},
            {"name": "finish", "arguments": {"answer": "done"}, "thought": "wrap up"},
        ]

    def score(state, cand):
        return 1.0 if cand["name"] == "finish" else 0.0

    strat = TreeOfThoughtsStrategy(
        propose_candidates=propose_candidates, score=score, beam_width=2, max_steps=5
    )
    h = DictToolHarness({"echo": lambda text="": f"echo:{text}"})
    log = run_episode(strat, h, "search then answer")
    outs = [e.payload.get("output") for e in log if e.kind is EventKind.TOOL_RESULT]
    assert "done" in outs
    assert "echo:noise" not in outs  # the low-scored branch was never committed
    # reasoning from the selected candidate's thought made it into the log
    thoughts = [e.payload.get("text") for e in log if e.kind is EventKind.REASONING]
    assert "wrap up" in thoughts


def test_run_episode_greedy_beam_one_still_reaches_finish():
    def propose_candidates(state):
        return [
            {"name": "echo", "arguments": {"text": "x"}},
            {"name": "finish", "arguments": {"answer": "ok"}},
        ]

    strat = TreeOfThoughtsStrategy(
        propose_candidates=propose_candidates,
        score=lambda s, c: 1.0 if c["name"] == "finish" else 0.0,
        beam_width=1,
        max_steps=5,
    )
    h = DictToolHarness({"echo": lambda text="": f"echo:{text}"})
    log = run_episode(strat, h, "task")
    outs = [e.payload.get("output") for e in log if e.kind is EventKind.TOOL_RESULT]
    assert "ok" in outs


def test_export():
    from agenttune.agentic import TreeOfThoughtsStrategy as T

    assert T is TreeOfThoughtsStrategy


# ---------- real tree search with backtracking ----------
#
# A small graph the agent walks. The GREEDY-best first move ("A", score 0.9) leads
# into a dead end ("A2"); the goal is only reachable via the LOWER-scored sibling
# ("B", score 0.1). Injected callables (candidates/score/transition/is_goal) drive
# the search with no model.
#
#   root --A(0.9)--> A --A2(0.5)--> A2 (dead end, no candidates)
#   root --B(0.1)--> B --GOAL(0.7)--> GOAL (accept)

_GRAPH = {
    "root": [("A", 0.9), ("B", 0.1)],
    "A": [("A2", 0.5)],
    "A2": [],  # dead end: greedy gets stuck here
    "B": [("GOAL", 0.7)],
    "GOAL": [],
}


def _mv(to):
    return {"name": "move", "arguments": {"to": to}}


def _graph_candidates(state):
    node = state.scratch.get("node", "root")
    return [_mv(to) for to, _ in _GRAPH[node]]


def _graph_score(state, cand):
    node = state.scratch.get("node", "root")
    target = cand["arguments"]["to"]
    return dict(_GRAPH[node])[target]


def _graph_transition(state, action):
    # Fresh state per branch — never mutate the parent (backtracking safety).
    return AgentState(
        task=state.task,
        events=list(state.events),
        scratch={**state.scratch, "node": action["arguments"]["to"]},
    )


def _graph_is_goal(state):
    return state.scratch.get("node") == "GOAL"


def _new_strat(**kw):
    return TreeOfThoughtsStrategy(
        propose_candidates=_graph_candidates,
        score=_graph_score,
        transition=_graph_transition,
        is_goal=_graph_is_goal,
        **kw,
    )


def _greedy_walk(strat, task, max_depth=10):
    """Control: always take the top-scored candidate, never backtrack."""
    state = strat.init(task, tools={"move"})
    for _ in range(max_depth):
        if _graph_is_goal(state):
            return state, True
        cands = _graph_candidates(state)
        if not cands:
            return state, False  # stuck
        best = max(cands, key=lambda c: _graph_score(state, c))
        state = _graph_transition(state, best)
    return state, _graph_is_goal(state)


def test_search_backtracks_where_greedy_dead_ends():
    strat = _new_strat(max_depth=5, max_nodes=50)

    # Greedy control: the top-scored first move leads to a dead end, goal never reached.
    _, greedy_ok = _greedy_walk(strat, "reach goal")
    assert greedy_ok is False

    # Search backtracks from the A-subtree to the lower-scored B sibling and succeeds.
    result = strat.search("reach goal", tools={"move"})
    assert result["found"] is True
    path_nodes = [a["arguments"]["to"] for a in result["path"]]
    assert path_nodes == ["B", "GOAL"]  # took the low-scored branch to reach goal


def test_search_returns_action_path_to_goal():
    strat = _new_strat(max_depth=5, max_nodes=50)
    result = strat.search("reach goal", tools={"move"})
    assert result["found"] is True
    assert [a["name"] for a in result["path"]] == ["move", "move"]
    assert result["goal_state"].scratch["node"] == "GOAL"
    assert result["visited"] >= 1


def test_search_respects_node_budget():
    # Budget too small to ever expand out to GOAL -> cap must bite.
    strat = _new_strat(max_depth=10, max_nodes=2)
    result = strat.search("reach goal", tools={"move"})
    assert result["visited"] <= 2
    assert result["found"] is False  # cap prevented reaching the goal


def test_search_respects_depth_budget():
    # GOAL is at depth 2; capping depth at 1 makes it unreachable.
    strat = _new_strat(max_depth=1, max_nodes=50)
    result = strat.search("reach goal", tools={"move"})
    assert result["found"] is False
    assert all(len(p) <= 1 for p in [result["path"]])


def test_search_leaves_greedy_propose_unchanged():
    # search() must not disturb the greedy propose/observe path used by run_episode.
    cands = _candidates()
    scores = {"a": 0.1, "b": 0.9, "c": 0.4}
    strat = TreeOfThoughtsStrategy(propose_candidates=lambda s: cands, score=_score_by_text(scores))
    st = strat.init("task", tools={"echo"})
    action = strat.propose(st)
    assert action == {"name": "echo", "arguments": {"text": "b"}}
    assert st.scratch["frontier"][0]["candidate"]["arguments"]["text"] == "b"
