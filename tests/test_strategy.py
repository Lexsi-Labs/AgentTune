import pytest

from agenttune.agentic.events import EventKind
from agenttune.agentic.harness import DictToolHarness, Observation
from agenttune.agentic.strategy import AgentState, AgentStrategy, ReActStrategy, run_episode

# ---------- Task 1 ----------


def test_agentstate_defaults():
    s = AgentState(task="t", events=[])
    assert s.step == 0 and s.done is False and s.scratch == {}


def test_strategy_is_abstract():
    with pytest.raises(TypeError):
        AgentStrategy()


# ---------- Task 2 ----------


def test_react_proposes_from_policy():
    calls = [{"name": "echo", "arguments": {"text": "hi"}, "thought": "let me echo"}]
    strat = ReActStrategy(policy=lambda state: calls[0])
    st = strat.init("task", tools={"echo"})
    assert strat.propose(st) == calls[0]


def test_react_observe_advances_and_caps():
    strat = ReActStrategy(policy=lambda s: {}, max_steps=2)
    st = strat.init("t", tools=set())
    st = strat.observe(st, Observation(text="o1"))
    assert st.step == 1 and st.done is False
    st = strat.observe(st, Observation(text="o2"))
    assert st.done is True


# ---------- Task 3 ----------


def test_run_episode_echo_then_finish():
    script = iter(
        [
            {"name": "echo", "arguments": {"text": "a"}, "thought": "echo first"},
            {"name": "finish", "arguments": {"answer": "done"}},
        ]
    )
    strat = ReActStrategy(policy=lambda state: next(script), max_steps=5)
    h = DictToolHarness({"echo": lambda text="": f"echo:{text}"})
    log = run_episode(strat, h, "please echo then finish")
    kinds = [e.kind for e in log]
    assert EventKind.REASONING in kinds
    assert EventKind.TOOL_CALL in kinds
    outs = [e.payload.get("output") for e in log if e.kind is EventKind.TOOL_RESULT]
    assert "echo:a" in outs and "done" in outs


def test_run_episode_respects_harness_done():
    strat = ReActStrategy(
        policy=lambda state: {"name": "echo", "arguments": {"text": "x"}}, max_steps=99
    )
    h = DictToolHarness({"echo": lambda text="": "x"}, max_steps=3)
    log = run_episode(strat, h, "loop")
    calls = [e for e in log if e.kind is EventKind.TOOL_CALL]
    assert len(calls) == 3


# ---------- Task 4 ----------


def test_exports():
    from agenttune.agentic import AgentStrategy as A
    from agenttune.agentic import ReActStrategy as R
    from agenttune.agentic import run_episode as run

    assert A is AgentStrategy and R is ReActStrategy and run is run_episode
