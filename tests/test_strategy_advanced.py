from agenttune.agentic.events import EventKind, EventLog
from agenttune.agentic.harness import DictToolHarness, Observation
from agenttune.agentic.strategy import run_episode
from agenttune.agentic.strategy_advanced import (
    PlanExecuteStrategy,
    ReflexionStrategy,
    run_reflexion,
)

# ---------- PlanExecuteStrategy ----------


def test_plan_execute_produces_plan_in_init():
    plan = [
        {"name": "echo", "arguments": {"text": "a"}},
        {"name": "echo", "arguments": {"text": "b"}},
    ]
    strat = PlanExecuteStrategy(policy=lambda state: plan)
    st = strat.init("task", tools={"echo"})
    assert st.scratch["plan"] == plan and st.scratch["cursor"] == 0


def test_plan_execute_consumes_plan_in_order():
    plan = [
        {"name": "echo", "arguments": {"text": "a"}},
        {"name": "echo", "arguments": {"text": "b"}},
    ]
    strat = PlanExecuteStrategy(policy=lambda state: plan)
    st = strat.init("task", tools={"echo"})
    assert strat.propose(st) == plan[0]
    st = strat.observe(st, Observation(text="o1"))
    assert st.done is False
    assert strat.propose(st) == plan[1]
    st = strat.observe(st, Observation(text="o2"))
    assert st.done is True  # plan exhausted


def test_plan_execute_empty_plan_is_done():
    strat = PlanExecuteStrategy(policy=lambda state: [])
    st = strat.init("task", tools=set())
    assert st.done is True


def test_plan_execute_run_episode_executes_each_step():
    plan = [
        {"name": "echo", "arguments": {"text": "a"}, "thought": "step 1"},
        {"name": "finish", "arguments": {"answer": "done"}},
    ]
    # policy called once (in init); returns the whole plan.
    strat = PlanExecuteStrategy(policy=lambda state: list(plan), max_steps=5)
    h = DictToolHarness({"echo": lambda text="": f"echo:{text}"})
    log = run_episode(strat, h, "plan then act")
    outs = [e.payload.get("output") for e in log if e.kind is EventKind.TOOL_RESULT]
    assert "echo:a" in outs and "done" in outs


# ---------- ReflexionStrategy ----------


def _finish_reward(expected: str):
    """Reward from the finish output only (DictToolHarness gives finish reward 0.0
    regardless of the answer, so success must be judged from the returned answer)."""

    def _fn(log: EventLog) -> float:
        outs = [e.payload.get("output") for e in log if e.kind is EventKind.TOOL_RESULT]
        return 1.0 if expected in outs else 0.0

    return _fn


def test_reflexion_init_injects_accumulated_reflections():
    strat = ReflexionStrategy(policy=lambda s: {}, reflect=lambda log: "r")
    strat.memory.append("try the other tool")
    st = strat.init("task", tools=set())
    assert st.scratch["reflections"] == ["try the other tool"]
    texts = [e.payload.get("text") for e in st.events]
    assert any("try the other tool" in (t or "") for t in texts)


def test_reflexion_reflects_after_poor_attempt_and_retries():
    # Actor conditions on whether a reflection is present: without one it answers
    # "wrong"; once a reflection has been fed back it answers "right".
    def actor(state):
        if state.scratch.get("reflections"):
            return {"name": "finish", "arguments": {"answer": "right"}}
        return {"name": "finish", "arguments": {"answer": "wrong"}}

    strat = ReflexionStrategy(
        policy=actor,
        reflect=lambda log: "answer 'wrong' failed; try 'right'",
        max_attempts=3,
        reward_threshold=1.0,
    )
    h = DictToolHarness({}, max_steps=5)
    logs = run_reflexion(strat, h, "give the right answer", reward_fn=_finish_reward("right"))

    # Two attempts: first fails, reflection generated, second succeeds and stops.
    assert len(logs) == 2
    assert len(strat.memory) == 1
    assert "try 'right'" in strat.memory[0]
    first_outs = [e.payload.get("output") for e in logs[0] if e.kind is EventKind.TOOL_RESULT]
    second_outs = [e.payload.get("output") for e in logs[1] if e.kind is EventKind.TOOL_RESULT]
    assert "wrong" in first_outs
    assert "right" in second_outs


def test_reflexion_stops_after_max_attempts_without_success():
    strat = ReflexionStrategy(
        policy=lambda s: {"name": "finish", "arguments": {"answer": "no"}},
        reflect=lambda log: "still failing",
        max_attempts=2,
        reward_threshold=1.0,
    )
    h = DictToolHarness({}, max_steps=5)
    logs = run_reflexion(strat, h, "task", reward_fn=_finish_reward("yes"))
    assert len(logs) == 2  # capped at max_attempts
    assert len(strat.memory) == 2  # a reflection stored per failed attempt


def test_exports():
    from agenttune.agentic import (
        PlanExecuteStrategy as PE,
    )
    from agenttune.agentic import (
        ReflexionStrategy as RX,
    )
    from agenttune.agentic import (
        run_reflexion as rr,
    )

    assert PE is PlanExecuteStrategy and RX is ReflexionStrategy and rr is run_reflexion
