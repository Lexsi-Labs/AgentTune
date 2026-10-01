from agenttune.agentic.events import EventKind
from agenttune.agentic.harness import DictToolHarness, Observation
from agenttune.agentic.memory import InContextMemory, MemoryItem, MemoryKind, Scope
from agenttune.agentic.strategy import run_episode
from agenttune.agentic.strategy_memory import MemoryReActStrategy

# ---------- recall on init ----------


def test_init_recalls_prior_memory_into_scratch_and_events():
    scope = Scope(agent_id="a", namespace="n")
    mem = InContextMemory()
    mem.write(
        MemoryItem(content="paris is the capital of france", kind=MemoryKind.SEMANTIC, scope=scope)
    )
    strat = MemoryReActStrategy(policy=lambda s: {}, memory=mem, scope=scope)

    st = strat.init("what is the capital of france", tools=set())

    # recalled memory is visible to the policy under scratch['recalled']
    assert "paris is the capital of france" in st.scratch["recalled"]
    # ... and surfaced as OBSERVATION events for context building
    recalled_texts = [
        e.payload.get("text", "") for e in st.events if e.kind is EventKind.OBSERVATION
    ]
    assert any("paris is the capital of france" in t for t in recalled_texts)


def test_init_with_empty_memory_recalls_nothing():
    strat = MemoryReActStrategy(policy=lambda s: {}, memory=InContextMemory())
    st = strat.init("t", tools=set())
    assert st.scratch["recalled"] == []


# ---------- write on observe ----------


def test_observe_writes_observation_into_memory():
    scope = Scope(agent_id="a", namespace="n")
    mem = InContextMemory()
    strat = MemoryReActStrategy(policy=lambda s: {}, memory=mem, scope=scope)
    st = strat.init("t", tools=set())

    before = len(mem.read(scope=scope, k=100))
    st = strat.observe(st, Observation(text="observed-42"))

    after = mem.read(scope=scope, k=100)
    assert len(after) == before + 1
    assert any(i.content == "observed-42" and i.kind is MemoryKind.EPISODIC for i in after)


def test_observe_advances_and_caps_like_react():
    strat = MemoryReActStrategy(policy=lambda s: {}, memory=InContextMemory(), max_steps=2)
    st = strat.init("t", tools=set())
    st = strat.observe(st, Observation(text="o1"))
    assert st.step == 1 and st.done is False
    st = strat.observe(st, Observation(text="o2"))
    assert st.done is True


# ---------- discriminating end-to-end test ----------


def test_prior_knowledge_recalled_and_run_written_back():
    """Knowledge written into the injected memory BEFORE the episode is recalled and
    made visible to the policy; observations produced by the run are written back."""
    scope = Scope(agent_id="a", namespace="shared")
    mem = InContextMemory()
    # prior knowledge, seeded before the episode runs
    mem.write(MemoryItem(content="prior-hint: prefer echo", kind=MemoryKind.SEMANTIC, scope=scope))

    seen_recalled: list[list] = []
    script = iter(
        [
            {"name": "echo", "arguments": {"text": "a"}, "thought": "echo first"},
            {"name": "finish", "arguments": {"answer": "done"}},
        ]
    )

    def policy(state):
        # capture exactly what the policy can see of recalled memory
        seen_recalled.append(list(state.scratch.get("recalled", [])))
        return next(script)

    strat = MemoryReActStrategy(policy=policy, memory=mem, scope=scope, max_steps=5)
    h = DictToolHarness({"echo": lambda text="": f"echo:{text}"})

    log = run_episode(strat, h, "please echo then finish")

    # (1) the policy saw the recalled prior knowledge in the AgentState
    assert seen_recalled, "policy was never called"
    assert "prior-hint: prefer echo" in seen_recalled[0]

    # (2) the run's observations were written back into memory (episodic growth)
    episodic = [i.content for i in mem.read(scope=scope, k=100, kind=MemoryKind.EPISODIC)]
    assert "echo:a" in episodic
    assert "done" in episodic

    # sanity: the episode still ran to a normal finish
    outs = [e.payload.get("output") for e in log if e.kind is EventKind.TOOL_RESULT]
    assert "echo:a" in outs and "done" in outs


def test_export():
    from agenttune.agentic import MemoryReActStrategy as M

    assert M is MemoryReActStrategy
