"""
Case study 5 — Compare all five agent designs on one task.
==========================================================

The agent-design pillar ships five strategies. This runs each on the same task with a canned,
model-free policy and prints how each reaches the answer — showing they are interchangeable
behind one `run_episode` / `EventLog` interface. GPU-free.

Run:  python examples/strategy_comparison.py
"""

from agenttune.agentic import (
    DictToolHarness,
    InContextMemory,
    MemoryReActStrategy,
    PlanExecuteStrategy,
    ReActStrategy,
    ReflexionStrategy,
    TreeOfThoughtsStrategy,
    run_episode,
)


def react_policy(state):
    if state.step == 0:
        return {"name": "search", "arguments": {"q": "answer"}, "thought": "look it up"}
    return {"name": "finish", "arguments": {"answer": "42"}}


def build(name):
    """Return a (strategy, task) for one design, all reaching the answer '42'."""
    if name == "react":
        return ReActStrategy(react_policy, max_steps=5)
    if name == "plan_execute":
        plan = [
            {"name": "search", "arguments": {"q": "answer"}, "thought": "step 1"},
            {"name": "finish", "arguments": {"answer": "42"}},
        ]
        return PlanExecuteStrategy(lambda s: list(plan), max_steps=5)
    if name == "reflexion":
        return ReflexionStrategy(
            policy=lambda s: {"name": "finish", "arguments": {"answer": "42"}},
            reflect=lambda log: "be more direct",
            max_attempts=1,
            max_steps=5,
        )
    if name == "tot":
        cands = [
            {"name": "search", "arguments": {"q": "answer"}},
            {"name": "finish", "arguments": {"answer": "42"}, "thought": "commit"},
        ]
        return TreeOfThoughtsStrategy(
            propose_candidates=lambda s: cands,
            score=lambda s, c: 1.0 if c["name"] == "finish" else 0.3,
            beam_width=2,
            max_steps=5,
        )
    if name == "memory":
        return MemoryReActStrategy(policy=react_policy, memory=InContextMemory(), max_steps=5)
    raise KeyError(name)


def answer_of(log):
    from agenttune.agentic import EventKind

    for e in log:
        act = e.payload.get("action") if e.kind is EventKind.TOOL_CALL else None
        if isinstance(act, dict) and act.get("name") == "finish":
            return act.get("arguments", {}).get("answer")
    return None


def main():
    harness = DictToolHarness({"search": lambda q="": f"result for {q}"}, max_steps=5)
    print(f"{'design':<14}{'events':>8}{'answer':>9}")
    print("-" * 31)
    for name in ("react", "plan_execute", "reflexion", "tot", "memory"):
        log = run_episode(build(name), harness, "what is the answer?")
        print(f"{name:<14}{len(log):>8}{str(answer_of(log)):>9}")


if __name__ == "__main__":
    main()
