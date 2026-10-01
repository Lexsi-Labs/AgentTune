"""
Case study 6 — Harness conformance and deterministic replay.
============================================================

The harness pillar is the agent's environment. Two guarantees a trainable env needs:
  1. conformance — the harness behaves as its declared capabilities claim (`run_conformance`).
  2. replay      — a recorded `EventLog`'s tool calls re-execute identically (`replay`).

Both are GPU-free and model-free.

Run:  python examples/conformance_and_replay.py
"""

from agenttune.agentic import DictToolHarness, ReActStrategy, replay, run_conformance, run_episode


def main():
    calls = {"n": 0}

    def add(a, b):
        calls["n"] += 1
        return a + b

    harness = DictToolHarness({"add": add}, max_steps=4)

    # 1) CONFORMANCE — does the harness match its declared capabilities?
    report = run_conformance(harness)
    print(f"[conformance] passed={report.passed}  drift={report.drift or 'none'}")

    # 2) Record an episode.
    def policy(state):
        if state.step == 0:
            return {"name": "add", "arguments": {"a": 2, "b": 3}, "thought": "add"}
        return {"name": "finish", "arguments": {"answer": "5"}}

    log = run_episode(ReActStrategy(policy, max_steps=4), harness, "what is 2+3?")
    calls_after_record = calls["n"]
    print(f"[record]      {len(log)} events, tool invoked {calls_after_record}x")

    # 3) REPLAY — re-execute the recorded tool calls through a fresh harness.
    fresh = DictToolHarness({"add": add}, max_steps=4)
    replayed = replay(fresh, log)
    print(
        f"[replay]      re-executed -> {len(replayed)} events, tool now invoked {calls['n']}x total"
    )
    print(f"[replay]      deterministic: {'yes' if calls['n'] > calls_after_record else 'no'}")


if __name__ == "__main__":
    main()
