"""
Simple case study — a competitor-research agent.
================================================

The everyday agent: given a product, search for competitors, compare them, and write a short
brief. This is the smallest useful ReAct loop — search → search → summarise → finish — and it
still produces a full `EventLog` you can evaluate, train on, or distil. GPU-free (canned tools
and policy; swap in a real web-search tool and model unchanged).

Run:  python examples/competitor_research.py
"""

from agenttune.agentic import DictToolHarness, EventKind, Project, ReActStrategy

# --- tools (deterministic stand-ins for a real web search) ---
_INDEX = {
    "note-taking apps": ["Notion", "Obsidian", "Roam"],
    "Notion": {"pricing": "$8/user/mo", "strength": "all-in-one workspace"},
    "Obsidian": {"pricing": "free / $8 sync", "strength": "local-first markdown"},
    "Roam": {"pricing": "$15/mo", "strength": "networked thought"},
}


def web_search(query=""):
    return _INDEX.get(query, f"no results for {query!r}")


def compare(name=""):
    return _INDEX.get(name, {"error": f"unknown {name}"})


TOOLS = {"web_search": web_search, "compare": compare}


def research_policy(state):
    """search competitors → compare the top one → finish with a brief."""
    if state.step == 0:
        return {
            "name": "web_search",
            "arguments": {"query": "note-taking apps"},
            "thought": "find the competitor set",
        }
    if state.step == 1:
        return {
            "name": "compare",
            "arguments": {"name": "Obsidian"},
            "thought": "compare the local-first option",
        }
    return {
        "name": "finish",
        "arguments": {
            "answer": "Top competitors: Notion, Obsidian, Roam. "
            "Obsidian is the local-first play at $8 sync."
        },
        "thought": "enough to brief the team",
    }


def main():
    proj = Project(
        strategy=ReActStrategy(research_policy, max_steps=5),
        harness=DictToolHarness(TOOLS, max_steps=5),
    )

    log = proj.infer("Who competes with our note-taking app, and how?")

    print(
        f"[run]     {len(log)} events over {sum(1 for e in log if e.kind is EventKind.TOOL_CALL)} tool calls"
    )
    for e in log:
        if e.kind is EventKind.TOOL_CALL:
            a = e.payload.get("action", {})
            print(f"  → {a.get('name'):<11} {a.get('arguments')}")
    answer = next(
        (
            e.payload["action"]["arguments"].get("answer")
            for e in log
            if e.kind is EventKind.TOOL_CALL and e.payload.get("action", {}).get("name") == "finish"
        ),
        None,
    )
    print(f"[brief]   {answer}")
    print("[spine]   this EventLog is ready for evaluate / train / distil — same as any other")


if __name__ == "__main__":
    main()
