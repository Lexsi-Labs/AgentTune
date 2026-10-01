"""
Retail case study — an order-issue triage agent, evaluated.
===========================================================

A post-purchase agent reads a shopper's message, checks the order, and routes it (WISMO /
REFUND / REPLACEMENT). It runs over a small labelled set and scores each routing with
`Project.evaluate`, so you see accuracy the way you would for any classifier — but the agent is
a tool-using ReAct loop, not a single prompt. GPU-free.

Run:  python examples/order_issue_triage.py
"""

from agenttune.agentic import DictToolHarness, Project, ReActStrategy


def lookup_order(order_id=""):
    return {"order_id": order_id, "status": "delivered", "carrier": "UPS"}


def route_for(text: str) -> str:
    t = text.lower()
    if "refund" in t or "money back" in t or "charged" in t:
        return "REFUND"
    if "broken" in t or "damaged" in t or "wrong item" in t:
        return "REPLACEMENT"
    return "WISMO"  # "where is my order"


def main():
    messages = [
        {"task": "where is my order, it hasn't arrived yet", "expected": "WISMO"},
        {"task": "the mug arrived damaged, it's broken", "expected": "REPLACEMENT"},
        {"task": "please refund me, I was charged twice", "expected": "REFUND"},
    ]

    def policy(state):
        # step 0: pull up the order; step 1: route on the message.
        if state.step == 0:
            return {
                "name": "lookup_order",
                "arguments": {"order_id": "o-9001"},
                "thought": "check order",
            }
        return {
            "name": "finish",
            "arguments": {"answer": route_for(state.task)},
            "thought": "route the issue",
        }

    proj = Project(
        strategy=ReActStrategy(policy, max_steps=4),
        harness=DictToolHarness({"lookup_order": lookup_order}, max_steps=4),
    )

    report = proj.evaluate(messages)
    print(f"[evaluate] routed {report['n']} messages, accuracy = {report['mean_score']:.2f}")
    for m, s in zip(messages, report["scores"], strict=False):
        verdict = "correct" if s == 1.0 else "wrong"
        print(f"  [{verdict:7}] {m['expected']:12} <- {m['task'][:44]}")


if __name__ == "__main__":
    main()
