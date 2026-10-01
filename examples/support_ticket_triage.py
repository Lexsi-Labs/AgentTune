"""
Simple case study — a support-ticket triage agent, evaluated.
=============================================================

A customer-support agent reads a ticket, checks the account, and routes it (BILLING / BUG /
ESCALATE). It runs over a small labelled set and scores each routing with `Project.evaluate`,
so you can see accuracy the same way you would for any classifier — but the agent is a
tool-using ReAct loop, not a single prompt. GPU-free.

Run:  python examples/support_ticket_triage.py
"""

from agenttune.agentic import DictToolHarness, Project, ReActStrategy


def lookup_account(user=""):
    return {"user": user, "plan": "pro", "open_tickets": 1}


def route_for(text: str) -> str:
    t = text.lower()
    if "refund" in t or "charge" in t or "invoice" in t:
        return "BILLING"
    if "crash" in t or "error" in t or "broken" in t:
        return "BUG"
    return "ESCALATE"


def main():
    tickets = [
        {"task": "I was double charged on my invoice this month", "expected": "BILLING"},
        {"task": "the export button crashes every time", "expected": "BUG"},
        {"task": "I want to speak to someone about my contract", "expected": "ESCALATE"},
    ]

    def policy(state):
        # step 0: look up the account; step 1: route based on the ticket text.
        if state.step == 0:
            return {
                "name": "lookup_account",
                "arguments": {"user": "u-42"},
                "thought": "check plan",
            }
        return {
            "name": "finish",
            "arguments": {"answer": route_for(state.task)},
            "thought": "route the ticket",
        }

    proj = Project(
        strategy=ReActStrategy(policy, max_steps=4),
        harness=DictToolHarness({"lookup_account": lookup_account}, max_steps=4),
    )

    report = proj.evaluate(tickets)
    print(f"[evaluate] routed {report['n']} tickets, accuracy = {report['mean_score']:.2f}")
    for t, s in zip(tickets, report["scores"], strict=False):
        verdict = "correct" if s == 1.0 else "wrong"
        print(f"  [{verdict:7}] {t['expected']:9} <- {t['task'][:44]}")


if __name__ == "__main__":
    main()
