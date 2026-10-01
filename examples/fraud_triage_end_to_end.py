"""
BFSI case study — Fraud-triage agent, end to end.
=================================================

A card-fraud triage agent that inspects a flagged transaction with tools, decides
APPROVE / REVIEW / BLOCK, and then — the point of the spine — its trajectories feed both
evaluation and failure-recovery. A stuck agent that loops on one check is caught by the real
FailureDetector and turned into corrective training data. GPU-free.

Run:  python examples/fraud_triage_end_to_end.py
"""

from agenttune.agentic import (
    DictToolHarness,
    Event,
    EventKind,
    EventLog,
    Project,
    ReActStrategy,
    SelfHealLoop,
    agentic_metrics,
)
from agenttune.decide.closed_loop.contracts import ClassifiedFailure, TrainingExample


# --- the fraud-analyst tools (deterministic stand-ins for real risk services) ---
def check_velocity(account=""):  # txns per hour on this account
    return {"account": account, "txn_per_hour": 9, "threshold": 5}


def check_geo(account=""):  # geo mismatch vs home region
    return {"account": account, "home": "US-TX", "txn_geo": "RO-B", "mismatch": True}


def sanctions_screen(name=""):
    return {"name": name, "ofac_hit": False}


TOOLS = {
    "check_velocity": check_velocity,
    "check_geo": check_geo,
    "sanctions_screen": sanctions_screen,
}


def analyst_policy(state):
    """A ReAct fraud analyst: velocity → geo → decision. Model-free, canned."""
    if state.step == 0:
        return {
            "name": "check_velocity",
            "arguments": {"account": "acct-4417"},
            "thought": "unusual spend — check velocity first",
        }
    if state.step == 1:
        return {
            "name": "check_geo",
            "arguments": {"account": "acct-4417"},
            "thought": "velocity is high; check for geo mismatch",
        }
    return {
        "name": "finish",
        "arguments": {"answer": "BLOCK"},
        "thought": "high velocity + geo mismatch → block and open a case",
    }


def main():
    proj = Project(
        strategy=ReActStrategy(analyst_policy, max_steps=5),
        harness=DictToolHarness(TOOLS, max_steps=5),
    )

    # 1) Triage a batch of flagged transactions (light-tier episodes).
    for txn in ("txn-1001 $4,200 electronics", "txn-1002 $980 gift cards"):
        log = proj.infer(f"triage flagged transaction: {txn}")
    print(
        f"[triage]   {len(proj.trajectories)} transactions triaged, "
        f"last decision reached in {len(log)} events"
    )

    # 2) Evaluate the triage trajectories with the real programmatic metrics.
    per = [agentic_metrics(l) for l in proj.trajectories]
    mean = {k: round(sum(p[k] for p in per) / len(per), 2) for k in per[0]}
    print(f"[evaluate] mean trajectory metrics -> {mean}")

    # 3) A degraded agent slips out: it loops on check_velocity, never deciding.
    stuck = EventLog(tier="light")
    for _ in range(4):
        stuck.append(
            Event(
                EventKind.TOOL_CALL,
                {"action": {"name": "check_velocity", "arguments": {"account": "acct-9"}}},
            )
        )
        stuck.append(Event(EventKind.TOOL_RESULT, {"output": "txn_per_hour: 9"}))
    proj.add_trajectory(stuck)
    failures = proj.heal()
    print(
        f"[monitor]  FailureDetector -> {len(failures)} stuck agent(s): {[f.failure_type for f in failures]}"
    )

    # 4) Turn the failure into corrective preference data (retrain-ready).
    def classify(fs):
        return [
            ClassifiedFailure(
                failure=f,
                root_cause=f.failure_type,
                confidence=0.85,
                analysis="re-checks velocity without deciding",
            )
            for f in fs
        ]

    def generate(cfs):
        return [
            TrainingExample(
                trajectory_id=c.failure.trajectory_id,
                original_failure_type=c.failure.failure_type,
                root_cause=c.root_cause,
                prompt=[{"role": "user", "content": "triage the flagged transaction"}],
                chosen=[
                    {"role": "assistant", "content": "check geo, then decide BLOCK/REVIEW/APPROVE"}
                ],
                rejected=[{"role": "assistant", "content": "check velocity again"}],
            )
            for c in cfs
        ]

    summary = SelfHealLoop(classify, generate).run(failures)
    print(
        f"[heal]     corrective dataset -> {summary['n_dataset_rows']} preference row(s) "
        f"{list(summary['dataset'][0].keys())}; ready to retrain the fraud agent"
    )


if __name__ == "__main__":
    main()
