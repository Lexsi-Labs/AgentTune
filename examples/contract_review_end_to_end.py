"""
Legal case study — Contract-review agent, end to end.
=====================================================

A contract-review agent reads a clause, checks the counterparty's obligations and the governing
law, and decides ACCEPT / NEGOTIATE / REJECT. The point of the spine: those trajectories feed
both evaluation and failure-recovery. A reviewer that loops re-reading the same clause without
deciding is caught by the real FailureDetector and turned into corrective training data.
GPU-free.

Illustrative only — not legal advice.

Run:  python examples/contract_review_end_to_end.py
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


# --- the reviewer's tools (deterministic stand-ins for a clause library / CLM lookup) ---
def read_clause(clause_id=""):
    return {
        "clause_id": clause_id,
        "type": "limitation_of_liability",
        "cap": "fees paid in prior 3 months",
    }


def check_obligation(party=""):
    return {"party": party, "indemnifies": False, "uncapped_carveouts": ["IP", "confidentiality"]}


def governing_law(clause_id=""):
    return {"clause_id": clause_id, "law": "Delaware", "acceptable": True}


TOOLS = {
    "read_clause": read_clause,
    "check_obligation": check_obligation,
    "governing_law": governing_law,
}


def reviewer_policy(state):
    """A ReAct contract reviewer: read the clause -> check obligations -> decide. Canned."""
    if state.step == 0:
        return {
            "name": "read_clause",
            "arguments": {"clause_id": "LOL-7"},
            "thought": "start with the liability cap",
        }
    if state.step == 1:
        return {
            "name": "check_obligation",
            "arguments": {"party": "counterparty"},
            "thought": "cap is low and one-sided; check indemnity",
        }
    return {
        "name": "finish",
        "arguments": {"answer": "NEGOTIATE"},
        "thought": "low mutual cap, no indemnity -> push back",
    }


def main():
    proj = Project(
        strategy=ReActStrategy(reviewer_policy, max_steps=5),
        harness=DictToolHarness(TOOLS, max_steps=5),
    )

    # 1) Review a batch of clauses (light-tier episodes).
    for clause in ("LOL-7 limitation of liability", "IND-2 indemnification"):
        log = proj.infer(f"review contract clause: {clause}")
    print(
        f"[review]   {len(proj.trajectories)} clauses reviewed, "
        f"last decision reached in {len(log)} events"
    )

    # 2) Evaluate the review trajectories with the real programmatic metrics.
    per = [agentic_metrics(l) for l in proj.trajectories]
    mean = {k: round(sum(p[k] for p in per) / len(per), 2) for k in per[0]}
    print(f"[evaluate] mean trajectory metrics -> {mean}")

    # 3) A degraded reviewer slips out: it re-reads the same clause, never deciding.
    stuck = EventLog(tier="light")
    for _ in range(4):
        stuck.append(
            Event(
                EventKind.TOOL_CALL,
                {"action": {"name": "read_clause", "arguments": {"clause_id": "LOL-7"}}},
            )
        )
        stuck.append(Event(EventKind.TOOL_RESULT, {"output": "type: limitation_of_liability"}))
    proj.add_trajectory(stuck)
    failures = proj.heal()
    print(
        f"[monitor]  FailureDetector -> {len(failures)} stuck reviewer(s): {[f.failure_type for f in failures]}"
    )

    # 4) Turn the failure into corrective preference data (retrain-ready).
    def classify(fs):
        return [
            ClassifiedFailure(
                failure=f,
                root_cause=f.failure_type,
                confidence=0.85,
                analysis="re-reads the clause without deciding",
            )
            for f in fs
        ]

    def generate(cfs):
        return [
            TrainingExample(
                trajectory_id=c.failure.trajectory_id,
                original_failure_type=c.failure.failure_type,
                root_cause=c.root_cause,
                prompt=[{"role": "user", "content": "review the contract clause"}],
                chosen=[
                    {
                        "role": "assistant",
                        "content": "check obligations, then decide ACCEPT/NEGOTIATE/REJECT",
                    }
                ],
                rejected=[{"role": "assistant", "content": "read the clause again"}],
            )
            for c in cfs
        ]

    summary = SelfHealLoop(classify, generate).run(failures)
    print(
        f"[heal]     corrective dataset -> {summary['n_dataset_rows']} preference row(s) "
        f"{list(summary['dataset'][0].keys())}; ready to retrain the review agent"
    )


if __name__ == "__main__":
    main()
