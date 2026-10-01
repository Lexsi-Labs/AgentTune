"""
Healthcare case study — a clinical triage agent, end to end.
============================================================

A nurse-triage agent reads a presenting complaint, checks vitals and history, and assigns an
acuity level (ESI 1 = most urgent … 5 = least). Its trajectories are evaluated, and an agent
that loops on the vitals check without deciding is caught by the real FailureDetector and
turned into corrective training data. GPU-free.

NOT a medical device — an illustrative, deterministic example of the spine on clinical triage.

Run:  python examples/clinical_triage_end_to_end.py
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


def check_vitals(patient=""):
    return {"patient": patient, "hr": 128, "spo2": 91, "bp": "88/60"}


def check_history(patient=""):
    return {"patient": patient, "conditions": ["COPD"], "allergies": ["penicillin"]}


TOOLS = {"check_vitals": check_vitals, "check_history": check_history}


def triage_policy(state):
    """vitals → history → acuity. Canned, model-free."""
    if state.step == 0:
        return {
            "name": "check_vitals",
            "arguments": {"patient": "p-77"},
            "thought": "unstable-sounding complaint — vitals first",
        }
    if state.step == 1:
        return {
            "name": "check_history",
            "arguments": {"patient": "p-77"},
            "thought": "low SpO2 + tachycardia; check comorbidities",
        }
    return {
        "name": "finish",
        "arguments": {"answer": "ESI-2"},
        "thought": "hypoxia + COPD → high acuity, ESI-2",
    }


def main():
    proj = Project(
        strategy=ReActStrategy(triage_policy, max_steps=5),
        harness=DictToolHarness(TOOLS, max_steps=5),
    )

    for complaint in ("chest tightness, short of breath", "sprained ankle, walking"):
        log = proj.infer(f"triage patient: {complaint}")
    print(f"[triage]   {len(proj.trajectories)} patients triaged, last in {len(log)} events")

    per = [agentic_metrics(l) for l in proj.trajectories]
    mean = {k: round(sum(p[k] for p in per) / len(per), 2) for k in per[0]}
    print(f"[evaluate] mean trajectory metrics -> {mean}")

    # A degraded agent loops on vitals without assigning acuity.
    stuck = EventLog(tier="light")
    for _ in range(4):
        stuck.append(
            Event(
                EventKind.TOOL_CALL,
                {"action": {"name": "check_vitals", "arguments": {"patient": "p-9"}}},
            )
        )
        stuck.append(Event(EventKind.TOOL_RESULT, {"output": "hr: 128"}))
    proj.add_trajectory(stuck)
    failures = proj.heal()
    print(
        f"[monitor]  FailureDetector -> {len(failures)} stuck agent(s): {[f.failure_type for f in failures]}"
    )

    def classify(fs):
        return [
            ClassifiedFailure(
                failure=f,
                root_cause=f.failure_type,
                confidence=0.9,
                analysis="re-checks vitals without assigning acuity",
            )
            for f in fs
        ]

    def generate(cfs):
        return [
            TrainingExample(
                trajectory_id=c.failure.trajectory_id,
                original_failure_type=c.failure.failure_type,
                root_cause=c.root_cause,
                prompt=[{"role": "user", "content": "assign an acuity level"}],
                chosen=[{"role": "assistant", "content": "check history, then assign ESI-1..5"}],
                rejected=[{"role": "assistant", "content": "check vitals again"}],
            )
            for c in cfs
        ]

    summary = SelfHealLoop(classify, generate).run(failures)
    print(
        f"[heal]     corrective dataset -> {summary['n_dataset_rows']} row(s) "
        f"{list(summary['dataset'][0].keys())}; ready to retrain the triage agent"
    )


if __name__ == "__main__":
    main()
