"""
Case study 7 — End to end: collect, evaluate, find a failure, generate the fix.
===============================================================================

The full "improve an agent" loop in one script, GPU-free:

  collect_rollout  →  evaluate_agentic (real metrics)  →  heal (real FailureDetector
  finds a looping trajectory)  →  a corrective preference dataset ready to retrain on.

This is the value the spine exists for: a running agent's trajectories become the exact
input to evaluation AND failure-recovery, through one EventLog.

Run:  python examples/end_to_end_improve.py
"""

from agenttune.agentic import (
    Event,
    EventKind,
    EventLog,
    Project,
    SelfHealLoop,
    agentic_metrics,
)
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine
from agenttune.decide.closed_loop.contracts import ClassifiedFailure, TrainingExample


def main():
    proj = Project()

    # 1) COLLECT — full-tier trajectories from a (deterministic, GPU-free) engine.
    good = proj.collect_rollout(
        DemoRolloutEngine(), ["route ticket", "classify invoice"], tools=[], max_steps=2
    )
    print(f"[collect]  {len(good)} healthy trajectories, tiers={[l.tier for l in good]}")

    # 2) EVALUATE — the real programmatic metrics over what we collected.
    per = [agentic_metrics(l) for l in good]
    keys = per[0].keys()
    mean = {k: round(sum(p[k] for p in per) / len(per), 2) for k in keys}
    print(f"[evaluate] mean metrics -> {mean}")

    # 3) A failure slips into production: an agent that loops on the same action.
    stuck = EventLog(tier="light")
    for _ in range(4):
        stuck.append(Event(EventKind.TOOL_CALL, {"action": {"name": "search", "arguments": {}}}))
        stuck.append(Event(EventKind.TOOL_RESULT, {"output": "same result"}))
    proj.add_trajectory(stuck)

    failures = proj.heal()
    print(
        f"[detect]   FailureDetector -> {len(failures)} failure(s): {[f.failure_type for f in failures]}"
    )

    # 4) HEAL — classify + generate the corrective preference data (stages injected, model-free).
    def classify(fs):
        return [
            ClassifiedFailure(
                failure=f,
                root_cause=f.failure_type,
                confidence=0.8,
                analysis="loops without progress",
            )
            for f in fs
        ]

    def generate(cfs):
        return [
            TrainingExample(
                trajectory_id=c.failure.trajectory_id,
                original_failure_type=c.failure.failure_type,
                root_cause=c.root_cause,
                prompt=[{"role": "user", "content": "the task"}],
                chosen=[{"role": "assistant", "content": "try a different tool and finish"}],
                rejected=[{"role": "assistant", "content": "call search again"}],
            )
            for c in cfs
        ]

    summary = SelfHealLoop(classify, generate).run(failures)
    print(
        f"[heal]     corrective dataset -> {summary['n_dataset_rows']} row(s) "
        f"{list(summary['dataset'][0].keys())}; ready to retrain the agent"
    )


if __name__ == "__main__":
    main()
