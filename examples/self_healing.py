"""
Case study 3 — Closed-loop self-healing.
=========================================

Value path:  detect a failing (looping) agent  ->  classify the root cause
             ->  generate corrective preference data  ->  (retrain on GPU)

`Project.heal` runs the REAL closed-loop `FailureDetector` over collected trajectories.
`SelfHealLoop` then drives the full loop; the litellm-bound classify/generate stages are
injected here as deterministic stand-ins so the whole thing runs model-free. The corrective
dataset is real `{prompt, chosen, rejected}` preference data built by the closed-loop's own
`derive_preference_from_completions`.

Run:  python examples/self_healing.py
"""

from agenttune.agentic import Event, EventKind, EventLog, Project, SelfHealLoop
from agenttune.decide.closed_loop.contracts import ClassifiedFailure, TrainingExample


def seed_looping_failure(proj: Project) -> None:
    """Inject a stuck agent: the same tool call 4x with no progress -> loop_collapse."""
    log = EventLog(tier="light")
    for _ in range(4):
        log.append(Event(EventKind.TOOL_CALL, {"action": {"name": "search", "arguments": {}}}))
        log.append(Event(EventKind.TOOL_RESULT, {"output": "same result again"}))
    proj.add_trajectory(log)


def demo_classifier(failures):
    return [
        ClassifiedFailure(
            failure=f,
            root_cause=f.failure_type,
            confidence=0.8,
            analysis=f"demo analysis: {f.failure_type}",
        )
        for f in failures
    ]


def demo_generator(classified):
    return [
        TrainingExample(
            trajectory_id=cf.failure.trajectory_id,
            original_failure_type=cf.failure.failure_type,
            root_cause=cf.root_cause,
            prompt=[{"role": "user", "content": f"Recover from {cf.failure.failure_type}"}],
            chosen=[{"role": "assistant", "content": "a corrected, non-looping response"}],
            rejected=[{"role": "assistant", "content": "the repeated failing action"}],
        )
        for cf in classified
    ]


def main() -> None:
    proj = Project()
    seed_looping_failure(proj)
    print(f"[seed]    injected 1 looping trajectory ({len(proj.trajectories)} total)")

    # 1) DETECT — the real FailureDetector, GPU-free.
    failures = proj.heal()
    print(
        f"[detect]  FailureDetector found {len(failures)} failure(s): "
        f"{[f.failure_type for f in failures]}"
    )

    # 2..4) The full loop: classify -> generate -> corrective dataset.
    summary = SelfHealLoop(demo_classifier, demo_generator).run_on(proj)
    print(f"[classify] root causes -> {[c.root_cause for c in summary['classified']]}")
    print(
        f"[dataset]  {summary['n_dataset_rows']} corrective preference row(s); "
        f"keys={list(summary['dataset'][0].keys())}"
    )
    print(
        f"[summary] {{n_failures: {summary['n_failures']}, "
        f"n_classified: {summary['n_classified']}, "
        f"n_dataset_rows: {summary['n_dataset_rows']}, trained: {summary['trained']}}}"
    )


if __name__ == "__main__":
    main()
