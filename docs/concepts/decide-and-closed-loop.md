# DECIDE & the closed loop

DECIDE is AgentTune's YAML-defined decision-workflow engine. The closed loop is a
separate, optional system built on top of it that watches DECIDE's production traffic,
learns from failures, and safely redeploys a retrained model. This page explains both,
and defines the two closed-loop stages by what they do.

## DECIDE: workflows as YAML, not code

`agenttune.decide.GraphRunner` compiles a YAML template into a graph of stages and runs
it, a decision pipeline defined as config instead of a hand-written script. Stage types
include `llm_call`, `router`, `llm_judge`, `rules`, `tool_call`, `parallel`, and `output`
(`src/agenttune/decide/stages/`). Templates ship for generic, custom, and BFSI use cases
(`src/agenttune/decide/templates/`), and every run is written to an append-only audit log
(`AuditWriter`); that log is the closed loop's only input.

```python
from agenttune.decide import GraphRunner

# template_id is a name registered under src/agenttune/decide/templates/ (here,
# templates/generic/text_classify.yaml), NOT an arbitrary file path.
runner = GraphRunner.from_template("generic/text_classify", config_path="config.yaml")
state = runner.run_sync("some input")
print(state.verdict)
```

See it run: [Local Notebooks](../notebooks/local-notebook.md),
[DECIDE examples](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/DECIDE_EXAMPLES.md).

## The closed loop: two stages, one shared buffer

The closed loop (`agenttune.decide.closed_loop`) reads DECIDE's audit log, learns from
what went wrong, and, if it decides the fix is good enough, redeploys automatically.
It's two stages connected by a thread-safe buffer, orchestrated end to end by
`FullClosedLoop` (`src/agenttune/decide/closed_loop/full_loop.py`):

```
production audit log (continuous)
        │
        ▼
┌───────────────────────────────────────────┐
│ 1. Failure Detection & Example Generation  │
│    FailureDetector.scan                    │
│    → FailureClassifier.classify_batch      │
│    → TrainingExampleGenerator.generate     │
└───────────────────────────────────────────┘
        │  TrainingExample → shared TrainingBuffer
        ▼
┌───────────────────────────────────────────┐
│ 2. Retrain, Gate & Deploy                  │
│    RetrainingTrigger (6 conditions,        │
│      2 safety gates)                       │
│    → BackgroundRetrainer.retrain_job       │
│    → DeploymentGate.evaluate_decision      │
│    → apply_decision (deploy or keep old)   │
└───────────────────────────────────────────┘
```

### Stage 1: Failure Detection & Example Generation

Scans the audit log for failed or low-scoring runs, classifies *why* each one failed,
and turns the good ones into training examples.

!!! note "Fires correctly against a real DECIDE audit log"
    `FailureDetector.scan_audit_log` requires every audit line to carry `trajectory_id`,
    `stage_name`, and a dict `state_snapshot`. `AuditWriter`, the thing that actually
    writes DECIDE's `audit.jsonl`, now emits all three (as aliases of the
    `pipeline_id`/`stage_id` fields already written, plus a snapshot of `input_text`/
    `output`/`stage_outputs`). Verified against a real DECIDE-generated log that
    genuinely loops: the scanner correctly reports the failures, and `ingest_once()`
    classifies, generates, and buffers training examples from them end-to-end.

- **`FailureDetector.scan_audit_log`** (`closed_loop/failure_detector.py`): walks the
  audit log from the last read offset, flags a run as a failure below `judge_threshold` or
  after `max_revisits` retries.
- **`FailureClassifier.classify_batch`** (`closed_loop/failure_classifier.py`): an LLM
  call (via `litellm`) assigns a root cause: `wrong_tool`, `wrong_routing`,
  `incomplete_reasoning`, `hallucinated_output`, or `loop_collapse`.
- **`TrainingExampleGenerator.generate`** (`closed_loop/training_example_generator.py`):
  synthesizes a corrected response for the failure, validated by `ReplayValidator`,
  and emits a `TrainingExample` (`closed_loop/contracts.py`), either a multi-completion
  form (for GRPO-style training) or a `chosen`/`rejected` preference pair (for DPO/BCO).

### Stage 2: Retrain, Gate & Deploy

Decides *when* enough evidence has accumulated to retrain, runs the retrain, and only
ships the result if it's actually better.

- **`RetrainingTrigger`** (`closed_loop/retraining_trigger.py`): evaluates six trigger
  conditions (buffer size, reward drift, etc.) plus two safety gates
  (`retrain_in_progress`, a buffer-health `drop_rate` guard) before draining the buffer.
- **`BackgroundRetrainer`** (`closed_loop/retrain_runner.py`): runs the retrain job
  (e.g. a DPO LoRA adapter) on a daemon thread so the loop keeps ingesting failures
  while training happens.
- **`DeploymentGate.evaluate_decision`** (`closed_loop/deployment_gate.py`): scores
  old vs. new model on a held-out task test set reconstructed from *successful* past
  runs, **and** checks the trajectory-eval score doesn't regress. Both must pass, or the
  gate blocks the deploy and keeps the old model.
- **`apply_decision`**: on approval, deploys the new adapter through the existing
  deployment bridge; on rejection, no-ops and the old model stays live.

### Running it

`FullClosedLoop` drives both stages from one object; call `ingest_once()` for stage 1
and `tick()` for stage 2 yourself, or use `run_until()` / `run_forever()` for a turnkey
daemon. Every model-dependent boundary (the retrain job, the old/new model scoring
runners) is an injected callable, so the orchestration itself is GPU-free and
unit-testable; only the injected callables need a real model.

See it run for real: a real DPO retrain, gated on real accuracy, deploying a real
adapter — see the [real-examples index](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/REAL_EXAMPLES.md), narrated
in the [Local Notebooks](../notebooks/local-notebook.md) index.

## `EventLog`: what connects DECIDE to everything else

DECIDE's `PipelineState` is one of several sources the agentic spine's `EventLog`
projects from (alongside the strategy/harness rollout path and eval dicts); see
[`src/agenttune/agentic/README.md`](https://github.com/Lexsi-Labs/AgentTune/blob/main/src/agenttune/agentic/README.md) for the full
two-tier schema. That's what lets a DECIDE run feed the same evaluation and healing
machinery as an agentic-spine rollout, without a separate integration per source.
