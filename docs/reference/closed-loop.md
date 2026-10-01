# Python API: Self-Healing Closed Loop

`agenttune.decide.closed_loop`: detects a failing/looping DECIDE pipeline, classifies why,
generates a corrective training example, retrains, and gates redeployment on real accuracy.

!!! note "The audit-log-driven version works against a real log"
    `FailureDetector.scan_audit_log` and `AuditWriter` now agree on the audit-log schema
    (`trajectory_id`/`stage_name`/`state_snapshot`), so pointed at a real `audit.jsonl` it
    detects real failures; verified against a real DECIDE run that genuinely loops.

## What's real and independently testable right now, with no GPU/API key

Everything below is pure Python; you supply fake `Failure`/`TrainingExample` objects or
trivial stub jobs to exercise the mechanism itself.

| Component | Class | What it does |
|---|---|---|
| Shared buffer | `TrainingBuffer` (`retraining_trigger.py`) | Thread-safe FIFO-capped deque with eviction, lifetime counters, `.health()` snapshot |
| Trigger logic | `RetrainingTrigger` | 6 named conditions (total failures, dominance, enough examples, reward drift, novel failure, staleness) + 2 safety gates |
| Reward drift | `RewardDriftTracker` | Rolling mean/std drift detection. `.load_from_audit()` correctly matches the real audit schema. |
| Non-blocking retrain | `BackgroundRetrainer` | Pure `threading`-based job runner; the injected `job_fn` carries any GPU dependency, not the runner itself |
| Deployment gate | `DeploymentGate` | `build_test_set()` reconstructs ground-truth cases from *successful* past audit entries; also correctly matches the real schema. `.evaluate_decision()` blocks deploy on task OR trajectory-quality regression. |
| Tool isolation | `IsolatedTool` / `isolate_tools()` (`tool_isolation.py`) | Runs any `BaseTool` in a separate OS process with a timeout, automatic in-process fallback if unpicklable |
| Diversity monitor | `BehavioralDiversityMonitor` | Tracks rolling tool-call-sequence signatures, flags behavioral collapse |
| Contracts | `Failure`, `ClassifiedFailure`, `TrainingExample`, `BufferHealth`, `AgenticEvalResult` | Pure dataclasses; construct these by hand to test any component above without going through detection at all |

**Rating A** for all of the above as infrastructure.

## What needs an LLM/API key

- **`FailureClassifier.classify_batch(failures)`**: a litellm call, classifies root cause
  into 5 buckets, safe fallback on error. **Rating B.**
- **`TrainingExampleGenerator.generate_batch(...)`**: synthesizes corrected completions via
  litellm, scores with real TAC/TER metrics. **Rating B.**
- **`ReplayValidator`**: runs a user-supplied shell command per example to validate a
  correction; default is a no-op. **Rating B** in practice, since a meaningful validation
  script is what makes it useful.

Both classifier and generator can be constructed and called on manually-built `Failure`
objects; they don't strictly require the detector to produce their input.

## The two orchestrators: different scope, document/use accordingly

- **`SelfHealingPipeline`** (`closed_loop/pipeline.py`): 2-stage only: detect + classify,
  writes classified failures to a JSONL file.
- **`FullClosedLoop`** (`closed_loop/full_loop.py`): the complete 7-component loop
  (detect → classify → generate → buffer → trigger → retrain → gate/deploy), with
  `.run_until()`/`.run_forever()` turnkey daemons.

Both drive detection the same way, via `ingest_once()`, which now works against a real
audit log (see the note above).

## The recommended path for producing training data today

Since audit-log-based DPO/BCO extraction doesn't work (see
[Python API: DECIDE Engine](decide-engine.md) and
[Known Issues](../community/known-issues.md)), use **`CollectRunner`**: it builds
records directly from in-memory `PipelineState` during a real run, not by re-parsing
`audit.jsonl` afterward.

See it run for real: a real DPO retrain, gated on real accuracy, deploying a real
adapter, in the [Local Notebooks](../notebooks/local-notebook.md) index.
