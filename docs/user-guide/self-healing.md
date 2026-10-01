# Self-Healing & the Closed Loop

Self-healing means an agent's own failures become its next training data, automatically:
detect a bad trajectory, figure out *why* it failed, synthesize a corrected version, and
(optionally) retrain and redeploy, gated on real accuracy, not vibes.

There are two ways into this loop, and **both detect failures automatically today**:
the agentic-spine path (`Project.heal()`), and the DECIDE-based path (`FullClosedLoop`
watching a real `audit.jsonl`). They used to behave differently: `AuditWriter` didn't
emit the fields `FailureDetector.scan_audit_log()` requires, so Path 2 silently found
nothing against a real log, ever. `AuditWriter` now emits them (additively, nothing it
already wrote was renamed or removed), verified against a real `bfsi/kyc_triage` run:
6 genuine `loop_collapse` failures detected where the old schema mismatch would have
found 0. Both paths share the same classify → generate → buffer → trigger → retrain →
gate machinery downstream.

## Path 1: self-heal a `Project`'s own trajectories

If your agent runs through the agentic spine (`Project`/`EventLog`), detection and
healing both work end to end with no extra wiring.

### Why detection works here

`Project.heal()` (`src/agenttune/agentic/project.py`) projects each collected
trajectory into the exact schema `FailureDetector.scan_audit_log()` requires,
`trajectory_id`, `stage_name`, and a dict `state_snapshot`, via
`EventLog.to_audit_records()`, writes it to a temp JSONL, and scans it with the real
detector:

```python
def heal(self, *, detector=None, max_revisits: int = 3):
    detector = detector or FailureDetector(max_revisits=max_revisits)
    records = [r for log in self._trajectories for r in log.to_audit_records()]
    self._emit("heal", "started", n_records=len(records))

    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="agenttune_heal_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        failures = list(detector.scan_audit_log(path))
    finally:
        for p in (path, path + ".offset"):
            if os.path.exists(p):
                os.remove(p)

    self._emit("heal", "detected", n_failures=len(failures),
               types=sorted({f.failure_type for f in failures}))
    return failures
```

`to_audit_records()` builds one record per tool call, folding the tool name **and**
arguments into `stage_name` (so a legitimately-repeated tool call with different
arguments doesn't look like a routing loop to the detector's revisit counter):

```python
def to_audit_records(self) -> list[dict]:
    records: list[dict] = []
    for e in self.events:
        if e.kind is EventKind.TOOL_CALL:
            action = e.payload.get("action") or {}
            name = action.get("name") if isinstance(action, dict) else str(action)
            args = action.get("arguments", {}) if isinstance(action, dict) else {}
            stage_name = name or "unknown"
            if args:
                stage_name = f"{name} {json.dumps(args, sort_keys=True, default=str)}"
            records.append({
                "trajectory_id": self.id,
                "stage_name": stage_name,
                "tool_name": name,
                "stage_type": "tool_call",
                "state_snapshot": {"arguments": args},
                "status": e.payload.get("status", "ok"),
            })
        elif e.kind is EventKind.TOOL_RESULT and e.payload.get("error") and records:
            records[-1]["status"] = "error"
            records[-1]["error_details"] = str(e.payload.get("error"))
    return records
```

That's the field-for-field match with what `FailureDetector.scan_audit_log()` checks
(it hard-rejects any line missing `trajectory_id`/`stage_name`/`state_snapshot`), which
is exactly the schema DECIDE's `AuditWriter` does *not* produce. Same detector, two
producers, one of them compatible.

### Detect, then classify + generate + retrain

```python
from agenttune.agentic import Project
from agenttune.agentic.heal_loop import SelfHealLoop, as_sync_classifier, as_sync_generator
from agenttune.decide.closed_loop.failure_classifier import FailureClassifier
from agenttune.decide.closed_loop.training_example_generator import TrainingExampleGenerator

proj = Project(strategy=strategy, harness=harness)
# ... run some episodes, e.g. proj.collect_rollout(engine, tasks) ...

# 1. Detect — pure Python, no LLM. Real FailureDetector under the hood, fed from
#    this project's own EventLog.to_audit_records() (schema-compatible, see above).
failures = proj.heal()

# 2. Classify + generate + (optionally) retrain — needs an LLM (litellm).
loop = SelfHealLoop(
    classifier=as_sync_classifier(FailureClassifier(model_name="gpt-4o-mini")),
    generator=as_sync_generator(TrainingExampleGenerator(model_name="gpt-4o-mini")),
    trainer_factory=my_trainer_factory,   # optional — omit to just preview the dataset
)
result = loop.run(failures)
print(result["n_classified"], result["n_generated"], result["trained"])
```

Or combine both steps: `SelfHealLoop(...).run_on(proj)` calls `proj.heal()` internally
then runs the loop over whatever it finds; `run_on` is a two-line wrapper around
`project.heal(...)` + `self.run(failures)`.

`SelfHealLoop` (`src/agenttune/agentic/heal_loop.py`) is deliberately GPU/network-free
at the orchestration level: `classifier` and `generator` are injected plain callables
(`list[Failure] -> list[ClassifiedFailure]` and the generator's equivalent), so the loop
itself never imports `litellm`. `as_sync_classifier`/`as_sync_generator` are the
production adapters that wrap the real async `FailureClassifier`/
`TrainingExampleGenerator` (`asyncio.run(...)` under the hood) into that plain-callable
shape:

```python
def as_sync_classifier(classifier) -> Classifier:
    return lambda failures: asyncio.run(classifier.classify_batch(list(failures)))

def as_sync_generator(generator) -> Generator:
    return lambda classified: asyncio.run(generator.generate_batch(list(classified)))
```

`loop.run(failures)` returns a summary dict: `n_failures`, `n_classified`,
`n_generated`, `n_dataset_rows`, `n_skipped`, `trained`, `train_result`,
`trigger_checked`/`trigger_fired`/`trigger_reason`, `deploy_decision`, plus the raw
`classified`/`generated`/`dataset` artifacts. If you pass a `retraining_trigger`
(a real `RetrainingTrigger`, see below), every generated example is added to its buffer
first and training is gated on `.should_trigger()` instead of firing the moment the
dataset is non-empty; that's what closes the gap the code calls out explicitly:
*"the fix for a known gap: `Project.heal()`/
`SelfHealLoop` used to train immediately with no volume/drift check."*

**Test the mechanism with zero GPU/API key** by passing trivial fakes instead of the
real classifier/generator; both just need to match the `list[Failure] ->
list[ClassifiedFailure]` / `list[ClassifiedFailure] -> list[TrainingExample]` shapes.
See the [Local Notebooks](../notebooks/local-notebook.md) index for the real
version: a real DPO retrain, gated on real accuracy (0.33 → 1.00), deploying a real
adapter reloaded as a `PeftModel`.

## Path 2: the DECIDE-based closed loop

`FullClosedLoop` (`src/agenttune/decide/closed_loop/full_loop.py`) watches a DECIDE
`audit.jsonl` and runs the complete 7-component loop end to end:

```
Production audit log (continuous)
  │
  ▼  detect & learn
FailureDetector.scan  →  FailureClassifier.classify_batch (parallel)
                      →  TrainingExampleGenerator.generate_batch (parallel,
                         with ReplayValidator)  →  TrainingExample(s)
  │
  ▼  shared thread-safe TrainingBuffer
ClosedLoopRunner.submit(example)        ← buffer keeps filling
  │
  ▼  decide & act
ClosedLoopRunner.tick()  →  RetrainingTrigger (6 conditions + 2 gates)
  │ fires
  ▼  BackgroundRetrainer (daemon thread; loop keeps running)
retrain_job(examples)  →  RetrainResult
  │ on done
  ▼
BehavioralDiversityMonitor.check  +  TrajectoryEvaluator.evaluate_batch
  │
  ▼
DeploymentGate.evaluate_decision (task A/B + trajectory mean)
  │
  ▼
DeploymentGate.apply_decision  →  deploy  OR  keep old
```

The orchestration is real and correct end to end, including `ingest_once()`'s detection
step: `FailureDetector.scan_audit_log()` now finds real failures against a real DECIDE
`audit.jsonl`, because `AuditWriter.log_stage()`/`.write()` emit the
`trajectory_id`/`stage_name`/`state_snapshot` fields the scanner requires (aliases of the
`pipeline_id`/`stage_id` fields it already wrote, plus a snapshot of `input_text`/
`output`/`stage_outputs` at that point in the run, purely additive, nothing existing was
renamed). `DeploymentGate.build_test_set()` and `RewardDriftTracker.load_from_audit()`
already matched the real DECIDE audit schema independently of this fix.

### Constructing `FullClosedLoop`

```python
from agenttune.decide.closed_loop.full_loop import FullClosedLoop, PathAConfig, GateConfig

loop = FullClosedLoop(
    path_a=PathAConfig(
        audit_log_path="audit.jsonl",
        classifier_model="gpt-4o-mini",
        generator_model="groq/llama-3.3-70b-versatile",
        judge_threshold=0.6,
        max_revisits=3,
    ),
    retrain_job=my_retrain_job,              # list[TrainingExample] -> {"path": ...}
    build_model_runner=my_build_model_runner, # model_path -> (input_text -> verdict)
    old_model_runner=my_old_model_runner,     # the currently-deployed model's verdict fn
    gate_cfg=GateConfig(backend="transformers"),
)
```

`path_a` also builds the gate's test set once, up front, from `build_test_set()`;
if there aren't `min_test_samples` successful runs yet, the gate approves by default
until enough accumulate (logged as a warning, not an error).

### Alternative: construct `Failure` objects yourself

`ingest_once()`'s auto-detection (above) covers the common case: scanning DECIDE's own
audit log. If your failure signal comes from somewhere else instead (a labelled test
set, a downstream monitoring system, manual review), skip detection and feed the pipeline
directly:

```python
import asyncio
from agenttune.decide.closed_loop.contracts import Failure

failures = [
    Failure(trajectory_id="run-123", failure_type="low_judge_score",
            failed_stage_name="process", judge_score=0.3,
            context={"input": "...", "output": "..."}),
    # ... one per failure you've identified ...
]

# classify_batch/generate_batch are async (they call an LLM via litellm).
classified = asyncio.run(loop.classifier.classify_batch(failures))
examples = asyncio.run(loop.generator.generate_batch(classified))
for ex in examples:
    loop.runner.submit(ex)          # sync — adds to the shared TrainingBuffer
loop.tick()                         # sync — checks the trigger, retrains + gates + deploys if conditions are met
```

This drives the exact same downstream machinery `ingest_once()` does, just fed manually.
For a turnkey daemon (the common case, reading DECIDE's own audit log automatically),
`run_until(stop_fn)` / `run_forever()` drive `ingest_once()` + `tick()` on a poll loop,
blocking on the final in-flight retrain if you ask it to.

### Turnkey driving loops

Once examples are flowing into the buffer, via real auto-detection (the default), or via
the manual alternative above, `FullClosedLoop` can drive itself instead of you calling
`ingest_once()`/`tick()` by hand:

```python
import asyncio

# Run until some external stop condition, blocking on the final retrain before returning:
cycles = asyncio.run(loop.run_until(stop=lambda: should_stop, poll_interval_s=5.0))
for cycle in cycles:
    print(cycle.tick.fired, cycle.tick.reason, cycle.applied)

# Or run forever (production daemon) — each iteration is ingest_once() -> tick() -> sleep:
# asyncio.run(loop.run_forever(poll_interval_s=5.0))
```

Each iteration ingests any new audit-log lines, ticks the trigger, and sleeps. The
retrain itself always runs on `BackgroundRetrainer`'s daemon thread, so `tick()` never
blocks and the buffer keeps accumulating examples while a retrain is in flight. Every
completed cycle is recorded on `loop.cycles` as a `LoopCycle` (`tick`, `retrain_success`,
`decision`, `applied`, `diversity_collapsed`, `trajectory_mean_new`) so you can inspect
what happened without re-deriving it from logs.

### Producing training data without going through detection at all

If what you actually want is DPO/BCO/GRPO training data from DECIDE runs, not the
self-healing loop specifically, use `CollectRunner` instead (see
[How-To: Set Up DECIDE](../user-guide/decide-workflows.md)).
It builds records directly from live pipeline runs, sidestepping the audit-log path
entirely.

## The downstream machinery, standalone

Everything past detection is real, pure Python, and independently testable with fakes,
no GPU, no API key, no network. This section builds each piece by hand so you can see
exactly what it needs and what it gives back.

### `TrainingBuffer` + `RetrainingTrigger`

`TrainingBuffer` (`decide/closed_loop/retraining_trigger.py`) is the thread-safe,
FIFO-capped buffer that sits between the detect/classify/generate side and the
retrain/gate side. `RetrainingTrigger` wraps it with six trigger conditions (any one
fires a retrain) and two safety gates:

```python
from agenttune.decide.closed_loop.contracts import TrainingExample
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig

trigger = RetrainingTrigger(config=TriggerConfig(
    total_failures_threshold=50,   # T1: buffer size
    dominance_ratio=0.70,          # T2: one root_cause >= 70% of buffer
    min_examples_ready=30,         # T3: accepted-example count
    max_drop_rate=0.60,            # gate: refuse to train on a mostly-bad buffer
))

# Simulate a handful of "wrong_tool" corrections landing in the buffer.
for i in range(5):
    trigger.buffer.add(TrainingExample(
        trajectory_id=f"run-{i}",
        original_failure_type="tool_crash",
        root_cause="wrong_tool",
        prompt=[{"role": "user", "content": "book a flight"}],
        chosen=[{"role": "assistant", "content": '{"name": "search_flights", "arguments": {}}'}],
        rejected=[{"role": "assistant", "content": '{"name": "book_flight", "arguments": {}}'}],
    ))

health = trigger.buffer.health()
print(health.buffer_size, health.dominant_failure_type, health.drop_rate)
# 5 wrong_tool 0.0

fired, reason = trigger.should_trigger()
print(fired, reason)   # True dominance:wrong_tool — all 5 share one root_cause, which
                        # clears T2's 70% dominance threshold even though T1 (buffer size
                        # >= 50) is nowhere close
```

`TrainingBuffer.add()` counts an example as "accepted" if it carries a `chosen` or
`rejected` response (an example with neither carries no signal for the `drop_rate`
gate). `.health()` returns a `BufferHealth` snapshot: `buffer_size`,
`dominant_failure_type`/`dominant_failure_ratio`, `drop_rate`, `novel_failure_types`,
and drain/retrain timestamps. `RetrainingTrigger.check_and_fire()` evaluates
`should_trigger()` and atomically drains the buffer if it fires; that's what
`ClosedLoopRunner.tick()` calls under the hood.

The six trigger conditions, in evaluation order (`_evaluate_triggers`):

| # | Condition | Fires when |
|---|---|---|
| T1 | `total_failures_exceeded` | buffer size ≥ `total_failures_threshold` |
| T2 | `dominance:<type>` | one `root_cause` ≥ `dominance_ratio` of the buffer |
| T3 | `enough_examples` | accepted-example count ≥ `min_examples_ready` |
| T4 | `reward_drift` | rolling mean episode reward drifts below baseline (via `RewardDriftTracker`) |
| T5 | `novel_failure:<type>` | a never-before-seen `root_cause` shows up ≥ `novel_type_min_count` times |
| T6 | `staleness` | ≥ `staleness_window_hours` since the last retrain, with a minimum buffer size |

Two gates run before any of that fires a real retrain: `retrain_in_progress` (never
double-fire while one is running) and the `drop_rate` buffer-health guard (refuse to
train on a mostly-bad buffer, once there are `min_attempts_for_drop_gate` attempts to
judge it by).

### `DeploymentGate.evaluate_decision()` standalone

`DeploymentGate` (`decide/closed_loop/deployment_gate.py`) is model-agnostic: you
supply `model_runner_fn(input_text) -> verdict` callables, so it's fully testable
without a GPU:

```python
from agenttune.decide.closed_loop.deployment_gate import DeploymentGate

gate = DeploymentGate(seed=0)

test_set = [
    {"pipeline_id": "p1", "input_text": "refund request for order 42",
     "expected_verdict": "APPROVE"},
    {"pipeline_id": "p2", "input_text": "suspicious high-value transfer",
     "expected_verdict": "DENY"},
    # ... build these by hand, or via gate.build_test_set("audit.jsonl") against
    # real successful DECIDE runs (this DOES match the real audit schema) ...
]

# Fake model_runners: old model gets both wrong, new model gets both right.
old_model_fn = lambda _input: "DENY"
new_model_fn = {"refund request for order 42": "APPROVE",
                "suspicious high-value transfer": "DENY"}.get

decision = gate.evaluate_decision(
    test_set,
    old_model_fn=old_model_fn,
    new_model_fn=new_model_fn,
    task_regression_tol=0.0,
    trajectory_regression_tol=0.05,
)
print(decision.approved, decision.reason, decision.task_delta)
# True approved: no regression on task or trajectory 0.5
# (old model gets p1 wrong / p2 right -> 0.5 accuracy; new model gets both right -> 1.0;
# task_delta = new_task - old_task = 0.5)
```

`evaluate_decision` blocks the deploy if **either** signal regresses: task accuracy
(`new_task < old_task - task_regression_tol`) or trajectory quality (when you pass
`old_trajectory_results`/`new_trajectory_results`, lists of `AgenticEvalResult` from
`TrajectoryEvaluator.evaluate_batch`; omit both to skip that check and gate on task
accuracy alone). `apply_decision(decision, config_path=..., trained_path=...,
deploy_fn=..., rollback_fn=...)` then actually deploys or keeps the old model;
`deploy_fn`/`rollback_fn` are injectable so this is testable without touching a real
deployment; omitted, they default to the Decide `model_deployment` bridge.

### `IsolatedTool` / `isolate_tools()`: real off-process sandboxing

`decide/closed_loop/tool_isolation.py` runs a tool in a separate OS process
(`ProcessPoolExecutor`, Level 1 of a planned 3-level isolation ladder; containers/remote
hosts are future work) so a crashing, hanging, or untrusted tool can't take down the
rollout loop or read the training process's memory. This is genuinely standalone; no
LLM, no training loop, just a `BaseTool`:

```python
from agenttune.agentic.tools.base import BaseTool, ToolResult
from agenttune.decide.closed_loop.tool_isolation import IsolatedTool, isolate_tools

class EchoTool(BaseTool):
    name = "echo"
    description = "echoes input"

    def _parameters(self):
        return {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=kwargs.get("msg", ""))

tool = IsolatedTool(EchoTool(), timeout_s=30)
result = tool.execute(msg="hello")
print(result.success, result.output, result.metadata["isolation"])
# True hello process

# Wrap a whole tool list at once:
tools = isolate_tools([EchoTool(), EchoTool()], timeout_s=10)
```

`IsolatedTool` presents the exact same `BaseTool` interface (`name`, `description`,
`to_schema`, `execute`) as the tool it wraps, so a rollout loop can't tell an isolated
tool from a local one. **Fallback guarantee**: if isolation fails for any reason
(process pool unavailable, worker crash, timeout, an un-picklable tool, e.g. one
holding a live socket or a `threading.Lock`), it falls back to running the tool
in-process rather than blocking the loop. Every result's `metadata["isolation"]` tells
you which path ran: `"process"`, `"in_process_fallback"` (with a `fallback_reason` of
`"not_picklable"`, `"timeout"`, or `"isolation_error"`), or `"unavailable"` if you passed
`require_isolation=True` and isolation genuinely wasn't possible; that flag turns a
silent fallback into an explicit error `ToolResult` for callers that must not run
untrusted code in-process:

```python
class UnpicklableTool(BaseTool):
    name = "unpicklable"
    description = "holds a lock (not picklable)"

    def __init__(self):
        self._lock = threading.Lock()   # locks can't be pickled

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="ran locally")

safe = IsolatedTool(UnpicklableTool())                       # falls back silently
strict = IsolatedTool(UnpicklableTool(), require_isolation=True)  # returns an error result instead

print(safe.execute().metadata)     # {'isolation': 'in_process_fallback', 'fallback_reason': 'not_picklable', ...}
print(strict.execute().success)    # False
```

Picklability is checked once at construction (`pickle.dumps(tool)`), not on every call,
so the common case (a plain, picklable tool) pays the isolation cost every `execute()`
without re-checking picklability each time.

## Building the preference dataset by hand

`SelfHealLoop.run()` turns generated `TrainingExample`s into DPO/BCO rows via a public
helper you can call directly to preview the dataset without a trainer:

```python
from agenttune.agentic.heal_loop import build_dataset

rows = build_dataset(generated_examples)   # list[TrainingExample] -> list[{prompt, chosen, rejected}]
print(rows[0])
# {"prompt": [...], "chosen": [...], "rejected": [...]}
```

`build_dataset` skips any example with no derivable preference pair.
`TrainingExampleGenerator` (the closed loop's real generator) currently only populates the
multi-completion form (`completions` + `rewards`, meant for future GRPO-style use), not
`chosen`/`rejected` directly, so each `TrainingExample.derive_preference_from_completions()`
bridges that into a usable pair: best-reward completion becomes `chosen`, worst becomes
`rejected`. It's a no-op (returns `False`, pair left unset) if a preference pair already
exists, if there are fewer than two completions, or if every completion scored the same
(no meaningful preference to derive). `has_preference_pair()` tells you whether an
example is DPO/BCO-ready before you bother building a row from it.

## Troubleshooting

- **`FullClosedLoop.ingest_once()` submits 0 examples against a real audit log.** This
  used to always be true (a schema mismatch between `AuditWriter` and
  `FailureDetector.scan_audit_log()`, now fixed; see the note at the top of this page).
  If you're still seeing 0 and the log genuinely has failures in it (a `loop_collapse`
  needs more than `max_revisits` repeats of the *same* stage in one trajectory; a
  `tool_crash` needs a `tool_call` stage with `status: error`; `low_judge_score` needs an
  `llm_judge` stage scoring below `judge_threshold`), check you're on a version of
  `audit.py` that writes `trajectory_id`/`stage_name`/`state_snapshot`.
- **`DeploymentGate` always approves with `task_old=0.0, task_new=0.0`.** Its test set
  is empty; `build_test_set()` logs a warning and returns `[]` when there are fewer
  than `min_samples` (default 20) successful runs in the audit log. Lower
  `min_test_samples` in `GateConfig` for early testing, or seed more successful runs.
- **`IsolatedTool` always reports `"in_process_fallback"`.** The wrapped tool isn't
  picklable; check for open sockets, locks, file handles, or bound methods captured in
  a closure held on the tool instance. `IsolatedTool.__repr__()` shows `picklable=False`
  if you want to confirm without executing.
- **`RetrainingTrigger.should_trigger()` returns `retrain_already_running` forever.**
  Something called `mark_retrain_started()` without a matching
  `mark_retrain_finished()`. If you're driving `RetrainingTrigger` directly (not
  through `ClosedLoopRunner`, which manages this for you via `BackgroundRetrainer`),
  make sure every start has a finish, including on the error path.
- **A retrain fires but `_on_retrain_done` never gates/deploys anything.** Check
  `result.result["path"]` is actually set; `FullClosedLoop._on_retrain_done` logs
  `"retrain result missing 'path'; cannot gate"` and returns early if your
  `retrain_job` doesn't return a dict with a `path` key.

## What needs an LLM/API key vs. what's pure Python

| Component | Needs | Rating |
|---|---|---|
| `Project.heal()` / `FailureDetector.scan_audit_log` | nothing, pure Python | A |
| `TrainingBuffer` / `RetrainingTrigger` / `RewardDriftTracker` | nothing, pure Python | A |
| `DeploymentGate.build_test_set` / `.score_model` / `.evaluate_decision` | a `model_runner_fn` you supply (can be a fake) | A |
| `IsolatedTool` / `isolate_tools()` | nothing, pure Python + stdlib `multiprocessing` | A |
| `FailureClassifier.classify_batch` | an LLM via `litellm` | B |
| `TrainingExampleGenerator.generate_batch` | an LLM via `litellm`, plus `ReplayValidator`'s shell command | B |
| `SelfHealLoop`'s `trainer_factory` | your real trainer (e.g. a TRL DPO trainer) | B |

Everything rated **A** above can be exercised right now with zero GPU and zero API key,
that's most of the surface area of this page. Only the classify/generate/retrain steps
actually need a model.

## See also

- [Known Issues](../community/known-issues.md): the audit-log schema details behind
  Path 2's detection behaviour, and other edge cases.
- [Concepts: DECIDE & the closed loop](../concepts/decide-and-closed-loop.md): the
  same material from the DECIDE-pipeline side, described in plain-English stage names.
- [How-To: Set Up DECIDE](../user-guide/decide-workflows.md): building the pipeline whose
  audit log this page's Path 2 reads.
- [Reference: Self-Healing Closed Loop](../reference/closed-loop.md): the full
  component table with ratings, one row per class.
