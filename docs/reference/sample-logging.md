# Python API: Audit & Experiment Logging

Two unrelated things share the word "logging" in this codebase:

1. **`decide/audit.py`**: the JSONL trail of what a DECIDE pipeline actually did
   (`AuditWriter`/`AuditReader`). This is a decision audit trail, not a training-metrics log.
2. **`utils/logging.py`** (+ `core/sft/logging.py`): experiment/metrics logging to
   console, WandB, and TensorBoard during training.

They don't interact. This page covers both, plus where the audit-log reader's training-data
extraction genuinely breaks against real DECIDE output.

## `decide/audit.py`: the DECIDE audit trail

### `AuditWriter`

```python
class AuditWriter:
    def __init__(self, path: str = "./audit.jsonl") -> None: ...
    def log_stage(self, state: PipelineState, stage: Dict[str, Any], result: Dict[str, Any]) -> None: ...
    def write(self, state: PipelineState) -> None: ...
```

`log_stage` is called once per stage execution during a `GraphRunner` run; `write` is called
once at the end of a pipeline run. Each appends one JSON line to `path` (directories created
automatically via `os.makedirs`).

**`log_stage` entry shape:**

```python
{
    "timestamp": <ISO 8601>,
    "pipeline_id": state.pipeline_id,
    "trajectory_id": state.pipeline_id,          # alias of pipeline_id, added for FailureDetector
    "template_id": state.template_id,
    "template_version": state.template_version,
    "input_hash": state.input_hash,
    "stage_id": stage["id"],
    "stage_name": stage["id"],                   # alias of stage_id, added for FailureDetector
    "stage_type": stage["type"],
    "iteration": <int>,
    "input": stage.get("prompt"),
    "output": result.get("output"),
    "latency_ms": result.get("latency_ms"),
    "cost_usd": result.get("cost_usd"),
    "reward": <float 0.0-1.0, or None>,   # via _compute_reward, only if reward.stages.<id> is configured
    "result": {"score": <reward>} if reward is not None else None,
    "model": stage.get("model"),
    "is_retry": <bool>,
    "status": "error" if result.get("error") else "ok",
    "error": result.get("error"),
    "error_details": result.get("error"),
    "state_snapshot": {                          # added for FailureDetector
        "input_text": state.input_text,
        "output": result.get("output"),
        "stage_outputs": dict(state.stage_outputs),
    },
}
```

**`write` entry shape** (the final completion line):

```python
{
    "timestamp_end": state.timestamp_end,
    "pipeline_id": state.pipeline_id,
    "trajectory_id": state.pipeline_id,          # alias of pipeline_id, added for FailureDetector
    "template_id": state.template_id,
    "stage_name": "__pipeline_complete__",       # added for FailureDetector — see note below
    "stage_type": "output",
    "verdict": state.verdict,
    "verdict_label": state.verdict_label,
    "is_complete": state.is_complete,
    "step_count": state.step_count,
    "elapsed_seconds": state.elapsed_seconds,
    "episode_reward": <float or None>,   # via _compute_episode_reward, only if reward.final_fn == "weighted_mean"
    "status": "error" if state.error else "ok",
    "error": state.error,
    "error_details": state.error,
    "state_snapshot": {                          # added for FailureDetector
        "verdict": state.verdict,
        "verdict_label": state.verdict_label,
        "confidence": state.confidence,
        "reason": state.reason,
        "stage_outputs": dict(state.stage_outputs),
    },
}
```

`trajectory_id`/`stage_name`/`state_snapshot` are additive fields; `FailureDetector.scan_audit_log`
requires them, and `AuditWriter` now emits them on both lines (see
[Known Issues](../community/known-issues.md) for
that fix in full). The completion line's `stage_name` is a sentinel
(`"__pipeline_complete__"`) rather than a real stage id, specifically so
`DecideToTrainerBridge.extract_trajectories()` (which filters on `entry.get("stage_id") ==
stage_id`) keeps ignoring it; the completion line still carries no `stage_id`.

What's still **not** in either shape: no `human_feedback` field, and the completion line
has no top-level `verdict` key at the position `extract_dpo_pairs`/`extract_bco_labels`
expect (it's nested under `state_snapshot` instead). That's a separate, still-open gap;
see the training-data extraction section below.

### `AuditReader`

```python
class AuditReader:
    def __init__(self, path: str) -> None: ...
    def read_all(self) -> List[Dict[str, Any]]: ...
    def filter_by_stage(self, stage_id: str) -> List[Dict[str, Any]]: ...
    def filter_by_pipeline_id(self, pipeline_id: str) -> List[Dict[str, Any]]: ...
    def extract_dpo_pairs(self, stage_id: str) -> List[Dict[str, Any]]: ...
```

`read_all`/`filter_by_stage`/`filter_by_pipeline_id` are plain, correct JSONL readers;
malformed lines are skipped, nothing else surprising.

`extract_dpo_pairs(stage_id)` scans for lines where `entry.get("stage_id") == stage_id` **and**
`entry.get("human_feedback") == "rejected"`, and builds a pair from `model_output`/
`human_output`/`human_explanation`. **`AuditWriter.log_stage` never writes a
`human_feedback` field** (see the entry shape above), so this always returns an empty list
against a real DECIDE-generated log; there is no code path in `decide/` that writes
`human_feedback` at all today. Already documented in
[Known Issues](../community/known-issues.md#looks-like-it-works-doesnt-or-gives-a-quietly-wrong-answer);
this page just states the mechanism.

### `extract_bco_labels` is not on `AuditReader`

Despite the pairing suggested elsewhere, **`AuditReader` has no `extract_bco_labels`
method**; grep confirms it. The method lives on a *different* class,
`DecideToTrainerBridge` in `decide/training_bridge.py`, which wraps an `AuditReader`
internally (`self.reader = AuditReader(audit_path)`) but implements
`extract_bco_labels(output_stage_id)` itself by re-reading the JSONL file directly rather
than delegating to the reader:

```python
class DecideToTrainerBridge:
    def __init__(self, audit_path: str):
        self.reader = AuditReader(audit_path)

    def extract_dpo_pairs(self, stage_id: str) -> List[Dict[str, Any]]:
        return self.reader.extract_dpo_pairs(stage_id)   # delegates — same emptiness as above

    def extract_trajectories(self, stage_id: str) -> TrajectoryDataset: ...  # real, walks stage entries into Trajectory objects

    def extract_bco_labels(self, output_stage_id: str) -> List[Dict[str, Any]]: ...
```

`extract_bco_labels` looks for lines where `entry.get("stage_id") == output_stage_id` **and**
a top-level `"verdict"` key is present, mapping `APPROVE`→`1`, `DENY`/`REVIEW`→`0`. But per
the entry shapes above, `stage_id` only appears on `log_stage` lines and `verdict` only
appears on `write` completion lines; no single audit line carries both. Against a real log,
`extract_bco_labels` also always returns an empty list, for the same class of reason as
`extract_dpo_pairs`. `train_from_audit(..., algorithm="bco")` in the same module will raise
`ValueError("No BCO labels found...")` the moment it's pointed at a real log.

**The working alternative for both:** `CollectRunner(runner).run(inputs)` builds DPO/BCO/
GRPO-PPO-RLOO-shaped records directly from in-memory `PipelineState` during a real run,
never by re-parsing `audit.jsonl` afterward. See
[Python API: DECIDE Engine](decide-engine.md#running-episodes-and-collecting-training-data).

```python
from agenttune.decide.audit import AuditWriter, AuditReader

writer = AuditWriter(path="./audit.jsonl")
writer.log_stage(state, stage={"id": "income_agent", "type": "llm_call"},
                  result={"output": {"verdict": "APPROVE"}, "latency_ms": 340})
writer.write(state)

reader = AuditReader("./audit.jsonl")
entries = reader.read_all()
income_entries = reader.filter_by_stage("income_agent")
one_pipeline = reader.filter_by_pipeline_id(state.pipeline_id)
dpo_pairs = reader.extract_dpo_pairs("income_agent")  # [] against a real log, per above
```

## `utils/logging.py`: `LoggingManager`

Unified metrics logging across console, WandB, and TensorBoard, used by the general
training utilities (separate from the SFT-specific `SFTLogger` below).

```python
class LoggingManager:
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "./logs",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_project: str = "agenttune",
        wandb_entity: Optional[str] = None,
        log_level: str = "INFO",
    ): ...

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None: ...
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None: ...
    def log_model_graph(self, model, input_sample) -> None: ...
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None: ...
    def watch_model(self, model, log_freq: int = 100) -> None: ...
    def save_model_artifact(self, model_path: str, artifact_name: str, artifact_type: str = "model") -> None: ...
    def finish(self) -> None: ...
    # context-manager: __enter__ returns self, __exit__ calls finish()
```

Behavior notes:

- `WANDB_AVAILABLE`/`TENSORBOARD_AVAILABLE` are module-level flags set once at import time
  by `try`/`except ImportError`. `use_wandb=True` only actually turns WandB on if
  `wandb` is importable (`self.use_wandb = use_wandb and WANDB_AVAILABLE`); same pattern for
  TensorBoard. Passing `use_wandb=True` on a machine without `wandb` installed silently
  degrades to no-op, it does not raise.
  Symmetrically, `output_dir` is created either way (`Path.mkdir(parents=True, exist_ok=True)`)
  and console logging is *always* configured via `_setup_console_logging`, which calls
  `logging.basicConfig(...)` with both a `StreamHandler` and a `FileHandler` writing to
  `<output_dir>/<experiment_name>.log`; this is real console+file logging with zero
  optional dependencies.
- `_setup_wandb`/`_setup_tensorboard` each wrap their init call in `try`/`except`; on failure
  they log a warning and flip the corresponding `use_*` flag back to `False` rather than
  raising.
- `log_metrics`/`log_hyperparameters`/`log_text`/`watch_model`/`save_model_artifact` all
  independently check `self.use_wandb and self.wandb_run` / `self.use_tensorboard and
  self.tensorboard_writer` before doing anything backend-specific, and wrap each backend
  call in its own `try`/`except`; a WandB failure won't stop the TensorBoard write or vice
  versa.
- `finish()` calls `wandb.finish()` / `tensorboard_writer.close()` as applicable; safe to
  call even if a backend was never enabled.

```python
from agenttune.utils.logging import LoggingManager

# console-only, zero optional deps
with LoggingManager(experiment_name="run-1") as log:
    log.log_metrics({"loss": 0.42, "accuracy": 0.91}, step=100)

# WandB + TensorBoard (needs `wandb` and `tensorboard` installed)
log = LoggingManager(
    experiment_name="run-2",
    use_wandb=True,
    use_tensorboard=True,
    wandb_project="agenttune-experiments",
)
log.log_hyperparameters({"lr": 2e-4, "batch_size": 8})
log.log_metrics({"loss": 0.31}, step=200)
log.finish()
```

### Config-builder helpers

Three small functions that build the `logging_config` dict `create_logging_manager` expects;
all verified present under these exact names:

```python
def create_logging_manager(experiment_name: str, logging_config: Dict[str, Any]) -> LoggingManager: ...
def create_wandb_config(project="agenttune", entity=None, tags=None, notes=None) -> Dict[str, Any]: ...
def create_tensorboard_config(log_dir="./logs/tensorboard") -> Dict[str, Any]: ...
def create_full_logging_config(
    wandb_project="agenttune", wandb_entity=None,
    tensorboard_dir="./logs", log_level="INFO",
) -> Dict[str, Any]: ...
```

`create_logging_manager` just unpacks a dict into `LoggingManager(...)` kwargs (with
defaults for every key it might be missing). `create_wandb_config`/`create_tensorboard_config`
each build a *partial* config dict for one backend; `create_full_logging_config` builds one
dict with both `use_wandb` and `use_tensorboard` set `True`. None of the three merge with
each other; if you want WandB + TensorBoard from the individual helpers, you must manually
merge the two dicts yourself (`create_full_logging_config` exists precisely to skip that
step).

```python
from agenttune.utils.logging import create_full_logging_config, create_logging_manager

cfg = create_full_logging_config(wandb_project="agenttune-experiments")
manager = create_logging_manager("run-3", cfg)
```

## `core/sft/logging.py`: `SFTLogger`

A second, narrower logging class specific to the SFT config layer; takes a
[`LoggingConfig`](configuration-classes.md#loggingconfig) dataclass rather than a raw dict.

```python
class SFTLogger:
    def __init__(self, config: LoggingConfig): ...
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None: ...
    def log_config(self, config: Any) -> None: ...
    def close(self) -> None: ...
```

- `__init__` creates `config.output_dir`, then iterates `config.loggers` (a `List[str]`,
  default `["tensorboard"]`) and calls `_setup_tensorboard()` / `_setup_wandb()` for each
  recognized name, warning on anything else. Both setup methods lazily import their backend
  and catch `ImportError` with a warning, same "degrade, don't crash" pattern as
  `LoggingManager`. Successfully-initialized backends land in `self.loggers: Dict[str, Any]`
  keyed by name (`"tensorboard"` → a `SummaryWriter`, `"wandb"` → the `wandb` module itself).
- `log_metrics` iterates `self.loggers` and calls `add_scalar` (TensorBoard) or `.log(...)`
  (WandB, with a `"step"` key mixed into the metrics dict) per active backend, each wrapped
  in its own `try`/`except`.
- `log_config` writes the config as a `add_text("config", ...)` blob on TensorBoard (there's
  no native TensorBoard config API), or `wandb.config.update(...)` on WandB. Accepts either
  a dataclass with `.to_dict()` or any object with `.__dict__`.
- `close()` calls `.close()`/`.finish()` per backend and clears `self.loggers`.

Since `wandb.init(project="agenttune-sft", ...)` is called unconditionally inside
`_setup_wandb` whenever `"wandb"` is in `config.loggers`, there's no project-name override
here the way `LoggingManager` exposes via `wandb_project`; it's hardcoded to
`"agenttune-sft"`.

```python
from agenttune.core.sft.config import LoggingConfig
from agenttune.core.sft.logging import SFTLogger

logger = SFTLogger(LoggingConfig(output_dir="./output/sft", loggers=["tensorboard"]))
logger.log_config(sft_config)          # sft_config: an SFTConfig instance
logger.log_metrics({"train_loss": 0.55}, step=50)
logger.close()
```

**Rating A** for `LoggingManager`/`SFTLogger` console+TensorBoard-off paths; TensorBoard
itself needs `torch`+`tensorboard` installed but nothing beyond that; **rating B** the
moment `use_wandb=True` or `"wandb"` is in `loggers`; needs the `wandb` package and (for a
real run, not just local mode) network + an API key. **Rating A** for `AuditWriter`/
`AuditReader.read_all`/`filter_by_*`, pure stdlib JSONL I/O, no dependencies. **Rating F**
in practice for `extract_dpo_pairs`/`extract_bco_labels` against a real log, per above; use
`CollectRunner` instead.
