# Python API: Unified SDK

`agenttune.api`: the *stable, documented* facade the package's own top-level `__init__.py`
re-exports. Everything here is a thin wrapper over internals that are otherwise
undocumented and free to change; that's the point of this module existing at all.

```python
from agenttune import run_pipeline, arun_pipeline, train_agentic, PipelineResult
```

`agenttune/__init__.py` does exactly this re-export; those four names (plus `__version__`)
are the entire top-level public surface. Everything else under `agenttune.*` is an internal
import path, not part of this contract.

## `PipelineResult`

```python
@dataclass(frozen=True)
class PipelineResult:
    pipeline_id: str
    template_id: str
    verdict: Any
    verdict_label: Any
    confidence: Any
    reason: Any
    step_count: int
    elapsed_seconds: float
    is_complete: bool
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]: ...
```

The outcome of a single DECIDE pipeline run (inference mode), a frozen dataclass built by
`_state_to_result()` from the `PipelineState` that `GraphRunner.run()`/`run_sync()`
produces, field-for-field (same names on both sides). `to_dict()` is `dataclasses.asdict(self)`,
a plain nested dict, no custom serialization logic.

```python
result = run_pipeline(
    "bfsi/kyc_triage",
    "Customer: Jane Doe, DOB 1985-03-12, address 45 Oak Ave Denver, income $85,000 "
    "annual, employed at Globex Corp, provided passport and utility bill, PEP status: no.",
)
result.verdict        # e.g. "DENY"
result.to_dict()       # {"pipeline_id": ..., "template_id": ..., "verdict": ..., ...}
```

A vague input with no extractable name/DOB/income (e.g. just `"some input text"`) makes
`extract`'s quality-gate retry loop run all the way to `max_total_steps` before giving up
with `verdict=None`; give it a document that actually has the fields the template asks
for, as above, for a fast, real decision.

## `run_pipeline` / `arun_pipeline`

```python
async def arun_pipeline(template: str, input_text: str, *, config: Optional[str] = None) -> PipelineResult: ...
def       run_pipeline(template: str, input_text: str, *, config: Optional[str] = None) -> PipelineResult: ...
```

| Param | Meaning |
|---|---|
| `template` | Template ID (e.g. `"bfsi/kyc_triage"`) or a path to a template YAML |
| `input_text` | Input passed to the pipeline's first stage |
| `config` | Optional path to a global `config.yaml`; defaults to `"./config.yaml"` |

Both are thin wrappers over `agenttune.decide.graph_runner.GraphRunner`:

```python
runner = GraphRunner.from_template(template, config or "./config.yaml")
state = await runner.run(input_text)       # arun_pipeline
return _state_to_result(state)
```

`run_pipeline` is `asyncio.run(arun_pipeline(...))`: a synchronous convenience wrapper.
**Do not call `run_pipeline` from inside a running event loop** (e.g. a FastAPI handler);
`asyncio.run` will raise; use `arun_pipeline` there directly, and `await` it.

Both import `GraphRunner` lazily, inside the function body, not at module load, so
importing `agenttune.api` (or `agenttune` itself) never pulls in the DECIDE stack. See
[Python API: DECIDE Engine](decide-engine.md) for what `GraphRunner` itself does (YAML→
`langgraph` compilation, observation-schema validation, step-limit enforcement, audit
logging, destination routing); this page only documents the SDK entry point, not
`GraphRunner`'s internals.

Whether a given call needs a GPU or an API key depends entirely on the template's stages,
not on this function; the orchestration itself has no hard dependency (Rating B in
`decide-engine.md`'s terms).

## `train_agentic`

```python
AGENTIC_ALGORITHMS = ("grpo", "dpo", "ppo", "rloo", "bco")

def train_agentic(algorithm: str, **kwargs: Any): ...
```

A thin wrapper over `agenttune.core.backend_factory.create_agentic_trainer`; see
[Python API: `agenttune.core`](core-api.md) for its full signature and parameter routing:

```python
algo = algorithm.lower()
if algo not in AGENTIC_ALGORITHMS:
    raise ValueError(f"Unknown algorithm {algorithm!r}. Valid algorithms: ...")
from agenttune.core.backend_factory import create_agentic_trainer
return create_agentic_trainer(algo, **kwargs)
```

The validation against `AGENTIC_ALGORITHMS` happens **before** the `create_agentic_trainer`
import; a bad algorithm string fails fast with a plain `ValueError`, without importing
`agenttune.core.backend_factory` (which itself tries to import all five TRL trainer
wrapper classes at module load, pulling in `torch`/`transformers`/`trl`). Passing a valid
algorithm name still requires that heavy stack to actually be installed; the eager check
only short-circuits the *invalid-name* case.

`**kwargs` is forwarded verbatim; see [Python API: `agenttune.core`](core-api.md) for the
full per-algorithm parameter routing table (which kwargs reach TRL's `*Config`, which
reach TRL's `*Trainer`, which are AgentTune-only data params). The returned object exposes
`.train()`.

```python
trainer = train_agentic(
    "grpo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=my_reward_fn,
    tools=[my_tool],
    train_dataset=my_dataset,
    output_dir="./runs/grpo",
    max_steps=100,
)
results = trainer.train()
```

Needs a GPU and a compatible `trl`/`torch` stack; see
[RL Training](../user-guide/rl-training.md) for the full setup walkthrough and
[Getting Started](../getting-started/installation.md) for the install.

## Relationship to the CLI

`agenttune train` and `agenttune pipeline` are Typer commands that call `train_agentic` and
`run_pipeline` respectively; the CLI adds no logic of its own beyond argument parsing and
printing the result; it is not a second implementation. See [CLI](cli.md) for the full
command reference (the `agenttune` console script, resolving to `agenttune.cli:main`)
rather than duplicating it here. Anything you can do with `agenttune train --algorithm grpo
...` you can do identically, with more control over kwargs, by calling `train_agentic`
directly from this module.

## Honesty notes

- `run_pipeline`/`arun_pipeline` are as reliable as the DECIDE stack they delegate to,
  real and well-tested for inference mode. They do **not** run collection, eval, or training
  modes; those live on `CollectRunner`/`EvalRunner`/the `TrainerConfigBridge`, one level
  below this SDK surface; see [DECIDE Engine](decide-engine.md).
- `train_agentic` supports exactly the same five algorithms as `create_agentic_trainer`:
  `grpo`, `dpo`, `ppo`, `rloo`, `bco`.
- Nothing in `agenttune.api` validates that `torch`/`transformers`/`trl` are installed
  before you call `train_agentic` with a *valid* algorithm name; only the algorithm string
  itself is checked eagerly. The import error (if the stack is missing) surfaces from
  inside `create_agentic_trainer`, not from this wrapper.
