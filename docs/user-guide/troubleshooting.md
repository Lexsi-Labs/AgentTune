# Troubleshooting

Real problems, found by reading the code and (where possible) reproducing them, not a
generic FAQ template. Each entry is Problem → Cause → Fix. If you hit something not listed
here, check [Known Issues](../community/known-issues.md) first; it's the fuller audit this
page draws from.

## `ImportError`/`ModuleNotFoundError: No module named 'langchain_community'` from a tool that doesn't need it

**Problem**: You call `ToolRegistry.get("read_file")` (or any other tool name, including one
you just registered yourself) and get a `ModuleNotFoundError` for `langchain_community`,
even though the tool you asked for has nothing to do with LangChain.

```pycon
>>> from agenttune.agentic.tools.registry import ToolRegistry
>>> ToolRegistry.get("read_file")
Traceback (most recent call last):
  ...
ModuleNotFoundError: No module named 'langchain_community'
```

**Cause**: `ToolRegistry.get()` calls `auto_register_builtins()` exactly once per process,
the first time *any* tool is fetched (`registry.py`):

```python
@classmethod
def get(cls, name: str) -> BaseTool:
    if not cls._builtins_registered:
        cls.auto_register_builtins()
    ...
```

`auto_register_builtins()` unconditionally imports **every** builtin module, including
`slack.py`, `github.py`, `playright.py`, and `sql.py`, all of which do
`from langchain_community... import ...` at module level. If `langchain_community` isn't
installed, that import raises before your requested tool is ever looked up, regardless of
which tool you asked for. This also means `register_custom()` doesn't save you: it doesn't
touch `_builtins_registered`, so the very next `.get()` call (yours or anyone else's in the
same process) still tries to import everything first.

**Fix**: two options, depending on what you need.

1. **Install `langchain_community`** if you actually want the tools that need it (Slack,
   GitHub, Playwright, SQL, web search). It's a base dependency in `pyproject.toml`, so
   `pip install -e .` from a clean checkout should already include it; this failure mode
   shows up when it's missing from a partial or pinned install rather than a clean
   `pip install -e .`.

2. **Skip the registry entirely** for the zero-dependency tools; import the classes
   directly from their `builtin/` module. This is the reliable fix if you only need
   `read_file`/`write_file`/`list_dir`/`run_python`/`run_bash`/`grep`:

   ```python
   from agenttune.agentic.tools.builtin.file_tools import ReadFileTool
   tool = ReadFileTool()
   result = tool.execute(path="README.md")
   ```

3. **If you need the registry specifically** (e.g. for `ToolExecutor`, which takes a
   registry object) but want to avoid pulling in the langchain-backed builtins, register
   your own tools and set the "already registered" flag yourself before the first `.get()`:

   ```python
   from agenttune.agentic.tools.registry import ToolRegistry
   from my_tools import MyTool

   ToolRegistry.register_custom(MyTool())
   ToolRegistry._builtins_registered = True   # skip auto_register_builtins() entirely
   tool = ToolRegistry.get("my_tool")          # works — no langchain_community needed
   ```

   This is reaching into a private attribute, not a supported public API, but it's the
   only lever the current code exposes, and it's verified to work (confirmed against the
   actual `registry.py`). It also means none of the builtin tools (including the
   zero-dependency ones) will be registered; register those explicitly too if you need
   them alongside your own.

See [Tool Library](tool-library.md) and [Known Issues](../community/known-issues.md) for
the full tool-by-tool dependency breakdown.

## The closed loop detects nothing against an old audit log

**Problem**: You point `SelfHealingPipeline.run_once()` or `FullClosedLoop.ingest_once()` /
`run_forever()` at a real DECIDE-generated `audit.jsonl`, and it runs without producing a
single `Failure`, even though you know some of those runs failed or scored below your
judge threshold.

**Cause**: `FailureDetector.scan_audit_log` (`decide/closed_loop/failure_detector.py`)
requires every audit line to carry `trajectory_id`, `stage_name`, and a dict
`state_snapshot`. `AuditWriter` now writes all three (as of the current release), so a
*freshly generated* `audit.jsonl` is detected correctly; confirmed against a real
`bfsi/kyc_triage` run that genuinely loops: `scan_audit_log` reports the 6
`loop_collapse` failures, and `FullClosedLoop.ingest_once()` buffers all 6 training
examples end-to-end. If you're still seeing zero detections, the log itself predates
this fix; regenerate it by re-running the pipeline, rather than replaying an old
`audit.jsonl` written by an older `AuditWriter`.

`FailureClassifier`, `TrainingExampleGenerator`, `TrainingBuffer`, `RetrainingTrigger`,
`BackgroundRetrainer`, and `DeploymentGate` were already real and correctly wired
downstream of detection; nothing there changed.

Constructing `Failure` objects yourself (from `decide/closed_loop/contracts.py`) and
feeding them directly into `FailureClassifier`/`TrainingExampleGenerator`, bypassing
`scan_audit_log` entirely, remains a valid alternative when your failure signal doesn't
come from a DECIDE audit log at all (a labelled test set, external monitoring, manual
review):

```python
import asyncio
from agenttune.decide.closed_loop.contracts import Failure
from agenttune.decide.closed_loop.failure_classifier import FailureClassifier
from agenttune.decide.closed_loop.training_example_generator import TrainingExampleGenerator
from agenttune.decide.closed_loop.replay_validator import ReplayValidator

failures = [
    Failure(
        trajectory_id="traj-001",
        failure_type="low_judge_score",
        failed_stage_name="llm_call_answer",
        context={"input": "...", "output": "..."},
        judge_score=0.2,
    ),
]

classifier = FailureClassifier(model_name="gpt-4o-mini")
classified = asyncio.run(classifier.classify_batch(failures))

generator = TrainingExampleGenerator(validator=ReplayValidator(), model_name="gpt-4o-mini")
examples = asyncio.run(generator.generate_batch(classified))
# examples are real TrainingExample objects — feed them into TrainingBuffer as usual
```

`DeploymentGate.build_test_set()` and `RewardDriftTracker.load_from_audit()` already
matched the real `AuditWriter` schema independently of this fix. `CollectRunner` remains
the recommended way to produce DPO/BCO/GRPO training records directly from a live run
(no audit-log parsing involved either way); see [Collecting training data:
`CollectRunner`](decide-workflows.md#collecting-training-data-collectrunner). See
[Concepts: DECIDE & the closed loop](../concepts/decide-and-closed-loop.md) and [Known
Issues](../community/known-issues.md).

## `RougeMetric`/`BleuMetric` return `0.0` with no error

**Problem**: You run an eval with `RougeMetric()`/`BleuMetric()` and every score comes back
`0.0`, no exception, no obviously wrong output, just a suspiciously perfect zero.

**Cause**: both metrics (`agenttune.eval.metrics.text`) catch `ImportError` in their
constructor and fall back to a disabled state instead of raising:

```python
class RougeMetric(Metric):
    def __init__(self):
        super().__init__("rouge")
        self.available = False
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            self.available = True
        except ImportError:
            logger.warning("rouge_score not installed. ROUGE metric will return 0.")
```

If `self.available` is `False`, `compute()` unconditionally returns
`{"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}` (and the snake_case aliases derived from
them) without ever touching your predictions/references. `BleuMetric` does the same thing
for `nltk`, returning `{"bleu": 0.0}`. The only signal this happened is a `logger.warning`
call, easy to miss if your logging isn't configured to show warnings, and the eval run
itself reports success.

**Fix**: install the metric's real dependency, and don't trust a `0.0` from these two
metrics until you've confirmed the package is importable:

```bash
pip install rouge-score nltk
```

```python
python -c "from rouge_score import rouge_scorer; print('ok')"
python -c "from nltk.translate.bleu_score import sentence_bleu; print('ok')"
```

Both packages are part of `pyproject.toml`'s base `dependencies`, so a clean
`pip install -e .` includes them already; this failure mode mainly shows up in a partial
or pinned install that skips some of the base dependencies. See
[Known Issues](../community/known-issues.md); the same
silently-return-zero pattern also applies to `DistilledJudge` when `transformers` is
missing or its checkpoint fails to load.

## Connecting to Groq / OpenRouter / a local API endpoint

**Problem**: unclear whether to pass a provider-prefixed model string, a plain model name,
or a `base_url`, and which combination actually works, when pointing `APIRolloutEngine` (or
`create_rollout_fn`/`create_rollout_engine`) at something other than plain OpenAI.

**Cause / how it actually works**: `APIRolloutEngine` (`agentic/rollout_engines/api_engine.py`)
is a thin wrapper around `litellm.completion(**call_kwargs)`. There are two genuinely
different, both-supported ways to point it at a model, and they answer different needs:

1. **Provider-prefixed model string**: e.g. `"groq/llama-3.3-70b-versatile"`,
   `"openrouter/meta-llama/llama-3.1-70b-instruct"`, `"together_ai/mistralai/Mixtral-8x7B-v0.1"`,
   `"ollama/llama3"`. LiteLLM parses the prefix and routes to that provider's real endpoint
   internally; you never set `base_url` for this. The engine also auto-resolves the API
   key from a per-provider environment variable via `_provider_prefix()` /
   `_ENV_VAR_MAP` (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHERAI_API_KEY`, ...; `ollama`
   needs no key at all); pass `api_key=` explicitly only to override.

2. **Plain model name + explicit `base_url`**: bypasses LiteLLM's provider-prefix routing
   entirely, for a local vLLM/TGI/llama.cpp server, a self-hosted OpenAI-compatible proxy,
   or a LiteLLM proxy server. The docstring's own example: `model="gpt-4o-mini",
   base_url="http://localhost:4000"`. In this mode, `_provider_prefix()` won't match a
   known provider from an arbitrary model name, so the automatic env-var key lookup won't
   fire; **pass `api_key=` explicitly** if your endpoint needs one.

The real constructor parameter is **`base_url`**, confirmed directly from
`APIRolloutEngine.__init__(self, model="gpt-4o-mini", api_key=None, base_url=None, ...)`
and the `litellm.completion()` call site, which does `call_kwargs["base_url"] =
self.base_url` when set. There is no `api_base_url` parameter at this layer.

```python
# Groq, letting litellm route natively — no base_url needed
from agenttune.agentic.rollout_engines.api_engine import APIRolloutEngine
engine = APIRolloutEngine(model="groq/llama-3.3-70b-versatile")  # reads GROQ_API_KEY

# A local OpenAI-compatible server — plain name + explicit base_url + explicit key
engine = APIRolloutEngine(
    model="my-local-model",
    base_url="http://localhost:8000/v1",
    api_key="not-needed-but-some-servers-require-a-placeholder",
)
```

**One layer up, the parameter name changes.** `create_rollout_fn(...)` and
`create_rollout_engine(...)` (`rollout_factory.py`) expose this same value under the
parameter name **`api_base_url`**, and forward it internally as `base_url=api_base_url`
when constructing the engine:

```python
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

rollout_fn = create_rollout_fn(
    rollout_backend="api",
    api_model="groq/llama-3.3-70b-versatile",
    tools=[my_tool],
)

# Local/self-hosted endpoint at the create_rollout_fn layer:
rollout_fn = create_rollout_fn(
    rollout_backend="api",
    api_model="my-local-model",
    api_base_url="http://localhost:8000/v1",   # note: api_base_url here, not base_url
    api_key="...",
    tools=[my_tool],
)
```

So: use `base_url` when constructing `APIRolloutEngine` directly; use `api_base_url` when
going through `create_rollout_fn`/`create_rollout_engine`. Mixing the two up (e.g. passing
`base_url=` to `create_rollout_fn`) silently does nothing useful; it isn't a recognized
parameter there, so it either errors on an unexpected keyword or gets absorbed by
`**engine_kwargs` depending on how you called it, neither of which sets the actual base URL.

One more provider quirk baked into the same file: Groq, Ollama, and Together are in
`_NO_TOOL_CHOICE_PROVIDERS`; the engine still sends `tools=`, but skips
`tool_choice="auto"` for them, because those providers either reject or mishandle that
param.

## Qwen3 thinking traces (`enable_thinking`)

**Problem**: rollouts against a Qwen3-family model include a `<think>...</think>` reasoning
block you didn't ask for, inflating completion length and confusing reward functions that
expect a plain final answer.

**Cause / real support**: `enable_thinking` is a genuine parameter throughout the rollout
stack, not something you have to hand-roll. `create_rollout_fn(..., enable_thinking: bool =
False, ...)` (`rollout_factory.py`) defaults to `False` and threads the value down to every
rollout engine's chat-template application
(`transformers_engine.py`/`vllm_engine.py`), which calls:

```python
tokenizer.apply_chat_template(**common, enable_thinking=enable_thinking)
```

The same default (`False`) is repeated in every TRL agentic backend
(`agentic_grpo.py`, `agentic_dpo.py`, `agentic_ppo.py`, `agentic_rloo.py`, `agentic_bco.py`
under `backends/trl/agentic/`), all reading it via
`_get(self.kwargs, "enable_thinking", default=False)`.

**Fix**: it's off by default already; if you're seeing thinking traces, check that you (or
a config layer above you) haven't explicitly set `enable_thinking=True` somewhere. To
enable thinking on purpose (e.g. to compare thinking vs. non-thinking rollouts), pass it
explicitly at whichever layer you're calling:

```python
from agenttune.core.backend_factory import create_agentic_trainer

trainer = create_agentic_trainer(
    algorithm="grpo",
    model="Qwen/Qwen3-8B",
    enable_thinking=False,   # explicit — this is also the default
    tools=[my_tool],
    train_dataset=my_dataset,
    output_dir="./runs/grpo-qwen3",
)
```

or, calling the rollout factory directly:

```python
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

rollout_fn = create_rollout_fn(
    rollout_backend="vllm",
    model_path="Qwen/Qwen3-8B",
    enable_thinking=False,
    tools=[my_tool],
)
```

Note this is specifically the tokenizer-facing `enable_thinking` chat-template kwarg, and
only applies to the local (`transformers`/`vllm`) rollout engines; `APIRolloutEngine` has
no local tokenizer at all (`self._tokenizer = None`), so if you're hitting Qwen3 through an
OpenAI-compatible API endpoint instead, disable thinking the way that endpoint expects (for
many OpenAI-compatible servers, that's
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on the request, not an
AgentTune-level parameter).

## A few more, briefly

- **`agenttune.eval.cli` can't be imported at all**: it imports `EvaluationRegistry` from
  `.registry`; the real class there is named `EvalRegistry`. Nothing in the repo has a
  console-script entry point for it, so this only bites if you import the module directly.
- **The plain `"placeholder_coverage_reward"`/`"answer_correctness_reward"`/`"format_reward"`
  registry keys still resolve to `use_case.py`'s simpler versions** (unchanged, for backward
  compatibility) rather than `finqa.py`'s more numerically-careful ones, but the stronger
  versions are now reachable too, under their own keys: `"finqa_placeholder_coverage_reward"`,
  `"finqa_answer_correctness_reward"`, `"finqa_format_reward"`, `"finqa_sql_grounding_reward"`,
  `"finqa_calculator_grounding_reward"`, and `sql.py`'s own `format_reward` under
  `"sql_format_reward"`. Use those keys, or import `finqa.py`'s functions directly by module
  path, if you want the more careful numeric parsing.

Full details on all of these are in [Known Issues](../community/known-issues.md), which is
the primary audit this troubleshooting page draws from.
