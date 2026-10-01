# RL Training

`create_agentic_trainer` is the one factory function that actually trains a model in this
repo today. Per [Known Issues](../community/known-issues.md), `core/sft` is broken end to
end. **Agentic RL training is the only training path that runs.** Everything below is
real: read straight from `src/agenttune/core/backend_factory.py` and the five
`backends/trl/agentic/*/agentic_*.py` wrappers it dispatches to.

```python
from agenttune.core.backend_factory import create_agentic_trainer, AgenticAlgorithm

list(AgenticAlgorithm)
# [<AgenticAlgorithm.GRPO: 'grpo'>, <AgenticAlgorithm.DPO: 'dpo'>,
#  <AgenticAlgorithm.PPO: 'ppo'>, <AgenticAlgorithm.RLOO: 'rloo'>,
#  <AgenticAlgorithm.BCO: 'bco'>]
```

`create_agentic_trainer(algorithm, **kwargs)` is a thin dispatcher: it lower-cases
`algorithm`, looks it up in that enum, and instantiates the matching wrapper class with
every kwarg forwarded verbatim:

```python
# src/agenttune/core/backend_factory.py
def create_agentic_trainer(algorithm: str, **kwargs) -> Any:
    ...
    trainer_class = _REGISTRY.get(alg_enum)
    return trainer_class(**kwargs)
```

`grpo` → `TrlAgenticGrpo`, `dpo` → `TrlAgenticDPO`, `ppo` → `TrlAgenticPPO`,
`rloo` → `TrlAgenticRloo`, `bco` → `TrlAgenticBCO`. All five live under
`src/agenttune/backends/trl/agentic/`, all five are **kwargs-driven wrappers around a real
TRL trainer**; there is no AgentTune config object. Each wrapper uses
`inspect.signature()` on the installed TRL's `*Config` and `*Trainer` classes to split
your kwargs into "goes to the Config" vs "goes to the Trainer" automatically, so the
accepted parameter surface tracks whatever `trl` version you have installed rather than a
frozen AgentTune-side schema (see [Notes & gotchas](#notes-gotchas) for what that means in
practice).

There are only 5 dispatchable algorithms.

For per-algorithm theory, hyperparameter tables, and when to reach for which, see
[Algorithms Overview](../algorithms/overview.md). This page is the API reference and
worked examples for actually calling the five real trainers.

## Quick Start

This trains a tiny model to use a calculator tool with GRPO, self-contained, no dataset
download, no external API. It mirrors the pattern in `TrlAgenticGrpo`'s own docstring
("Agentic usage ... matches the BioGRID notebook pattern exactly"): a tool, a reward
function list, a pre-built `train_dataset`, `create_agentic_trainer`, `.train()`.

### 1. Define a tool

Tools are plain Python callables, no base class required. AgentTune builds the model-
facing JSON schema from your type hints and docstring via
`transformers.utils.get_json_schema`, so give the function real annotations and a
Google-style `Args:` block:

```python
import ast
import operator


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Args:
        expression: An arithmetic expression using +, -, *, /, and parentheses,
            e.g. "12 * (3 + 4)".
    """
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {expression}")

    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"
```

You could just as well pass a `BaseTool` instance's bound method, e.g.
`SQLDatabaseTool("sqlite:///mydb.db").query` (see [SQL tool below](#a-realistic-tool-sql)),
or a `langchain_community`-backed builtin. A bare function like this one has zero extra
dependencies, which is why it's the Quick Start choice.

### 2. Write reward functions: agentic mode

`TrlAgenticGrpo`'s own docstring claims two reward calling conventions:

```text
Standard mode  : f(completions: list[str], **kwargs) -> list[float]
Agentic mode   : f(completions: list[list[dict]], **kwargs) -> list[float]
    where each completion is a list of message dicts, e.g.:
    [{"role": "assistant", "tool_calls": [...]},
     {"role": "tool", "content": "..."},
     {"role": "assistant", "content": "*Yes*"}]
```

!!! warning "That docstring doesn't match what actually gets passed"
    Verified live against a real GRPO run with `tools=[calculator]`: each `completions[i]`
    is a plain **string**: the flattened rollout text (tool-call XML tag, tool result, and
    final answer all concatenated), e.g.
    `'<tool_call>\n{"name": "calculator", "arguments": {"expression": "12 * (3 + 4)"}}\n</tool_call>84The result of 12 * (3 + 4) is 84.'`,
    **not** `list[list[dict]]`. A reward function written against the docstring's claimed
    shape (iterating `turns` and calling `t.get("role")`) crashes with `AttributeError: 'str'
    object has no attribute 'get'` the moment GRPO actually calls it. Write agentic-mode
    reward functions against the real (string) shape instead, as below.

Two reward functions, combined later with weights:

```python
def used_calculator_reward(completions, **kwargs):
    """+0.2 if the completion text shows a tool call was made.

    In agentic mode, completions[i] is the flattened completion TEXT for one rollout (a
    str), not a list of {"role", ...} message dicts, check for the literal tag instead.
    """
    return [0.2 if "<tool_call>" in str(c) else 0.0 for c in completions]


def correct_answer_reward(completions, answer=None, **kwargs):
    """+1.0 if the completion text contains the gold numeric answer.

    `answer` arrives here because it's an extra column on `train_dataset`; TRL forwards
    every non-standard dataset column to reward_funcs as a matching kwarg.
    """
    golds = answer or [None] * len(completions)
    scores = []
    for c, gold in zip(completions, golds):
        scores.append(1.0 if gold is not None and str(gold) in str(c) else 0.0)
    return scores
```

Verified with a real run (`Qwen/Qwen2.5-0.5B-Instruct`, `max_steps=1`): this pair produces
a real non-zero, non-degenerate reward signal (`rewards/combined_reward(...)/mean: 0.64`),
whereas the dict-based version above crashes before completing a single step.

### 3. Build the dataset

`train_dataset` just needs a `prompt` column (plain string or a conversational
`list[dict]`); any extra columns (here `answer`) are forwarded to your reward functions
as kwargs by TRL.

```python
from datasets import Dataset

problems = [
    ("What is 12 * (3 + 4)?", "84"),
    ("What is 100 / 4 - 5?", "20"),
    ("What is (9 + 1) * 6?", "60"),
    ("What is 15 - 3 * 2?", "9"),
]

train_dataset = Dataset.from_dict({
    "prompt": [p for p, _ in problems],
    "answer": [a for _, a in problems],
})
```

### 4. Train

```python
from agenttune.core.backend_factory import create_agentic_trainer

trainer = create_agentic_trainer(
    algorithm="grpo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    tools=[calculator],
    reward_funcs=[correct_answer_reward, used_calculator_reward],
    reward_weights=[0.8, 0.2],
    train_dataset=train_dataset,
    system_prompt=(
        "You are a calculator agent. Use the calculator tool for arithmetic, "
        "then give the final numeric answer."
    ),
    output_dir="./runs/grpo-calculator",
    num_generations=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # default is 16, with only 4 toy examples,
                                     # the default effective batch (4*16=64) is
                                     # larger than the whole dataset, so the
                                     # dataloader silently has zero batches and
                                     # training exits at step 0. See the
                                     # gotcha under Config options table below.
    max_completion_length=256,
    max_steps_per_turn=6,     # tool-call loop depth per rollout, NOT total training steps
    max_steps=30,             # GRPOConfig's own field, total optimizer steps
    beta=0.04,
)

results = trainer.train()
print(results["final_loss"], results["total_steps"])
```

What happens under the hood, straight from `agentic_grpo.py`:

- Passing `tools=[calculator]` sets `self._agentic_mode = True` at construction time.
- `setup_trainer()` runs your two reward functions through
  `combine_rewards([correct_answer_reward, used_calculator_reward], weights=[0.8, 0.2])`
  into one weighted callable, because `reward_funcs` is required and GRPO always routes it
  through `combine_rewards`.
- Because `tools` is set and no `rollout_func` was passed, it builds one via
  `create_rollout_fn(tools=[calculator], max_steps=6, ...)`, the function that
  actually runs the multi-turn tool-calling loop.
- `GRPOTrainer._generate_single_turn` gets monkey-patched at the **class** level (once,
  process-wide) to call that `rollout_func` instead of TRL's native generation path. The
  module docstring explains why: "the installed TRL can't run those tools natively (native
  `tools=` needs transformers>=5 and a json-schema-able callable, not a `BaseTool` object)".
- `tools`/`rollout_func` are popped back out of the kwargs handed to `GRPOTrainer.__init__`
  (so TRL's own native tool loop never engages) and `rollout_func` is re-attached to the
  trainer instance afterward.
- `trainer.train()` calls `setup_data()` → `setup_trainer()` → `GRPOTrainer.train()` →
  `save_model(output_dir)`, and returns `get_training_stats()`, a JSON-serialisable dict
  with `final_loss`, `total_steps`, `agentic_mode`, `tools`, and your full kwargs.

!!! note "This needs a GPU"
    Every wrapper here is a real TRL trainer underneath. The CLI's own docstring says it
    plainly: *"agentic training requires a GPU and a compatible trl/torch stack."* There's
    no CPU-only smoke-test path. See the [Local Notebooks](../notebooks/local-notebook.md) index
    for a run that actually executes.

### A realistic tool: SQL

If you want something closer to a real use case than a calculator,
`agenttune.agentic.tools.builtin.sql.SQLDatabaseTool` is real and has three tool-shaped
entry points confirmed straight from its source:

```python
from agenttune.agentic.tools.builtin.sql import SQLDatabaseTool

tool = SQLDatabaseTool("sqlite:///biogrid.db")

# Option A: bind a query method that returns list[tuple] or an error dict:
#   tool.query(sql_command: str) -> list
trainer = create_agentic_trainer(
    algorithm="grpo", model="Qwen/Qwen3-1.7B",
    tools=[tool.query], reward_funcs=[correct_answer_reward],
    train_dataset=train_dataset, output_dir="./runs/grpo-sql",
)

# Option B: a named tool with a custom schema description, for a cleaner tool name
# in the model's function-calling schema:
query_biogrid = tool.make_query_tool(
    name="query_biogrid",
    description="Query the BioGRID protein interaction database.",
)
trainer = create_agentic_trainer(
    algorithm="grpo", model="Qwen/Qwen3-1.7B",
    tools=[query_biogrid], reward_funcs=[correct_answer_reward],
    train_dataset=train_dataset, output_dir="./runs/grpo-sql",
)
```

`SQLDatabaseTool` also exposes `create_from_dataset(dataset_name, table_name="data",
split="train")` to build a fresh SQLite file from any HF dataset, and `list_tables()` /
`schema(tables)` for inspection. Per [Known Issues](../community/known-issues.md), fetching
*any* builtin tool through `ToolRegistry.get(...)` triggers
`auto_register_builtins()`, which unconditionally imports every builtin tool module,
including `SQLDatabaseTool`'s `langchain_community` imports. Importing `SQLDatabaseTool`
directly (as above) sidesteps the registry entirely, so it works even if you never touch
`ToolRegistry`; but you still need `langchain_community` installed, since the module-level
`from langchain_community.utilities.sql_database import SQLDatabase` import runs the moment
you import `agenttune.agentic.tools.builtin.sql` at all. The calculator in the main Quick
Start has no such dependency.

## Swapping in other algorithms

All five wrappers share the pattern: `model` is always required, dataset loading has the
same two paths (`train_dataset=<Dataset>` pre-loaded, or `dataset_name="..."` through
`DataManager`), and `output_dir`/`peft_config`/most `*Config` fields work identically. What
changes is required vs. optional args and how `reward_funcs` gets used. Full parameter
tables and theory live on each algorithm's own page; this is the delta.

!!! note "These reuse the Quick Start's 4-row toy dataset: set `gradient_accumulation_steps` for real data"
    The snippets below reuse `train_dataset` from the Quick Start above and, like it, don't
    set `gradient_accumulation_steps` (default `16`). See the
    [Config options table](#config-options-table) below. A small illustrative dataset
    needs it set low (e.g. `1`) to actually produce a step; a real dataset large enough to
    exceed `per_device_train_batch_size * gradient_accumulation_steps` won't hit this.

### DPO: `TrlAgenticDPO`

Three modes, auto-detected from what you pass (`use_rollouts` overrides the detection
explicitly):

| Mode | Trigger | Dataset needs | `reward_funcs` |
|---|---|---|---|
| 1. Standard offline | default, dataset already has `chosen`/`rejected` | `prompt`, `chosen`, `rejected` | not required; ignored if passed |
| 2. Reward-ranked rollouts | `use_rollouts=True`, no tools | `prompt` only | required; ranks generations into chosen/rejected |
| 3. Agentic tool rollouts | `tools=[...]` passed | `prompt` only | required |

```python
trainer = create_agentic_trainer(
    algorithm="dpo",
    model="Qwen/Qwen3-0.6B",
    tools=[calculator],
    reward_funcs=correct_answer_reward,   # single callable, DPO does not combine_rewards
    train_dataset=train_dataset,          # "prompt"-only dataset triggers rollout mode
    num_generations=4,                    # must be >= 2, generates then picks best/worst
    beta=0.1,
    output_dir="./runs/dpo-calculator",
)
trainer.train()
```

Fresh rollouts get generated every training step via a patched
`DPOTrainer.training_step`; the reward's best/worst pair per prompt becomes
chosen/rejected. See [DPO](../algorithms/dpo.md).

### PPO: `TrlAgenticPPO`

Needs a **policy model and something that scores it**: either a real `reward_model`
`nn.Module`, or a plain `reward_funcs` callable that gets auto-wrapped into one
(`RewardFnWrapper`). Same for the value function (`value_model`, `value_fn`, or an
auto-built `ValueModelWrapper` off the policy backbone). `eval_dataset` is required by the
underlying `PPOTrainer`.

```python
trainer = create_agentic_trainer(
    algorithm="ppo",
    model="Qwen/Qwen3-0.6B",
    tools=[calculator],
    reward_funcs=correct_answer_reward,   # auto-wrapped into RewardFnWrapper
    train_dataset=train_dataset,          # needs "input_ids" or "prompt" (auto-tokenized)
    eval_dataset=train_dataset,           # PPOTrainer requires one
    output_dir="./runs/ppo-calculator",
    kl_coef=0.05,
    cliprange=0.2,
)
trainer.train()
```

PPO does **not** rewrite `train()`; it patches the module-level `batch_generation`
function and (optionally) `generate_completions` so the rest of TRL's PPO update math,
logging, and checkpointing run untouched. See [PPO](../algorithms/ppo.md).

### RLOO: `TrlAgenticRloo`

`reward_funcs` is forwarded **as-is** to TRL's own `RLOOTrainer`. RLOO is the one wrapper
here that does not build a `combine_rewards` composite for you.

```python
trainer = create_agentic_trainer(
    algorithm="rloo",
    model="Qwen/Qwen3-0.6B",
    tools=[calculator],
    reward_funcs=correct_answer_reward,   # → passed straight through to RLOOTrainer
    train_dataset=train_dataset,
    num_generations=4,
    beta=0.05,
    output_dir="./runs/rloo-calculator",
)
trainer.train()
```

If you omit `tools`/`rollout_engine`/`rollout_func`, this is exactly TRL's stock RLOO:
`_agentic_mode` is `False` and nothing is patched. See [RLOO](../algorithms/rloo.md).

### BCO: `TrlAgenticBCO`

Same three-mode shape as DPO, but the dataset columns are `prompt` / `completion` / `label`
(boolean desirable/undesirable) instead of `chosen`/`rejected`, and online mode regenerates
its dataset **every epoch** (not every step) via a dynamically-built `OnlineBCOTrainer`
subclass:

```python
def bco_reward(responses, prompts=None, **kwargs):
    """BCO/DPO rollout-mode rewards go through _wrap_reward_fn, not the GRPO contract;
    see 'The reward calling convention' below."""
    return [1.0 if "84" in r or "20" in r or "60" in r or "9" in r else 0.0 for r in responses]

trainer = create_agentic_trainer(
    algorithm="bco",
    model="Qwen/Qwen3-0.6B",
    ref_model="Qwen/Qwen3-0.6B",
    tools=[calculator],
    reward_funcs=bco_reward,
    train_dataset=train_dataset,     # "prompt"-only triggers online rollout mode
    score_threshold=0.5,             # reward cutoff for the desirable label
    num_generations=2,
    prompts_per_epoch=16,
    output_dir="./runs/bco-calculator",
    beta=0.1,
)
trainer.train()
```

BCO requires `processing_class` (tokenizer) to be resolvable. Pass `model` as a string and
it loads one for you; pass a live model object and you must supply `processing_class`
yourself. See [BCO](../algorithms/bco.md).

## PEFT / LoRA

All five wrappers resolve `peft_config` through the exact same helper
(`_resolve_peft_config`, duplicated verbatim in each of the five files):

```python
def _resolve_peft_config(raw):
    if raw is None:
        return None
    try:
        from peft import PeftConfig
        if isinstance(raw, PeftConfig):
            return raw
    except ImportError:
        raise ImportError("[AgentTune] peft is required. pip install peft")
    if isinstance(raw, dict):
        from peft import LoraConfig
        return LoraConfig(**raw)
    raise TypeError(f"[AgentTune] peft_config must be a dict or PeftConfig, got {type(raw)}")
```

So `peft_config` can be a **plain dict**, no `import peft` needed in your own code, and
its keys are whatever `LoraConfig(**raw)` accepts:

```python
trainer = create_agentic_trainer(
    algorithm="grpo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    tools=[calculator],
    reward_funcs=[correct_answer_reward, used_calculator_reward],
    reward_weights=[0.8, 0.2],
    train_dataset=train_dataset,
    output_dir="./runs/grpo-calculator-lora",
    peft_config={
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    },
)
```

This is also verified against a real, GPU-run script
(`examples/agentic_grpo_real.py`), which passes a `LoraConfig` object directly instead
of a dict; both forms work, since `_resolve_peft_config` checks `isinstance(raw,
PeftConfig)` first and only builds `LoraConfig(**raw)` if you handed it a plain dict:

```python
from peft import LoraConfig

grpo = TrlAgenticGrpo(
    model=MODEL, processing_class=tok, reward_funcs=contract_reward,
    train_dataset=ds, output_dir=OUT_DIR,
    peft_config=LoraConfig(
        r=16, lora_alpha=32, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
    num_generations=16, per_device_train_batch_size=16,
    num_train_epochs=30, learning_rate=1e-5, beta=0.02,
)
```

If `peft` isn't installed, passing a dict/`PeftConfig` raises immediately with
`"[AgentTune] peft is required. pip install peft"`; it's not a silent no-op.

## Rollout engines standalone

The tool-calling loop that GRPO/PPO/RLOO/DPO/BCO all drive under the hood is not
training-specific. `agenttune.agentic.rollout_engines.rollout_factory` exposes it directly.
This is useful for debugging a tool/reward pair, or generating agentic trajectories without
training anything.

`create_rollout_engine` builds the underlying generation backend:

```python
def create_rollout_engine(
    backend: str = "auto",       # "auto" | "vllm" | "transformers" | "api"
    model=None,
    tokenizer=None,
    model_path: Optional[str] = None,
    api_provider: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,                    # forwarded verbatim to the chosen engine class
) -> RolloutEngine: ...
```

`create_rollout_fn` wraps an engine (or builds one for you) into the callable that
`_execute_trajectory` actually drives the multi-turn loop with:

```python
def create_rollout_fn(
    rollout_engine: Optional[RolloutEngine] = None,
    rollout_backend: Optional[str] = None,   # eagerly builds an engine if no rollout_engine
    model=None, tokenizer=None, model_path=None,
    api_provider=None, api_model=None, api_key=None, api_base_url=None,
    engine_kwargs: Optional[Dict] = None,
    tools: Optional[List] = None,
    max_steps: int = 20,
    reward_fn: Optional[Callable] = None,
    system_prompt: Optional[str] = None,
    ...
) -> Callable: ...
```

Calling it outside of any trainer, against a local transformers model, and inspecting the
output keys directly (these are exactly the keys `_format_for_grpo` builds in
`rollout_factory.py`):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine, create_rollout_fn

tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

engine = create_rollout_engine(backend="transformers", model=model, tokenizer=tok)

rollout_fn = create_rollout_fn(
    rollout_engine=engine,
    tools=[calculator],
    max_steps=6,
    system_prompt="You are a calculator agent. Use the tool, then answer.",
)

batch = rollout_fn(["What is 12 * (3 + 4)?", "What is 15 - 3 * 2?"])

sorted(batch.keys())
# ['completion_ids', 'conversations', 'env_mask', 'logprobs', 'prompt_ids',
#  'queries', 'responses', 'retrieved_chunk_ids', 'rewards', 'tool_call_counts',
#  'tools_used', 'trajectories']

batch["responses"]        # list[str], final assistant text per rollout
batch["trajectories"]     # list[Trajectory], full step-by-step record, incl. tool calls
batch["tool_call_counts"] # list[int], tool calls made per rollout
```

Every entry is one full trajectory per input prompt. `rewards` will be all-zero unless you
also pass `reward_fn=` (in which case `create_rollout_fn` scores each trajectory's
`final_response` and writes it back onto `Trajectory.reward` before returning). Calling
`rollout_fn(prompts)` with no `trainer=` kwarg routes generation through the standalone
engine path in `_gen`, which calls `engine.generate(...)` directly: no trainer,
no gradient, just inference.

## Using an API-based rollout backend

`APIRolloutEngine` (`agenttune/agentic/rollout_engines/api_engine.py`) is a universal
LiteLLM-backed engine, one class speaks OpenAI, Anthropic, Groq, OpenRouter, Ollama,
Together, Azure, and anything else LiteLLM supports, keyed entirely off the model string
prefix:

```python
class APIRolloutEngine(RolloutEngine):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,     # ← the real kwarg name, NOT api_base_url
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        provider: Optional[str] = None,     # kept for backwards-compat, ignored
        extra_litellm_kwargs: Optional[Dict] = None,
        **kwargs,
    ): ...
```

`base_url` is for things like local Ollama (`"http://localhost:11434"`) or a LiteLLM proxy;
it is **not** called `api_base_url` anywhere on this class.

### Building the engine directly

```python
import os
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine

engine = create_rollout_engine(
    backend="api",
    api_model="groq/llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],   # or omit, falls back to GROQ_API_KEY env var
    base_url=None,                        # only needed for a proxy / non-default endpoint
)
```

`create_rollout_engine(backend="api", ...)` forwards every extra `**kwargs` straight into
`APIRolloutEngine(model=api_model or "gpt-4o-mini", api_key=api_key, **kwargs)`, so
`base_url=` (and `max_retries=`, `extra_litellm_kwargs=`, etc.) all pass through by name
unmodified. There is no `api_base_url` parameter on `create_rollout_engine` itself; it only
recognizes `backend`, `model`, `tokenizer`, `model_path`, `api_provider`, `api_model`,
`api_key`, and `**kwargs`.

### Through `create_rollout_fn` (builds the engine for you)

`create_rollout_fn` is one level up and *does* expose a parameter literally named
`api_base_url`, but only so it can immediately forward it into `create_rollout_engine`
as `base_url=`:

```python
# rollout_factory.py: create_rollout_fn, eager engine build:
if rollout_engine is None and rollout_backend is not None:
    rollout_engine = create_rollout_engine(
        backend=rollout_backend, model=model, tokenizer=tokenizer,
        model_path=model_path, api_provider=api_provider, api_model=api_model,
        api_key=api_key, base_url=api_base_url, **(engine_kwargs or {}),
    )
```

So at the `create_rollout_fn` layer, the kwarg you pass is `api_base_url=`:

```python
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

rollout_fn = create_rollout_fn(
    rollout_backend="api",
    api_model="groq/llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    api_base_url=None,          # forwarded as base_url= to APIRolloutEngine
    tools=[calculator],
    max_steps=6,
    system_prompt="You are a calculator agent.",
)

batch = rollout_fn(["What is 12 * (3 + 4)?"])
print(batch["responses"][0])
```

The rule of thumb: **`create_rollout_engine`/`APIRolloutEngine` → `base_url=`.
`create_rollout_fn` → `api_base_url=`** (it just renames it on your behalf one layer down).
Writing `api_base_url=` directly into `create_rollout_engine(...)` or
`APIRolloutEngine(...)` does nothing useful. Neither accepts that name; it would land in
`**kwargs` and be silently ignored by `APIRolloutEngine.__init__`, since that constructor
never reads an `api_base_url` key.

Feeding this engine into training instead of a local model is the same `rollout_engine=`
kwarg every trainer accepts:

```python
trainer = create_agentic_trainer(
    algorithm="grpo",
    model="Qwen/Qwen2.5-1.5B-Instruct",   # the model being trained, still local
    tools=[calculator],
    reward_funcs=[correct_answer_reward, used_calculator_reward],
    rollout_backend="api",                 # generation for the tool loop goes via Groq
    api_model="groq/llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    train_dataset=train_dataset,
    output_dir="./runs/grpo-api-rollout",
)
```

Note the asymmetry this implies: `rollout_backend="api"` only changes how *rollouts* get
generated for the multi-turn tool loop; it does not change which model the trainer
computes gradients against. Mixing "generate with a hosted model, train a different local
model" is exactly what `rollout_engine=`/`rollout_backend=` is *for*. See
[GRPO](../algorithms/grpo.md) for the on-policy caveat that implies (importance-sampling
correction gets murkier the further the rollout policy drifts from the trained one).

## Low-level API: bypassing `create_agentic_trainer`

Every agentic wrapper's real hook is `rollout_func`; `tools=` is just a convenience that
builds one for you via `create_rollout_fn`. You can build and pass your own for full
control while still keeping every bit of `TrlAgenticGrpo`'s kwargs-routing, PEFT
resolution, and save/stats plumbing:

```python
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
from agenttune.core.backend_factory import create_agentic_trainer

my_rollout_func = create_rollout_fn(
    tools=[calculator],
    max_steps=6,
    reward_fn=correct_answer_reward,
    system_prompt="You are a calculator agent.",
)

trainer = create_agentic_trainer(
    algorithm="grpo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    rollout_func=my_rollout_func,     # used as-is, tools= building step is skipped
    reward_funcs=correct_answer_reward,
    train_dataset=train_dataset,
    output_dir="./runs/grpo-custom-rollout",
)
```

`setup_trainer()` checks for an explicit `rollout_func` **before** it tries to build one
from `tools`:

```python
rollout_func = trainer_kw.get("rollout_func") or _get(self.kwargs, "rollout_func")
if rollout_func is None and self._agentic_mode and trainer_kw.get("tools"):
    rollout_func = _build_grpo_rollout_fn(...)
```

so this is the supported "full control" path for anything `create_rollout_fn`'s own
`custom_rollout_fn=`, `pre_step_hook=`, `post_step_hook=`, or `on_trajectory_end=` params
don't already cover.

### The exact contract, for a truly bare `trl.GRPOTrainer`

If you want to go all the way down to raw `trl.GRPOTrainer` (no AgentTune wrapper at
all), the dict your `rollout_func` must return is enforced right here, inside
`agentic_grpo.py`'s own monkey-patch of `GRPOTrainer._generate_single_turn`:

```python
output = self_trainer.rollout_func(prompts, self_trainer)

required_keys = {"prompt_ids", "completion_ids", "logprobs"}
missing = required_keys - output.keys()
if missing:
    raise ValueError(
        f"[TrlAgenticGrpo] rollout_func must return keys {sorted(missing)} "
        "in its output dict."
    )

extra_fields = {k: v for k, v in output.items() if k not in required_keys}
return (output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields)
```

So the contract is: `prompt_ids: list[list[int]]`, `completion_ids: list[list[int]]`,
`logprobs: list[list[float]]` (plain floats; TRL does `torch.tensor(logps)` directly on
each sequence), plus anything else you want forwarded to your reward functions as extra
kwargs (`create_rollout_fn`'s own output already includes `trajectories`, `responses`,
`tool_call_counts`, `env_mask`, etc. for exactly this purpose).

!!! warning "This patch is process-wide, not per-instance"
    `GRPOTrainer._generate_single_turn = _patched_generate_single_turn` is a **class-level**
    assignment, installed the moment you construct any `TrlAgenticGrpo`/`create_agentic_trainer("grpo", ...)`
    with `tools`, `rollout_func`, or `environment_factory` set. It persists for every
    `GRPOTrainer` built afterward in the same process. The patch is arity-tolerant and
    falls through to the original method for instances that never set `rollout_func`, so
    mixing agentic and plain GRPO runs in one process is safe. But it means the "bare
    `trl.GRPOTrainer` respects my `rollout_func`" behavior described above is actually
    coming from AgentTune's patch, not from your installed TRL's own native support (which
    may or may not have grown a real `rollout_func` hook by the time you're reading this;
    check `inspect.signature(GRPOTrainer._generate_single_turn)` on your version if in
    doubt). Constructing one `TrlAgenticGrpo` anywhere in the process is enough to have the
    patch active for every subsequent raw `GRPOTrainer` you build by hand.

## The reward function calling convention

Straight from `TrlAgenticGrpo`'s class docstring, but see the warning under [Quick
Start](#2-write-reward-functions-agentic-mode) above, this doesn't match what's actually
passed:

```text
Standard mode  : f(completions: list[str], **kwargs) -> list[float]
Agentic mode   : f(completions: list[list[dict]], **kwargs) -> list[float]  # claimed, not what you get
    where each completion is a list of message dicts, e.g.:
    [{"role": "assistant", "tool_calls": [...]},
     {"role": "tool", "content": "..."},
     {"role": "assistant", "content": "*Yes*"}]
TRL handles the agentic loop; reward functions just inspect the result.
```

**Verified live: in agentic mode, `completions[i]` is actually a plain `str`**: the
flattened rollout text (tool-call tag, tool result, and final answer concatenated), the
same shape as standard mode. Write reward functions that treat every `completions[i]` as
text (`str(c)`, substring/regex checks) regardless of mode; that's the pattern used above
and it's what a real run confirms, whichever mode you're in.

Which mode you're in is decided by whether `tools`/`rollout_func`/`environment_factory` are
present, not by anything you configure explicitly. `**kwargs` always includes every extra
dataset column (like `answer` in the Quick Start) plus, in agentic mode, the extra fields
`create_rollout_fn` attaches: `tool_call_counts`, `retrieved_chunk_ids`, `trajectories`, etc.

This is the contract GRPO and RLOO both honor directly, because **GRPOTrainer and
RLOOTrainer call `reward_funcs` themselves** using TRL's own native contract. The other
three wrappers route reward callables through their own glue instead, with different
calling conventions:

| Algorithm | Who actually calls your reward function | Convention |
|---|---|---|
| **GRPO** | `GRPOTrainer` itself, after `combine_rewards([...], weights=...)` | `f(completions, **kwargs)`, `list[str]` in both standard *and* agentic mode (verified; the docstring's `list[list[dict]]` claim for agentic mode does not hold) |
| **RLOO** | `RLOOTrainer` itself, `reward_funcs` forwarded as-is, no `combine_rewards` | same as GRPO |
| **DPO** (rollout modes) | `create_dpo_rollout_fn` via `_wrap_reward_fn` | tries `f(responses, prompts=prompts)`, then `f(responses)` |
| **BCO** (rollout mode) | `OnlineBCOTrainer._generate_bco_dataset` via `create_rollout_fn`'s `reward_fn=` + `_wrap_reward_fn` | same as DPO |
| **PPO** | `RewardFnWrapper.forward` (wraps your callable into a fake `nn.Module`) | tries `f(prompts=prompts, completions=completions)`, then `f(prompts=prompts, responses=completions)`, then `f(completions)` |

`_wrap_reward_fn` (used by DPO and BCO's rollout modes) is deliberately tolerant; it
inspects your function's first parameter name/annotation and falls back through three
calling styles before giving up, so a reward written for one algorithm's rollout mode
usually works unmodified for another's. It is a *different* code path from the one GRPO and
RLOO use, though, so don't assume a GRPO-style `f(completions, **kwargs)` reward will see
the same kwargs when reused under DPO/BCO rollout mode; `prompts`/`responses` are
positional-first there, and dataset-column forwarding is not guaranteed the same way.

Only GRPO's wrapper runs your `reward_funcs` through `combine_rewards`/`reward_weights`.
see [Notes & gotchas](#notes-gotchas).

## Config options table

These are the most commonly touched kwargs across the five wrappers. Anything not covered
here just needs to exist on the installed `trl` version's matching `*Config`/`*Trainer`
`__init__` signature. See [Notes & gotchas](#notes-gotchas) for what happens when it
doesn't.

| Kwarg | Applies to | Default (GRPO) | What it does |
|---|---|---|---|
| `output_dir` | all 5 | `"./output/grpo_agentic"` (per-algo default dir) | Where the model, tokenizer, config YAML, and `training_stats.json` get saved |
| `beta` | GRPO, DPO, RLOO, BCO | `0.04` (GRPO), `0.1` (DPO/BCO), `0.05` (RLOO) | KL-penalty / regularization strength against the reference model |
| `num_generations` | GRPO, RLOO, DPO*, BCO* | `4` (GRPO/RLOO), `2` (DPO/BCO rollout modes) | Completions sampled per prompt per step (DPO/BCO: per epoch) |
| `max_completion_length` | GRPO, RLOO | `256` | Max tokens generated per completion |
| `gradient_accumulation_steps` | all 5 | `16` | Effective batch = `per_device_train_batch_size * gradient_accumulation_steps`. With `drop_last=True` on the train dataloader, a dataset smaller than that effective batch produces **zero** batches per epoch; training silently exits at step 0 with `train_loss: 0`. Set this explicitly (e.g. `1`) for a small toy/demo dataset like the Quick Start's. |
| `temperature` | GRPO, RLOO, PPO | `0.7` | Sampling temperature for rollout generation |
| `use_vllm` | GRPO (routed via `GRPOConfig`) | `False` | Enables vLLM-backed generation; combines with `vllm_mode` |
| `vllm_mode` | GRPO | `"colocate"` (read by the wrapper's own vLLM patch check) | `"colocate"` or `"server"`: where the vLLM engine runs |
| `peft_config` | all 5 | `None` | Dict or `PeftConfig` → resolved to a real `LoraConfig` via `_resolve_peft_config` |
| `report_to` | all 5 (routed via `*Config`) | `"none"` (BCO only sets this default explicitly) | Logging backend(s): `"trackio"`, `"wandb"`, `"none"`, or a list |
| `push_to_hub` | all 5 | `False` | Calls `trainer.push_to_hub()` after `save_model()` |
| `max_steps_per_turn` | GRPO, DPO, PPO, RLOO, BCO (agentic modes) | `20` | Tool-call loop depth **per rollout**, distinct from `*Config`'s own `max_steps` (total optimizer steps) |
| `system_prompt` | all 5 (data setup) | `None` | Prepended to every prompt via the chat template |
| `reward_weights` | **GRPO only** | uniform | Per-function weights into `combine_rewards` |
| `use_rollouts` | DPO, BCO | auto-inferred | Explicitly force/disable live-generation mode |

## CLI equivalent

`--dataset` is passed straight to `datasets.load_dataset()` with no config name, so pick a
dataset that doesn't require one (`openai/gsm8k` does; it needs `main` or `socratic`, and
the CLI has no `--dataset-config` flag to supply it). `correctness_reward` is SQL-specific
(it grades a `*yes*`/`*no*` answer), so pair a numeric-answer dataset with
`numerical_match_reward` instead:

```bash
agenttune train --algorithm grpo --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset microsoft/orca-math-word-problems-200k --reward-funcs numerical_match_reward \
    --output ./runs/grpo-cli
```

Note the CLI also has no `--max-steps`/epoch flag, so this runs the full dataset for the
trainer's default epoch count; expect a long run; use the Python API (`max_steps=...` above)
for a bounded quickstart.

Straight from `src/agenttune/cli/unified.py`:

```python
@app.command()
def train(
    algorithm: str = typer.Option(..., help="Agentic RL algorithm: grpo | dpo | ppo | rloo | bco."),
    model: str = typer.Option(..., help="Base model name or path."),
    dataset: str = typer.Option(..., help="Training dataset (HF id or path)."),
    output: str = typer.Option("./agenttune-run", help="Output directory for the trained adapter."),
    reward_funcs: Optional[str] = typer.Option(
        None, "--reward-funcs",
        help="Comma-separated reward function names from REWARD_REGISTRY.",
    ),
) -> None:
    from agenttune.api import train_agentic
    kwargs = {"model": model, "dataset": dataset, "output_dir": output}
    if reward_funcs:
        # resolve each name against REWARD_REGISTRY, error out on an unknown name
        ...
    trainer = train_agentic(algorithm, **kwargs)
    trainer.train()
```

`agenttune.api.train_agentic(algorithm, **kwargs)` is a one-line pass-through to
`create_agentic_trainer(algo, **kwargs)`, after validating `algorithm` against the same
five-value tuple as `AgenticAlgorithm`.

!!! note "The --dataset flag now loads for real, and grpo/rloo are reachable via --reward-funcs"
    The CLI used to pass the dataset string as `train_dataset=` (the pre-loaded-object
    path), so it never went through `DataManager`. It now passes `dataset=` instead, one
    of the same aliases (`dataset_name`/`dataset`) every wrapper's `setup_data()` already
    accepted for Path 2, so a real HF dataset id now actually loads.

    `--reward-funcs correctness_reward,structure_reward` resolves each name against
    `REWARD_REGISTRY` (raising a clear error listing every valid name if one doesn't
    match) and passes the result as `reward_funcs=`, satisfying GRPO/RLOO's hard
    requirement for it. There's still no `--tools` flag; agentic tool-calling training
    needs real Python callables, which the CLI has no way to name/resolve, so that path is
    Python-API-only. DPO/BCO's standard offline mode (dataset already has
    `chosen`/`rejected` or `completion`/`label` columns) doesn't need `reward_funcs` at
    all and was already reachable once the dataset-loading fix landed.

## Notes & gotchas

Findings from reading the five wrapper files directly, in the same spirit as
[Known Issues](../community/known-issues.md):

- **`reward_weights`/`combine_rewards` is GRPO-only.** Only `TrlAgenticGrpo.setup_trainer()`
  calls `combine_rewards(raw_reward_funcs, weights=reward_weights)`. RLOO forwards
  `reward_funcs` straight to `RLOOTrainer` untouched (no weighting, no string-name lookup).
  DPO and BCO's rollout modes take `reward_list = raw_reward if isinstance(raw_reward,
  list) else [raw_reward]` and then use only `reward_list[0]`; passing a list of reward
  functions to DPO/BCO silently drops everything past the first entry. PPO's `reward_funcs`
  gets index-0'd the same way before being wrapped into `RewardFnWrapper`.

- **Unrecognised kwargs are dropped silently, not rejected.** `_split_kwargs` (one version
  per wrapper) buckets each kwarg into "goes to `*Config`" or "goes to `*Trainer`" purely by
  checking membership in `inspect.signature(...).parameters` for your *installed* `trl`
  version; anything matching neither is a no-op (`# else: data / meta param,
  intentionally ignored`, verbatim comment in the source). A typo'd parameter name, or one
  that existed in a different `trl` release than the one you have, fails silently instead of
  raising. `TrlAgenticBCO` is the one wrapper with an explicit safety net here; it drops
  and logs a warning for any `config_kw` entry the installed `BCOConfig` doesn't accept;
  the other four don't warn at all.

- **The GRPO monkey-patch is process-wide.** See the warning under
  [Low-level API](#the-exact-contract-for-a-truly-bare-trlgrpotrainer) above.
  `GRPOTrainer._generate_single_turn` gets replaced at the class level the first time any
  agentic `TrlAgenticGrpo` is constructed, for the rest of the process.

- **`base_url` vs `api_base_url`.** `APIRolloutEngine.__init__` and `create_rollout_engine`
  both use `base_url`. Only `create_rollout_fn` exposes `api_base_url`, purely as a
  same-layer rename that gets forwarded to `create_rollout_engine(..., base_url=api_base_url)`
  internally. There is no `api_base_url` anywhere on `APIRolloutEngine` or
  `create_rollout_engine` themselves; passing it there lands in an unused `**kwargs` slot.

- **`tools=` fetched through `ToolRegistry` can crash on an unrelated missing dependency.**
  `ToolRegistry.get(...)` triggers `auto_register_builtins()`, which imports every builtin
  tool module up front, including the `langchain_community`-dependent ones. Importing a
  tool class directly (`from agenttune.agentic.tools.builtin.sql import SQLDatabaseTool`)
  avoids the registry's blanket import, but `sql.py` itself still needs
  `langchain_community` at module import time regardless of which action you call.

- **`create_agentic_trainer` is TRL-only by design.** `BackendType` has exactly one member
  (`TRL`), and it isn't consulted for branching; there's no `backend=` flag.

- **`GRPOConfig`/`DPOConfig`/`BCOConfig` field names can drift with your `trl` version.**
  Because every wrapper introspects the *installed* signatures rather than hard-coding a
  parameter list, upgrading `trl` can silently change which of your kwargs land on the
  `Config` vs. get dropped. If a kwarg you expect to take effect doesn't seem to, check
  `inspect.signature(trl.GRPOConfig.__init__)` (or the matching class) against your
  installed version before assuming the wrapper is broken.

- **PPO uses `trl.experimental.ppo`, BCO uses `trl.experimental.bco`.** Both import from
  TRL's experimental namespace, not the stable top-level `trl.PPOTrainer`/`trl.BCOTrainer`
  path. Expect these two to be the most version-sensitive of the five.

## See it run for real

See the [Local Notebooks](../notebooks/local-notebook.md) index for the exact GRPO run
this page's Quick Start pattern is based on (a real `SmolLM2-360M-Instruct` LoRA
fine-tune, on a real GPU, with before/after sampling proving the policy moved), and the
equivalent real runs for the other three dispatchable algorithms.

- [`examples/agentic_grpo_real.py`](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/agentic_grpo_real.py): the
  same script, runnable directly with `python examples/agentic_grpo_real.py`.
- [Algorithms Overview](../algorithms/overview.md): theory and full parameter tables per
  algorithm.
- [Known Issues](../community/known-issues.md): what's broken or orphaned elsewhere in
  training/eval, read the same way this page was written: from the source, not the docs.
