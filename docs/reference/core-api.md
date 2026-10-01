# Python API: `agenttune.core`

The internals behind agentic RL training: the algorithm factory, the kwargs→TRL routing
mechanism, and the callback system. This page is the exhaustive parameter reference. For
the narrative, ratings-based tour (what's real, what's orphaned, what's broken) see
[Algorithms Overview](../algorithms/overview.md); for a hands-on walkthrough
see [RL Training](../user-guide/rl-training.md). For the public, stable wrapper over this
module, see [Unified SDK](unified-sdk.md).

## `agenttune.core.backend_factory`

```python
from agenttune.core.backend_factory import create_agentic_trainer, list_agentic_backends
```

### `create_agentic_trainer(algorithm: str, **kwargs) -> Any`

```python
def create_agentic_trainer(algorithm: str, **kwargs) -> Any: ...
```

Looks up `algorithm` in the `AgenticAlgorithm` enum, resolves it against an internal
registry of trainer wrapper classes, and calls `trainer_class(**kwargs)`: **every kwarg
you pass is forwarded verbatim**, no config object required. Raises `RuntimeError` if TRL
isn't importable, `ValueError` if `algorithm` isn't one of the five supported values.

```python
trainer = create_agentic_trainer(
    algorithm     = "grpo",
    model         = "Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs  = my_reward_fn,
    tools         = [my_tool],
    train_dataset = my_dataset,
    output_dir    = "./runs/grpo",
    max_steps     = 100,
)
results = trainer.train()
```

### `AgenticAlgorithm`: the exact enum values

```python
class AgenticAlgorithm(Enum):
    GRPO = "grpo"
    DPO  = "dpo"
    PPO  = "ppo"
    RLOO = "rloo"
    BCO  = "bco"
```

`algorithm.lower()` is matched against these five values; nothing else is dispatchable.

### `BackendType`

```python
class BackendType(Enum):
    TRL = "trl"  # Only TRL supported for agentic
```

A single-member enum. It exists but isn't consulted by `create_agentic_trainer` for
branching; TRL is the only agentic backend wired into the factory (see
[Algorithms Overview](../algorithms/overview.md)).
Don't confuse this with `core.sft.config.BackendType`, an unrelated same-named enum
(`SINGLE`/`DDP`/`FSDP`/`DEEPSPEED`) for distributed SFT backend selection.

### The registry

| `AgenticAlgorithm` | Wrapper class | Underlying class | Module |
|---|---|---|---|
| `GRPO` | `_GRPOWrapper` | `TrlAgenticGrpo` | `backends.trl.agentic.grpo.agentic_grpo` |
| `DPO` | `_DPOWrapper` | `TrlAgenticDPO` | `backends.trl.agentic.dpo.agentic_dpo` |
| `PPO` | `_PPOWrapper` | `TrlAgenticPPO` | `backends.trl.agentic.ppo.agentic_ppo` |
| `RLOO` | `_RLOOWrapper` | `TrlAgenticRloo` | `backends.trl.agentic.rloo.agentic_rloo` |
| `BCO` | `_BCOWrapper` | `TrlAgenticBCO` | `backends.trl.agentic.bco.agentic_bco` |

Each wrapper's `__new__` simply returns `UnderlyingClass(**kwargs)`; the wrapper classes
add nothing beyond an `is_available()` classmethod used by `list_agentic_backends()`.
Population of `_REGISTRY` happens once, at import time, only if the five `TrlAgentic*`
imports all succeed (`TRL_AVAILABLE = True`); otherwise the registry stays empty and every
call to `create_agentic_trainer` raises.

### `list_agentic_backends() -> Dict[str, Any]`

Pure Python, no GPU or TRL runtime needed beyond the import having succeeded; prints a
status table and returns:

```python
{
    "grpo": {"available": True, "backend": "trl", "class": "_GRPOWrapper"},
    "dpo":  {"available": True, "backend": "trl", "class": "_DPOWrapper"},
    ...
}
```

Safe to call to check what's usable in your environment before constructing a trainer.

## Parameter routing: how `**kwargs` finds its way to TRL

None of the five wrapper classes hard-code a parameter list. Each one's `__init__` calls a
`_split_kwargs`-style helper (see `agentic_grpo.py`) that does this at call time:

```python
from trl import GRPOConfig, GRPOTrainer
config_keys  = set(inspect.signature(GRPOConfig.__init__).parameters)  - {"self"}
trainer_keys = set(inspect.signature(GRPOTrainer.__init__).parameters) - {"self"}
```

Every kwarg you pass is looked up against those two live signatures: first matched
against `GRPOTrainer`'s parameters, then `GRPOConfig`'s. Anything matching neither
(dataset-loading knobs, `system_prompt`, etc.) is silently dropped from the TRL construction
and consumed separately by AgentTune's own data-loading step. Because this introspects the
**installed** `trl` package rather than hard-coding a list, it tracks upstream TRL API
changes automatically; the tradeoff is that a typo'd kwarg name is dropped silently rather
than raising.

### GRPO's split (`TrlAgenticGrpo`): reproduced from its own docstring

| Group | Routed to | Example keys |
|---|---|---|
| **GRPOTrainer params** | `GRPOTrainer.__init__` (via introspection) | `model`, `reward_funcs`, `processing_class`, `peft_config`, `tools`, `rollout_func`, `environment_factory`, `callbacks`, `optimizers` |
| **GRPOConfig params** | `GRPOConfig.__init__` (via introspection) | `output_dir`, `beta`, `epsilon`, `num_generations`, `max_completion_length`, `temperature`, `top_p`, `use_vllm`, `vllm_mode`, `chat_template_kwargs`, `log_completions`, `push_to_hub`, `report_to`, `trackio_space_id` |
| **Data params** | Consumed by `setup_data`, never forwarded to TRL | `dataset_name`, `dataset_config`, `split`, `max_samples`, `column_mapping`, `system_prompt`, `format_fn` (signature `fn(example: dict) -> dict`, applied post-load), `format_batched`, `format_remove_columns`; or skip loading entirely by passing `train_dataset`/`eval_dataset` directly |

Passing `tools=[...]` (or `rollout_func=`/`environment_factory=`) at construction switches
`TrlAgenticGrpo` into **agentic mode**, logged at construction time. In agentic mode, tools
are driven through AgentTune's own rollout engine, see
[Rollout Engines](rollout-engines.md), rather than TRL's native `tools=` handling, which
the installed TRL only executes natively on `transformers>=5` and can't schema a
`BaseTool` object.

Reward function signature depends on mode:

| Mode | Signature |
|---|---|
| Standard | `f(completions: list[str], **kwargs) -> list[float]` |
| Agentic | `f(completions: list[list[dict]], **kwargs) -> list[float]`: each completion is a list of message dicts (`{"role": "assistant", "tool_calls": [...]}`, `{"role": "tool", "content": "..."}`, …) |

### The other four algorithms: same mechanism, different data-param surface

All four use the identical introspection-based split against their own TRL Config/Trainer
pair: `DPOConfig`/`DPOTrainer`, `trl.experimental.ppo`'s `PPOConfig`/`PPOTrainer`,
`RLOOConfig`/`RLOOTrainer`, and `trl.experimental.bco`'s `BCOConfig`/`BCOTrainer`. What
differs is the data-side surface each exposes:

- **DPO** (`TrlAgenticDPO`): three modes selected by what you pass, not by an explicit
  `mode=` kwarg: a complete `prompt`/`chosen`/`rejected` dataset runs standard offline DPO
  (`reward_funcs` is ignored if present); a `prompt`-only dataset plus `reward_funcs` and
  `use_rollouts=True` generates completions each step and ranks them into chosen/rejected;
  adding `tools=` on top of that routes generation through the multi-turn tool-calling
  rollout engine. `tools=` or `rollout_engine=` auto-triggers rollout mode even without
  `use_rollouts=True` explicit.
- **PPO** (`TrlAgenticPPO`): routes `PPOConfig`/`PPOTrainer` params the same
  introspection way (`output_dir`, `total_episodes`, `response_length`, `learning_rate`,
  `kl_coef`, `cliprange`, `vf_coef`, `gamma`, `lam`, `temperature`, `num_ppo_epochs`,
  `num_mini_batches`, `local_rollout_forward_batch_size`, …), but additionally patches the
  constructed **instance** (`_patch_batch_generation`, optionally
  `_patch_generate_completions`) so generation inside `train()` routes through the agentic
  rollout function; TRL's own PPO update math is left untouched. Requires `train_dataset`
  with an `"input_ids"` column and (per TRL) an `eval_dataset`; exactly one of
  `reward_funcs`/`reward_model` is required, selected/forced via `reward_mode`
  (`"auto"`/`"model"`/`"fn"`), with a parallel `value_mode`
  (`"auto"`/`"model"`/`"fn"`/`"wrapper"`) and `value_fn` for a fully-callable value function
  with no `nn.Module` needed. Agentic-only extras: `tools`, `rollout_engine`,
  `rollout_backend`, `max_steps_per_turn` (default `20`), `system_prompt`.
- **RLOO** (`TrlAgenticRloo`): same `_split_kwargs`-style split as GRPO against
  `RLOOConfig`/`RLOOTrainer`; its docstring is entirely about a TRL API-version compatibility
  fix for `_generate_single_turn`'s argument count across TRL releases, not a params table.
- **BCO** (`TrlAgenticBCO`): mirrors DPO's three-mode shape: complete `prompt`/`completion`/
  `label` dataset runs standard offline BCO; `prompt`-only + `reward_funcs` +
  `use_rollouts=True` generates online and thresholds scores into desirable/undesirable via
  `score_threshold`; adding `tools=` routes that generation through the tool-calling loop.

## `agenttune.core.callbacks`

```python
from agenttune.core.callbacks import TrainerCallback, CallbackHandler, TrainerControl
```

A HuggingFace-`Trainer`-shaped callback system, real and independently usable on its own;
build a `CallbackHandler` and fire hooks directly if you want this callback shape without
pulling in a full trainer.

### `TrainerControl`

```python
@dataclass
class TrainerControl:
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False
```

### `TrainerCallback`

Abstract base class (`ABC`, but no `@abstractmethod`s; every hook is a real no-op you can
selectively override). Every hook receives `(self, args, state, control, **kwargs)`:

| Hook | Called |
|---|---|
| `on_init_end` | End of trainer initialization |
| `on_train_begin` / `on_train_end` | Start / end of training |
| `on_epoch_begin` / `on_epoch_end` | Start / end of an epoch |
| `on_step_begin` / `on_step_end` | Start / end of a training step |
| `on_evaluate` | After evaluation (`kwargs` includes `metrics=`) |
| `on_save` | After a checkpoint save |
| `on_log` | After logging (`kwargs` includes `logs=`) |
| `on_prediction_step` | After a prediction step |

A callback returning a `TrainerControl` from a hook lets it override the current control
flow; returning `None` leaves it unchanged.

### `CallbackHandler`

```python
CallbackHandler(callbacks: List[TrainerCallback], model, tokenizer, optimizer=None, scheduler=None)
```

Manages a de-duplicated list of callback instances (`add_callback` instantiates a bare
class if you pass one, and skips, with a warning, a second instance of a class already
present) and fires each hook across all of them via `call_event(event, args, state,
control, **kwargs)`, injecting `model=`/`tokenizer=`/`optimizer=`/`scheduler=` into every
call. Exposes one convenience method per hook (`on_train_begin`, `on_step_end`, …) plus
`callback_list` (newline-joined class names) and `remove_callback`.

`TrainerState` also lives in this module but is an empty pass-through class, a
compatibility placeholder, not a real state container.

## `core.sft`: supervised fine-tuning configuration

`core/sft/config.py` defines `SFTConfig` (nesting `ModelConfig`, `DatasetConfig`,
`TrainingConfig`, `EvaluationConfig`, `LoggingConfig`) plus six task-specific factory
helpers (`create_instruction_following_config`, `create_chat_completion_config`,
`create_text_classification_config`, `create_token_classification_config`,
`create_text_generation_config`, `create_supervised_finetuning_config`) and the enums
`TaskType`, `PrecisionType`, `TrainingType`, and a second, unrelated `BackendType`
(`SINGLE`/`DDP`/`FSDP`/`DEEPSPEED`, distributed-training strategy, not the agentic one
above). All pure dataclasses with `__post_init__` validation, `to_dict()`/`from_dict()`;
zero heavy deps, genuinely usable to build and validate a config today.

For running training itself, use agentic RL training (`grpo`/`dpo`/`ppo`/`rloo`/`bco` via
`create_agentic_trainer`, above); see [User Guide: RL Training](../user-guide/rl-training.md).
