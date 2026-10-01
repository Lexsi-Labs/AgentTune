# Python API: Configuration Dataclasses

Config objects that are real, importable, and validated at construction time, but scattered
across the codebase with no single index until this page. All of them are plain
`@dataclass` definitions, no YAML, no global state.

!!! tip "Not the same thing as `config.yaml`"
    This page is about **Python dataclasses** you construct directly in code
    (`SFTConfig`, `EvalConfig`, the closed-loop configs). [Configuration](configuration.md)
    is about DECIDE's YAML file (`config.yaml`) that `GraphRunner` reads at load time. The
    two don't overlap: nothing on this page is parsed from YAML, and `config.yaml` has no
    field that constructs any of these classes directly.

## `agenttune.core.sft.config`: SFT training config

See [Python API: `agenttune.core`](core-api.md#coresft-supervised-fine-tuning-configuration)
for the narrative version of this section; this section is the full field-by-field
reference. Every class below constructs and validates cleanly, and
`SFTConfigLoader`/`SFTEvaluator` use it for real. For running training itself, see agentic
RL training: GRPO/DPO/PPO/RLOO/BCO via `create_agentic_trainer`
([Algorithms Overview](../algorithms/overview.md)).

!!! note "One accelerated-training toggle needs a separately-installed package"
    `ModelConfig` has one boolean field (default `False`) that opts into an alternate,
    faster training backend package. That package is not a declared AgentTune dependency
    (dropped from the public install surface); the field only takes effect if you've
    installed it yourself, otherwise the loader logs a warning and falls back to plain
    `transformers`. Not documented field-by-field here since it isn't part of the
    supported public surface.

### Enums

| Enum | Values |
|---|---|
| `TaskType` | `INSTRUCTION_FOLLOWING`, `SUPERVISED_FINE_TUNING`, `TEXT_CLASSIFICATION`, `TOKEN_CLASSIFICATION`, `TEXT_GENERATION`, `CHAT_COMPLETION` |
| `PrecisionType` | `BF16`, `FP16`, `FP32`, `AUTO` |
| `BackendType` | `SINGLE`, `DDP`, `FSDP`, `DEEPSPEED`; declared and exported, but no field on any dataclass below actually holds a `BackendType`; it's unused dead weight today |
| `TrainingType` | `SFT`, `RL`; same story: declared, exported, never referenced by any field |

### `ModelConfig`

| Field | Type | Default |
|---|---|---|
| `name_or_path` | `str` | required |
| `precision` | `PrecisionType` | `PrecisionType.BF16` |
| `quantization` | `Dict[str, Any]` | `{}` |
| `attn_implementation` | `str` | `"auto"` |
| `gradient_checkpointing` | `bool` | `True` |
| `max_memory` | `Optional[Dict[str, str]]` | `None` |
| `max_seq_length` | `int` | `2048` |
| `peft_enabled` | `bool` | `False` |
| `lora_rank` | `int` | `16` |
| `lora_alpha` | `int` | `32` |
| `lora_dropout` | `float` | `0.1` |
| `target_modules` | `Optional[List[str]]` | `None` |
| `bias` | `str` | `"none"` |
| `use_gradient_checkpointing` | `bool` | `True` |
| `num_labels` | `Optional[int]` | `None` (classification-only) |
| `model_init_kwargs` | `Dict[str, Any]` | `{}` |
| `device_map` | `Optional[Union[str, Dict]]` | `"auto"` |
| `trust_remote_code` | `bool` | `True` |

`__post_init__` raises `ValueError` if `name_or_path` is empty, and coerces a string
`precision` into `PrecisionType` (raises `ValueError` on anything else invalid).

### `DatasetConfig`

| Field | Type | Default |
|---|---|---|
| `name` | `str` | required |
| `split` | `str` | `"train"` |
| `subset` | `Optional[str]` | `None` |
| `config` | `Optional[str]` | `None`; alias for `subset`; `__post_init__` mirrors whichever one is set onto the other |
| `percent` | `Optional[float]` | `None` |
| `max_samples` | `Optional[int]` | `None` |
| `column_mapping` | `Dict[str, str]` | `{}` |
| `task_type` | `TaskType` | `TaskType.SUPERVISED_FINE_TUNING` |
| `system_prompt` | `Optional[str]` | `None` |
| `auto_detect_fields` | `bool` | `False` |
| `format_type` | `Optional[str]` | `None` |
| `text_column` | `str` | `"text"` |
| `instruction_column` | `str` | `"instruction"` |
| `response_column` | `str` | `"response"` |
| `output_column` | `str` | `"output"` |
| `input_column` | `str` | `"input"` |
| `context_column` | `str` | `"context"` |
| `label_column` | `str` | `"label"` |
| `tokens_column` | `str` | `"tokens"` |
| `tags_column` | `str` | `"ner_tags"` |
| `messages_column` | `str` | `"messages"` |
| `dataset_text_field` | `str` | `"text"` |
| `chat_template` | `Optional[str]` | `None` |
| `dataset_num_proc` | `Optional[int]` | `None` |
| `pad_token` | `Optional[str]` | `None` |
| `preserve_columns` | `Optional[List[str]]` | `None` |
| `processing_fn` | `Optional[Callable]` | `None` |
| `processing_batched` | `bool` | `False` |
| `processing_fn_kwargs` | `Dict[str, Any]` | `{}` |

`__post_init__` validation: `name` required; `percent` must be in `(0, 100]`; `max_samples`
must be positive if set; `task_type` coerced from `str`. `_set_task_defaults()` runs after;
in the current code it only ensures `column_mapping` exists per task type (`{}` if unset); it
does not actually populate task-specific column names into `column_mapping`.

### `TrainingConfig`

| Field | Type | Default |
|---|---|---|
| `per_device_batch_size` | `int` | `1` |
| `gradient_accumulation_steps` | `int` | `1` |
| `max_steps` | `Optional[int]` | `None` |
| `epochs` | `Optional[int]` | `None`; `__post_init__` sets `3` if both `epochs` and `max_steps` are `None` |
| `learning_rate` | `float` | `1e-5` |
| `weight_decay` | `float` | `0.01` |
| `warmup_steps` | `int` | `0` |
| `warmup_ratio` | `float` | `0.1` |
| `eval_interval` | `int` | `100` |
| `save_interval` | `int` | `500` |
| `max_grad_norm` | `float` | `1.0` |
| `fp16` / `bf16` | `bool` | `False` / `False` |
| `dataloader_num_workers` | `int` | `0` |
| `remove_unused_columns` | `bool` | `False` |
| `optimizer` | `str` | `"adamw_torch"` |
| `lr_scheduler` | `str` | `"cosine"` |
| `group_by_length` | `bool` | `False` |
| `dataloader_drop_last` | `bool` | `False` |
| `eval_accumulation_steps` | `Optional[int]` | `None` |
| `label_smoothing_factor` | `float` | `0.0` |
| `early_stopping_patience` | `Optional[int]` | `None` |
| `early_stopping_threshold` | `float` | `0.0` |
| `load_best_model_at_end` | `bool` | `True` |
| `metric_for_best_model` | `str` | `"eval_loss"` |
| `greater_is_better` | `bool` | `False` |
| `use_trl` | `bool` | `False` |
| `dataset_num_proc` | `Optional[int]` | `None` |
| `dataset_kwargs` | `Dict[str, Any]` | `{}` |
| `packing` | `bool` | `False` |
| `packing_strategy` | `str` | `"bfd"`; must be `"bfd"` or `"wrapped"` |
| `eval_packing` | `Optional[bool]` | `None` |
| `padding_free` | `bool` | `False` |
| `pad_to_multiple_of` | `Optional[int]` | `None` |
| `completion_only_loss` | `Optional[bool]` | `None` |
| `assistant_only_loss` | `bool` | `False` |
| `loss_type` | `str` | `"nll"`; must be `"nll"` or `"dft"` |
| `activation_offloading` | `bool` | `False` |
| `use_flash_attention_2` | `Optional[bool]` | `None` |
| `gradient_checkpointing` | `bool` | `False` |
| `gradient_checkpointing_kwargs` | `Dict[str, Any]` | `{}` |
| `extra_params` | `Dict[str, Any]` | `{}` |
| `seed` | `int` | `42` |
| `data_seed` | `Optional[int]` | `None` |

`__post_init__` raises `ValueError` on non-positive `per_device_batch_size`,
`gradient_accumulation_steps`, or `learning_rate`; on non-positive `dataset_num_proc` /
`pad_to_multiple_of` if set; and on `packing_strategy`/`loss_type` outside their allowed sets.

### `EvaluationConfig`

| Field | Type | Default |
|---|---|---|
| `compute_perplexity` | `bool` | `True` |
| `compute_rouge` | `bool` | `True` |
| `compute_bleu` | `bool` | `True` |
| `compute_meteor` | `bool` | `False` (needs `nltk`) |
| `compute_bertscore` | `bool` | `False` (needs `bert-score`) |
| `compute_semantic_similarity` | `bool` | `False` (needs `sentence-transformers`) |
| `compute_codebleu` | `bool` | `False` (needs `codebleu`) |
| `custom_metrics` | `Optional[List[Callable[[str, str, str], Dict[str, float]]]]` | `None` |
| `max_samples_for_quality_metrics` | `int` | `50`; `__post_init__` raises `ValueError` if `< 1` |
| `bertscore_model` | `str` | `"microsoft/deberta-xlarge-mnli"` |
| `semantic_similarity_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` |

### `LoggingConfig`

| Field | Type | Default |
|---|---|---|
| `output_dir` | `str` | `"./output"` |
| `run_name` | `Optional[str]` | `None` |
| `loggers` | `List[str]` | `["tensorboard"]` |
| `log_level` | `str` | `"INFO"`; `__post_init__` raises `ValueError` if not one of `DEBUG/INFO/WARNING/ERROR/CRITICAL` |
| `log_interval` | `int` | `10` |
| `save_strategy` | `str` | `"steps"` |
| `eval_strategy` | `str` | `"steps"` |
| `report_to` | `str` | `"none"` |

### `SFTConfig`

```python
@dataclass
class SFTConfig:
    model: ModelConfig
    dataset: DatasetConfig
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
```

`__post_init__` re-checks `model.name_or_path`/`dataset.name` are non-empty, then calls
`_apply_task_specific_settings()`: for `TEXT_CLASSIFICATION` it defaults `num_labels` to `2`
if unset; for `TOKEN_CLASSIFICATION`, `num_labels` defaults to `9`; for
`INSTRUCTION_FOLLOWING`/`CHAT_COMPLETION` it logs an `INFO` suggestion to raise
`max_seq_length` past 1024; for `TEXT_GENERATION` it logs an `INFO` suggestion about the
accelerated-training toggle mentioned above. All of these are warnings/logs, not hard
failures.

Methods:

| Method | Behavior |
|---|---|
| `to_dict() -> Dict[str, Any]` | Flattens all fields (including nested `ModelConfig`/`DatasetConfig`/etc.) into a plain dict; `Enum` fields become their `.value` |
| `from_dict(config_dict) -> SFTConfig` (classmethod) | Inverse of `to_dict()`; only rebuilds `model`/`dataset`/`train`/`logging`. **Does not** restore `evaluation` from the dict, so a round-tripped config always gets a fresh default `EvaluationConfig()` |
| `get_task_type() -> TaskType` | Returns `dataset.task_type` |
| `is_classification_task() -> bool` | `True` for `TEXT_CLASSIFICATION`/`TOKEN_CLASSIFICATION` |
| `is_generation_task() -> bool` | `True` for `INSTRUCTION_FOLLOWING`/`SUPERVISED_FINE_TUNING`/`TEXT_GENERATION`/`CHAT_COMPLETION` |

Two more methods exist for recommending the accelerated-training backend mentioned in the
note above; not detailed here since that backend isn't part of the supported public
surface.

```python
from agenttune.core.sft.config import SFTConfig, ModelConfig, DatasetConfig, TaskType

cfg = SFTConfig(
    model=ModelConfig(name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
    dataset=DatasetConfig(name="tatsu-lab/alpaca", task_type=TaskType.INSTRUCTION_FOLLOWING),
)
cfg.is_classification_task()    # False
cfg.to_dict()["model"]["precision"]  # "bf16"
```

### The 6 factory helpers

Each returns a fully-populated `SFTConfig` for a specific task with sane defaults, reading
overrides out of `**kwargs`. Signature shape is identical across all six:
`(model_name, dataset_name, [num_labels,] output_dir=..., **kwargs) -> SFTConfig`:

| Factory | Extra positional arg | Notable defaults |
|---|---|---|
| `create_instruction_following_config` | — | `max_seq_length=1024`, `peft_enabled=True`, 4-bit quant, `epochs=3`, `lr=2e-4` |
| `create_chat_completion_config` | — | `max_seq_length=2048`, `epochs=2`, `batch_size=2`, `grad_accum=4` |
| `create_text_classification_config` | `num_labels: int` | `max_seq_length=512`, `epochs=3`, `lr=5e-5` |
| `create_token_classification_config` | `num_labels: int` | Same shape as classification, `tags_column="ner_tags"` |
| `create_text_generation_config` | — | `epochs=3`, `lr=2e-4` |
| `create_supervised_finetuning_config` | — | `auto_detect_fields=True`, `epochs=3`, `lr=2e-4` |

```python
from agenttune.core.sft.config import create_text_classification_config

cfg = create_text_classification_config(
    model_name="distilbert-base-uncased",
    dataset_name="imdb",
    num_labels=2,
    max_samples=5000,
)
```

## `agenttune.eval.core.EvalConfig`

Config for the older, non-agentic `EvalRunner`/`EvalRegistry` evaluation path (distinct from
[agentic eval](../user-guide/evaluation.md)).

| Field | Type | Default |
|---|---|---|
| `eval_type` | `EvalType` | required (`TRAINING` \| `STANDALONE` \| `BENCHMARK` \| `CUSTOM`) |
| `task_categories` | `List[TaskCategory]` | required |
| `metrics` | `List[str]` | `["accuracy", "f1", "bleu", "rouge"]` |
| `batch_size` | `int` | `32` |
| `max_samples` | `Optional[int]` | `None` |
| `device` | `str` | `"auto"` |
| `precision` | `str` | `"bf16"` |
| `use_cache` | `bool` | `True` |
| `cache_dir` | `Optional[str]` | `None` |
| `output_dir` | `str` | `"./eval_results"` |
| `save_predictions` | `bool` | `True` |
| `save_metrics` | `bool` | `True` |
| `verbose` | `bool` | `False` |

No `__post_init__`; no validation runs on construction, any value is accepted as-is.

```python
from agenttune.eval.core import EvalConfig, EvalType, TaskCategory

cfg = EvalConfig(
    eval_type=EvalType.STANDALONE,
    task_categories=[TaskCategory.TEXT_CLASSIFICATION],
    batch_size=16,
)
```

## Closed-loop configs: real and load-bearing

Unlike `SFTConfig` above, these four configs are consumed end-to-end by working code today:
`PathAConfig`/`GateConfig` drive `FullClosedLoop`, and `TriggerConfig` drives
`RetrainingTrigger`, both exercised in
`tests/test_w4_full_loop.py`. `RetrainConfig` drives the adapter-only
retrain path (`retrain_config.run_retrain`), which itself calls the real (working) agentic
`create_agentic_trainer` factory. See
[Python API: Self-Healing Closed Loop](closed-loop.md) for the surrounding architecture;
this section is just the field reference.

### `PathAConfig` (`decide/closed_loop/full_loop.py`)

Settings for the failure-detection signal path (detect → classify → generate).

| Field | Type | Default |
|---|---|---|
| `audit_log_path` | `str` | required |
| `classifier_model` | `str` | `"gpt-4o-mini"` |
| `generator_model` | `str` | `"groq/llama-3.3-70b-versatile"` |
| `api_base` | `Optional[str]` | `None`; point either model at a local OpenAI-compatible server |
| `judge_threshold` | `float` | `0.6` |
| `max_revisits` | `int` | `3` |
| `validation_script` | `str` | `"python -c 'import sys; sys.exit(0)'"` (a no-op) |
| `batch_size` | `int` | `10` |

### `GateConfig` (`decide/closed_loop/full_loop.py`)

Settings for the deployment decision and applying it.

| Field | Type | Default |
|---|---|---|
| `config_path` | `str` | `"config.yaml"` |
| `backend` | `str` | `"transformers"` (`transformers` \| `vllm` \| `api`) |
| `task_regression_tol` | `float` | `0.0` |
| `trajectory_regression_tol` | `float` | `0.05` |
| `min_test_samples` | `int` | `1` |

### `TriggerConfig` (`decide/closed_loop/retraining_trigger.py`)

Thresholds for `RetrainingTrigger`'s six trigger conditions plus its two safety gates.

| Field | Type | Default | Trigger/gate |
|---|---|---|---|
| `total_failures_threshold` | `int` | `50` | T1: buffer size |
| `dominance_ratio` | `float` | `0.70` | T2: one root_cause ≥ this fraction |
| `min_examples_ready` | `int` | `30` | T3: accepted-example count |
| `reward_drift_window` | `int` | `50` | T4: rolling window size |
| `reward_drift_baseline_size` | `int` | `50` | T4: episodes to lock baseline |
| `reward_drift_sigma` | `float` | `1.5` | T4: std devs below baseline to fire |
| `novel_type_min_count` | `int` | `3` | T5: instances of a new type before firing |
| `staleness_window_hours` | `float` | `6.0` | T6: hours since last retrain |
| `min_stale_buffer_size` | `int` | `10` | T6: minimum buffer size to consider staleness |
| `max_drop_rate` | `float` | `0.60` | Gate: skip retrain if buffer drop rate exceeds this |
| `min_attempts_for_drop_gate` | `int` | `10` | Gate: attempts needed before the drop-rate gate applies |
| `max_buffer_size` | `int` | `500` | FIFO eviction cap on `TrainingBuffer` |

```python
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig

trigger = RetrainingTrigger(TriggerConfig(total_failures_threshold=20, min_examples_ready=10))
fire, reason = trigger.should_trigger()
```

### `RetrainConfig` (`decide/closed_loop/retrain_config.py`)

Adapter-only (LoRA) retrain settings, consumed by `build_retrain_config`/`build_retrainer`/
`run_retrain`.

| Field | Type | Default |
|---|---|---|
| `model` | `str` | required |
| `algorithm` | `str` | `"dpo"` (`"dpo"` \| `"bco"`; GRPO retrain is deferred) |
| `output_dir` | `str` | `"./output/retrain"` |
| `lora_r` | `int` | `8` |
| `lora_alpha` | `int` | `16` |
| `lora_dropout` | `float` | `0.05` |
| `lora_target_modules` | `Optional[List[str]]` | `None` (peft picks a default per architecture) |
| `num_train_epochs` | `int` | `1` |
| `per_device_train_batch_size` | `int` | `1` |
| `gradient_accumulation_steps` | `int` | `4` |
| `learning_rate` | `float` | `1e-5` |
| `max_steps` | `Optional[int]` | `None` |
| `beta` | `float` | `0.1` |
| `seed` | `int` | `42` |
| `extra` | `Dict[str, Any]` | `{}`; passthrough kwargs to the trainer |

`peft_config_dict()` turns `lora_*` into the dict the agentic trainers' `_resolve_peft_config`
expects (`r`, `lora_alpha`, `lora_dropout`, `task_type: "CAUSAL_LM"`, and `target_modules` if
set).

```python
from agenttune.decide.closed_loop.retrain_config import RetrainConfig, run_retrain

cfg = RetrainConfig(model="Qwen/Qwen2.5-1.5B-Instruct", algorithm="dpo", max_steps=20)
# `examples` is a List[TrainingExample] drained from a RetrainingTrigger
stats = run_retrain(examples, cfg)
```
