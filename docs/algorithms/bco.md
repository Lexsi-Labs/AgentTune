# BCO (Binary Classifier Optimization)

BCO learns a preference policy from **unpaired** desirable/undesirable labels on individual
completions, no `chosen`/`rejected` pair for the same prompt is needed, unlike DPO. `agenttune`
wires it up as `TrlAgenticBCO` (`backends/trl/agentic/bco/agentic_bco.py`), a kwargs-driven
wrapper around TRL's **experimental** `trl.experimental.bco.BCOTrainer`/`BCOConfig`. Note this
is an experimental TRL module, not the stable trainer surface, so its API is more likely to move
across TRL releases. It's dispatchable through `create_agentic_trainer("bco", ...)`.

## Three operating modes

`TrlAgenticBCO` actually supports three distinct data-sourcing modes, auto-detected from what you
pass:

1. **Standard offline BCO**: a complete dataset with `prompt`/`completion`/`label` columns.
   No live generation, no reward function required. This is the mode shown below.
2. **Online rollout BCO** (`use_rollouts=True`): dataset needs only a `prompt` column (or an
   explicit `prompt_pool`); completions are generated fresh every epoch and `reward_funcs`
   thresholds them into desirable/undesirable labels via `score_threshold`.
3. **Agentic tool-calling rollout BCO**: same as mode 2, but generation goes through
   multi-turn tool calls (pass `tools=[...]`).

Passing `reward_funcs` alone with a complete `prompt`/`completion`/`label` dataset does **not**
trigger rollouts; the wrapper checks whether the dataset already has all three columns and, if
`use_rollouts` wasn't set explicitly, disables live generation even if a reward function was
supplied. Only `tools`, `rollout_engine`, or an explicit `use_rollouts=True` force it on.

## Quick Start (Mode 1: offline, unpaired labels)

```python
from agenttune.core.backend_factory import create_agentic_trainer
from datasets import Dataset

data = Dataset.from_list([
    {"prompt": "Summarize the incident report.",
     "completion": "Server crashed at 02:14 UTC due to an OOM; restarted, no data loss.",
     "label": True},                                      # desirable
    {"prompt": "Summarize the incident report.",
     "completion": "so like, basically something happened with the server I think",
     "label": False},                                     # undesirable
    # ... more unpaired rows, each independently labeled
])

trainer = create_agentic_trainer(
    "bco",
    model="Qwen/Qwen2.5-1.5B-Instruct",     # a string id is auto-loaded with its tokenizer
    ref_model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=data,
    output_dir="./runs/bco",
    beta=0.1,
)

results = trainer.train()
```

`model` is the only strictly required kwarg (`ValueError` otherwise). If you pass it as a string,
the wrapper loads `AutoModelForCausalLM` and `AutoTokenizer` for you (and for `ref_model` too, if
that's also a string); you don't need to load them yourself first. If you pass a pre-loaded
model object instead, you must also supply `processing_class` (tokenizer) explicitly.

### Modes 2 and 3 in one line each

```python
# Mode 2 — online rollout, reward-thresholded, no tools:
create_agentic_trainer("bco", model=..., reward_funcs=my_reward,
                        train_dataset=prompt_only_ds, use_rollouts=True, score_threshold=0.5)

# Mode 3 — agentic tool-calling rollout:
create_agentic_trainer("bco", model=..., reward_funcs=my_reward,
                        tools=[calculator, web_search], train_dataset=prompt_only_ds)
```

In both, `reward_funcs` must be `callable(prompts, responses) -> list[float]`; rollout
completions scoring at or above `score_threshold` (default `0.5`) become desirable (`label=True`)
rows, the rest undesirable. If an epoch's batch comes out all-one-label, the wrapper force-balances
it by reward rank rather than crashing (BCO needs at least one example of each label).

## Configuration Parameters

| Parameter | Alias(es) | Default | Description |
|---|---|---|---|
| `model` | — | required | HF model id or loaded model |
| `ref_model` | — | `None` | reference model for the BCO loss |
| `train_dataset` | — | — | mode 1: `prompt`/`completion`/`label`; modes 2–3: `prompt` only |
| `beta` | — | `0.1` | BCO loss temperature |
| `max_length` | — | `512` | max combined prompt+completion tokens |
| `truncation_mode` | — | `"keep_end"` | truncation strategy when over `max_length` |
| `num_train_epochs` | `epochs` | `3` | training epochs |
| `per_device_train_batch_size` | `batch_size` | `2` | per-device batch size |
| `learning_rate` | `lr` | `1e-5` | optimizer learning rate |
| `num_generations` | — | `2` | (modes 2–3) rollouts sampled per prompt per epoch |
| `prompts_per_epoch` | — | `16` | (modes 2–3) prompts sampled per epoch |
| `score_threshold` | — | `0.5` | (modes 2–3) reward cutoff for the desirable label |
| `use_rollouts` | — | auto | force modes 2–3 on/off explicitly |
| `tools` | — | `None` | tool callables; auto-enables mode 3 |
| `embedding_func` / `embedding_tokenizer` | — | `None` | optional UDM density-ratio estimation (`prompt_sample_size`, `min_density_ratio`, `max_density_ratio` tune it further) |
| `peft_config` | — | `None` | dict (built into a `LoraConfig`) or a `PeftConfig` instance |

## Dataset Format

Mode 1 requires exactly `prompt`, `completion`, and `label` columns, where `label` is a plain
`bool`: `True` for a desirable completion, `False` for undesirable. Rows are **not** paired: a
given prompt can appear zero, one, or many times with either label, and there's no requirement
that every prompt have both a desirable and an undesirable example. This is the core distinction
from DPO, which requires a `chosen`/`rejected` pair sharing the same prompt.

## See Also

- [RLOO Algorithm](rloo.md): critic-free on-policy RL with a leave-one-out baseline
- [Algorithms Overview](overview.md)
- [RL Training Guide](../user-guide/rl-training.md)
- [Local Notebooks](../notebooks/local-notebook.md): notebook 17 runs a real `trl.experimental.bco.BCOTrainer` on GPU
