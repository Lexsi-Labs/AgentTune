# DPO (Direct Preference Optimization)

DPO is preference-pair-driven, not reward-function-driven: it trains
directly on `prompt`/`chosen`/`rejected` triples with a classification-style
loss, no reward model and (in its default mode) no rollout loop at all.
`TrlAgenticDPO` (`backends/trl/agentic/dpo/agentic_dpo.py`) wraps
`trl.DPOTrainer`, the standard trainer, deliberately **not**
`OnlineDPOTrainer`, which TRL still marks experimental.

## Quick start: offline mode (static pairs)

```python
from agenttune.core.backend_factory import create_agentic_trainer
from agenttune.agentic import build_dataset
from agenttune.decide.closed_loop.contracts import TrainingExample

examples = [
    TrainingExample(
        trajectory_id="ex-1",
        original_failure_type="format_violation",
        root_cause="answered in prose instead of the required contract",
        prompt=[{"role": "user", "content": "Classify: 'terrible, broke on day one'"}],
        chosen=[{"role": "assistant", "content": "SENTIMENT=negative"}],
        rejected=[{"role": "assistant", "content": "The sentiment of this review is negative."}],
    ),
    # ... more rows
]
rows = build_dataset(examples)   # -> list[dict] with prompt/chosen/rejected

from datasets import Dataset

trainer = create_agentic_trainer(
    "dpo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=Dataset.from_list(rows),
    beta=0.1,
    output_dir="./runs/dpo",
)
results = trainer.train()
```

This is **Mode 1**: `reward_funcs` is not required, and is silently ignored
if you pass one alongside a complete `prompt`/`chosen`/`rejected` dataset.

## Three operating modes

| Mode | Trigger | Dataset needs | `reward_funcs` |
|---|---|---|---|
| 1. Standard offline | Default | `prompt`, `chosen`, `rejected` | Ignored if passed |
| 2. Reward-ranked rollouts | `use_rollouts=True`, or `reward_funcs` alone (tentative) | `prompt` only | Required |
| 3. Agentic tool-calling rollouts | `tools=` or `rollout_engine=` present | `prompt` only | Required |

`use_rollouts` resolution, in priority order: an explicit `use_rollouts=`
kwarg always wins; otherwise `tools`/`rollout_engine` being present always
means rollouts; otherwise `reward_funcs` alone sets it *tentatively* to
`True`, and `setup_data()` flips it back to `False` if the dataset already
has real `chosen`/`rejected` columns (so passing `reward_funcs` alongside a
complete dataset is safe; it will **not** trigger rollouts). In rollout
mode, `chosen`/`rejected` columns are only injected as placeholders if
missing; real columns are never overwritten.

## Online rollout mode: a real, distinctive feature

In Mode 2/3, `TrlAgenticDPO` monkey-patches `DPOTrainer.training_step` at
the class level (`_patch_dpo_training_step`) so that **every training
step**, fresh `chosen`/`rejected` pairs are generated live instead of being
read from the dataset:

1. Decode the current batch's prompts back to text.
2. Call `self.rollout_func(prompts, trainer=self)`, built by
   `create_dpo_rollout_fn`, which drives `num_generations` (≥ 2) completions
   per prompt through your `reward_funcs` (the first one, if a list) to rank
   them into a chosen/rejected pair, optionally through multi-turn tool
   calls if `tools=` is set.
3. Re-pad and re-assemble `[prompt | chosen]` / `[prompt | rejected]`
   sequences and hand them to the trainer's normal `compute_loss`.

The patch is a no-op for any `DPOTrainer` instance without `rollout_func`
set, so standard offline runs are unaffected.

```python
trainer = create_agentic_trainer(
    "dpo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=my_reward_fn,          # ranks completions into chosen/rejected
    train_dataset=prompt_only_dataset,  # needs only a "prompt" column
    tools=[calculator],                 # optional — enables tool-calling rollouts
    num_generations=4,
    output_dir="./runs/dpo_online",
)
```

## Configuration parameters

Config class: `trl.DPOConfig`, auto-routed via introspection. Wrapper
defaults:

| Parameter | Default | What it does |
|---|---|---|
| `beta` | `0.1` | Preference loss temperature |
| `loss_type` | `"sigmoid"` | DPO loss variant |
| `max_length` | `1024` | Max total sequence length (prompt + completion) |
| `num_generations` | `2` | Completions per prompt in rollout mode (must be ≥ 2) |
| `max_new_tokens` | `256` | Generation length in rollout mode |
| `temperature` | `0.7` | Sampling temperature in rollout mode |
| `output_dir` | `"./output/dpo_agentic"` | Checkpoints, `dpo_config.yaml`, `training_stats.json` |

Agentic-only kwargs (`tools`, `rollout_engine`, `rollout_backend`,
`reward_funcs`, `max_steps_per_turn`, `engine_kwargs`, `num_generations`,
`use_rollouts`) are stripped before `DPOTrainer` is constructed; they never
leak into `DPOConfig`/`DPOTrainer`.

## See also

- [Algorithms overview](overview.md)
- [GRPO](grpo.md), [PPO](ppo.md)
- Self-heal DPO example: [`examples/self_heal_dpo_real.py`](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/self_heal_dpo_real.py)
  builds corrective preference rows through the real DECIDE spine
  (`build_dataset`/`TrainingExample`) and trains offline DPO on them.
- [RL training guide](../user-guide/rl-training.md)
