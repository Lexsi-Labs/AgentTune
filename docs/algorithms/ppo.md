# PPO (Proximal Policy Optimization)

PPO is the classic actor-critic RLHF algorithm, and the odd one out among
AgentTune's agentic trainers: it needs **both** a reward signal (model or
function) **and** a separate value function estimating the baseline used for
the advantage calculation; GRPO, DPO, RLOO, and BCO all get by without a
value model. `TrlAgenticPPO` (`backends/trl/agentic/ppo/agentic_ppo.py`)
wraps `trl.experimental.ppo.PPOTrainer`. TRL moved PPO (and BCO) under
`trl.experimental` in recent releases; the old top-level `trl.PPOTrainer`
import path is gone.

## Quick start: plain callables, no `nn.Module` required

The distinctive feature here: you don't need to hand-build a reward model or
value model. `RewardFnWrapper` and `ValueFnWrapper` wrap plain Python
callables into the `nn.Module` shape `PPOTrainer.get_reward()` expects
internally (a fake `.logits` tensor of shape `(B, T, 1)` with the score
placed at the context-length position).

```python
from agenttune.core.backend_factory import create_agentic_trainer

def my_reward_fn(prompts, completions):
    """list[str], list[str] -> list[float]"""
    return [1.0 if "SENTIMENT=" in c else 0.0 for c in completions]

def my_value_fn(prompts, completions):
    """Baseline estimate — can be as simple as a heuristic."""
    return [len(c) / 100.0 for c in completions]

trainer = create_agentic_trainer(
    "ppo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    train_dataset=prompt_dataset,   # "prompt" column, auto-tokenized, or pre-tokenized "input_ids"
    eval_dataset=eval_dataset,      # required by PPOTrainer
    reward_mode="fn", reward_funcs=my_reward_fn,   # -> wrapped in RewardFnWrapper
    value_mode="fn",  value_fn=my_value_fn,        # -> wrapped in ValueFnWrapper
    output_dir="./runs/ppo",
    total_episodes=1000,
)
results = trainer.train()
```

`reward_mode`/`value_mode` default to `"auto"`: if you just pass
`reward_funcs=my_reward_fn` with no `reward_mode`, it's auto-wrapped in
`RewardFnWrapper` the same way. Likewise, omitting `value_fn`/`value_model`
entirely falls back to `ValueModelWrapper`, a linear head added on top of
the policy's own backbone (zero-initialized), so PPO always has *some* value
estimator even if you never provide one.

| `reward_mode` | Behavior |
|---|---|
| `"auto"` (default) | Use `reward_model` if given, else wrap `reward_funcs` |
| `"model"` | Require an `nn.Module` `reward_model`; errors if absent |
| `"fn"` | Require a callable `reward_funcs`; wraps it in `RewardFnWrapper` |

| `value_mode` | Behavior |
|---|---|
| `"auto"` (default) | Use `value_model` if given, else `ValueModelWrapper` |
| `"model"` | Require an `nn.Module` `value_model`; errors if absent |
| `"wrapper"` | Always build `ValueModelWrapper` from the policy backbone |
| `"fn"` | Require a callable `value_fn`; wraps it in `ValueFnWrapper` |

## Agentic mode

Passing `tools=` (or `rollout_engine=`) builds a rollout function via
`_build_ppo_rollout_fn` and applies two targeted instance-level patches
rather than rewriting `train()`:

1. **`_patch_batch_generation`**: swaps the module-level
   `trl.experimental.ppo.ppo_trainer.batch_generation` for a closure that
   calls your rollout function and converts its output into the
   `(query_responses, logitss)` tensors `train()` expects (a fake one-hot
   `logitss` so `selective_log_softmax` recovers the rollout's own
   log-probs). A `__del__` hook restores the original when the trainer
   instance is garbage-collected, so the patch never leaks to other
   `PPOTrainer` instances.
2. **`_patch_generate_completions`**: only applied when
   `num_sample_generations > 0`; routes eval-time sample generation through
   the same rollout function.

Everything else (the PPO update math, logging, checkpointing) runs exactly
as TRL wrote it.

## Configuration parameters

Config class: `trl.experimental.ppo.PPOConfig`, auto-routed via
introspection. Wrapper defaults:

| Parameter | Default | What it does |
|---|---|---|
| `total_episodes` | `1000` | Total rollout episodes across training |
| `response_length` | `256` | Max generated tokens per completion |
| `learning_rate` | `1e-6` | Policy optimizer LR |
| `kl_coef` | `0.05` | KL penalty coefficient against the reference policy |
| `cliprange` / `cliprange_value` | `0.2` / `0.2` | PPO clipping ranges for the policy ratio and value loss |
| `vf_coef` | `0.1` | Value loss weight in the combined objective |
| `gamma` / `lam` | `1.0` / `0.95` | Discount factor / GAE lambda |
| `num_ppo_epochs` | `4` | Optimization epochs per batch of rollouts |
| `num_mini_batches` | `1` | Mini-batches per PPO epoch |
| `local_rollout_forward_batch_size` | `4` | Batch size used during rollout generation |
| `output_dir` | `"./output/ppo_agentic"` | Checkpoints, `ppo_config.yaml`, `training_stats.json` |

`model` is required (string HF id or an already-loaded model); if no
`processing_class`/tokenizer is given and `model` is a string, the tokenizer
is auto-loaded from it. `train_dataset` needs either a pre-tokenized
`input_ids` column or a raw `prompt` column, which gets auto-tokenized using
`max_prompt_length` (default `128`).

## See also

- [Algorithms overview](overview.md)
- [GRPO](grpo.md), [DPO](dpo.md)
- [`examples/ppo_real.py`](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/ppo_real.py): trains a
  real reward model with `trl.RewardTrainer` first, then runs
  `trl.experimental.ppo.PPOTrainer` directly against it (not through
  `TrlAgenticPPO`), and reports honestly that PPO's reward signal is noisy
  at small scale.
- [RL training guide](../user-guide/rl-training.md)
