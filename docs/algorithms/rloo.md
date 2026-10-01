# RLOO (REINFORCE Leave-One-Out)

RLOO is a critic-free on-policy RL algorithm, structurally very close to GRPO: it samples
several completions per prompt, scores them with a reward function, and turns those scores
into a training signal without training a separate value network. `agenttune` wires it up as
`TrlAgenticRloo` (`backends/trl/agentic/rloo/agentic_rloo.py`), a thin kwargs-driven wrapper
around `trl.RLOOTrainer`/`trl.RLOOConfig`, dispatchable through `create_agentic_trainer("rloo",
...)`. It supports the same two modes as GRPO: standard reward-function RL over a prompt
dataset, or agentic multi-turn tool-calling rollouts.

## How it differs from GRPO

Both algorithms sample a group of `num_generations` completions per prompt and skip the critic
network entirely. The difference is the baseline used to turn raw rewards into an advantage:
GRPO normalizes each completion's reward against the **mean and standard deviation of its whole
group** (a z-score). RLOO instead uses the mean reward of the **other completions in the group,
excluding the one being scored**: a leave-one-out baseline, with no division by a standard
deviation. It's closer to vanilla REINFORCE with a classic variance-reduction trick than to
GRPO's normalized-advantage objective. In practice this makes `TrlAgenticRloo` and
`TrlAgenticGrpo` nearly interchangeable at the wrapper level: same kwargs-splitting logic, same
auto-detection of agentic mode from `tools`/`rollout_engine`/`rollout_func`, so switching between
the two is mostly a matter of changing the algorithm string.

## Quick Start

```python
from agenttune.core.backend_factory import create_agentic_trainer

def length_reward(prompts, completions, **kwargs):
    """Reward longer (but not excessively long) completions, capped at 1.0."""
    return [min(len(c.split()), 50) / 50 for c in completions]

trainer = create_agentic_trainer(
    "rloo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=length_reward,
    train_dataset=my_prompt_dataset,      # needs a "prompt" column
    output_dir="./runs/rloo",
    num_generations=4,
    max_completion_length=256,
    learning_rate=1e-6,
    beta=0.05,
)

results = trainer.train()
```

`model` and `reward_funcs` are the only two required kwargs; `TrlAgenticRloo.setup_trainer()`
raises `ValueError` immediately if either is missing. If `train_dataset` is omitted, AgentTune's
`DataManager` loads `dataset_name` instead (defaults to `trl-lib/tldr`).

### Agentic (tool-calling) mode

Passing `tools`, `rollout_engine`, or a raw `rollout_func` switches the wrapper into agentic
mode. It builds a rollout function via `create_rollout_fn` and patches the underlying
`RLOOTrainer._generate_single_turn` to route generation through it (version-agnostically, across
both the old `(prompt_ids, images, multimodal_fields)` and new `(prompts)` TRL call signatures):

```python
trainer = create_agentic_trainer(
    "rloo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=my_tool_use_reward,
    tools=[calculator, web_search],
    train_dataset=my_prompt_dataset,
    max_steps_per_turn=20,
    output_dir="./runs/rloo_agentic",
)
```

## Configuration Parameters

| Parameter | Alias(es) | Default | Description |
|---|---|---|---|
| `model` | — | required | HF model id or a loaded model |
| `reward_funcs` | — | required | `callable(prompts, completions, **kwargs) -> list[float]`, or a list of callables |
| `train_dataset` | — | — | HF `Dataset` with a `prompt` column |
| `num_generations` | — | `4` | completions sampled per prompt; the group RLOO's leave-one-out baseline is computed over |
| `max_completion_length` | `max_new_tokens` | `256` | max tokens generated per completion |
| `temperature` | — | `0.7` | rollout sampling temperature |
| `top_p` | — | `0.95` | nucleus sampling cutoff |
| `beta` | `kl_coef` | `0.05` | KL penalty coefficient against the reference policy |
| `learning_rate` | `lr` | `1e-6` | optimizer learning rate |
| `per_device_train_batch_size` | `batch_size` | `1` | per-device batch size |
| `gradient_accumulation_steps` | — | `16` | gradient accumulation steps |
| `num_train_epochs` | `epochs` | `1` | training epochs |
| `output_dir` | — | `./output/rloo_agentic` | checkpoint/log directory |
| `tools` | — | `None` | tool callables; auto-enables agentic multi-turn rollouts |
| `rollout_engine` / `rollout_func` | — | `None` | custom rollout engine, or a raw `rollout_func(prompts, trainer=...)` |
| `peft_config` | — | `None` | dict (built into a `LoraConfig`) or a `PeftConfig` instance |

Every default above comes straight from `TrlAgenticRloo.setup_trainer()`'s `config_defaults`
dict; anything you don't pass falls back to these values before `RLOOConfig` is constructed.

## Dataset Format

Standard mode expects a `prompt` column only; completions are generated on-policy during
training, the same as GRPO. No `chosen`/`rejected` or reference-answer columns are required
unless your reward function needs one (in which case it's forwarded from the dataset row).

## See Also

- [BCO Algorithm](bco.md): unpaired desirable/undesirable preference learning
- [Algorithms Overview](overview.md)
- [RL Training Guide](../user-guide/rl-training.md)
- [Local Notebooks](../notebooks/local-notebook.md): notebook 21 runs a real TRL `RLOOTrainer` on GPU
