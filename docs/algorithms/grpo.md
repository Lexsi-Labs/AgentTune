# GRPO (Group Relative Policy Optimization)

GRPO is reward-function-driven, not preference-pair-driven: it samples
several completions per prompt (a "group"), scores every one with your
reward function(s), normalizes the reward within that group, and updates the
policy on the resulting relative advantage. No reward model, no value
model; just a scorer. `TrlAgenticGrpo` (`backends/trl/agentic/grpo/agentic_grpo.py`)
wraps `trl.GRPOTrainer` and adds a real agentic mode on top: passing
`tools=` drives a multi-turn tool-calling rollout instead of single-shot
generation.

## Quick start

```python
from agenttune.core.backend_factory import create_agentic_trainer
from agenttune.agentic.tools.builtin.finqa_tool import calculator

def contract_reward(completions, **kwargs):
    """Agentic mode: each completion is a list[dict] of chat messages."""
    rewards = []
    for comp in completions:
        used_tool = any(m.get("role") == "tool" for m in comp)
        final = next(
            (m["content"] for m in reversed(comp) if m.get("role") == "assistant"),
            "",
        )
        rewards.append(1.0 if used_tool and final.strip() else 0.0)
    return rewards

trainer = create_agentic_trainer(
    "grpo",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=contract_reward,
    tools=[calculator],                 # agentic tools — triggers the rollout path
    train_dataset=my_prompt_dataset,    # needs a "prompt" column (chat-formatted)
    output_dir="./runs/grpo",
    num_generations=4,
    max_completion_length=256,
    beta=0.04,
    max_steps=100,
)
results = trainer.train()
```

`reward_funcs` and `model` are the only required kwargs; `setup_trainer()`
raises `ValueError` if either is missing. `reward_funcs` accepts a single
callable, a list of callables, or registry names; multiple reward functions
are combined via `combine_rewards`, weighted by `reward_weights` if given.

## Standard vs. agentic mode

`TrlAgenticGrpo` detects agentic mode at construction time from
`tools`, `rollout_func`, or `environment_factory` being present. This changes
the shape reward functions receive:

| Mode | Reward function signature |
|---|---|
| Standard | `f(completions: list[str], **kwargs) -> list[float]` |
| Agentic | `f(completions: list[list[dict]], **kwargs) -> list[float]`, e.g. `[{"role": "assistant", "tool_calls": [...]}, {"role": "tool", "content": "..."}, {"role": "assistant", "content": "..."}]` |

The installed TRL can't natively schema a `BaseTool` object or run tool
calls on transformers<5, so when `tools=` is set (and no `rollout_func` is
supplied), `TrlAgenticGrpo` builds its own tool-calling `rollout_func` via
`create_rollout_fn` and monkey-patches `GRPOTrainer._generate_single_turn` at
the class level to route generation through it; this also covers vLLM
colocate mode by syncing weights before each rollout call. Once a
`rollout_func` is active, `tools`/`rollout_func` are popped from the kwargs
passed to `GRPOTrainer` (native `tools=` never reaches it) and re-attached to
the trainer instance afterward. Standard (non-agentic) `GRPOTrainer`
instances built after an agentic one are unaffected; the patch forwards
extra positional args and falls through when `rollout_func` isn't set.

## Configuration parameters

Config class: `trl.GRPOConfig`, auto-routed via introspection against
whatever kwargs you pass. The wrapper sets its own defaults for these:

| Parameter | Default | What it does |
|---|---|---|
| `beta` | `0.04` | KL penalty coefficient against the reference policy |
| `num_generations` | `4` | Completions sampled per prompt; the "group" GRPO normalizes over |
| `max_completion_length` | `256` | Max tokens generated per completion |
| `temperature` / `top_p` | `0.7` / `0.95` | Sampling parameters for rollout generation |
| `epsilon` | TRL's default | Clipping range for the policy ratio |
| `use_vllm` | `False` | Generate through vLLM instead of `model.generate` |
| `vllm_mode` | `"colocate"` | vLLM deployment mode; with `use_vllm=True` and colocate mode, the wrapper patches `trainer.llm` onto the colocated vLLM engine |
| `output_dir` | `"./output/grpo_agentic"` | Checkpoints, `grpo_training_config.yaml`, `training_stats.json` |

Any other `GRPOConfig` field (`log_completions`, `push_to_hub`, `report_to`,
`chat_template_kwargs`, `trackio_space_id`, ...) is accepted and passed
through untouched. `GRPOTrainer`-level kwargs (`processing_class`,
`peft_config`, `callbacks`, `optimizers`) are routed automatically too.

Data-loading kwargs (`dataset_name`, `split`, `max_samples`,
`column_mapping`, `system_prompt`, `format_fn`) are consumed by
`setup_data()` and never forwarded to TRL; see `setup_data()`'s docstring
for the full list, or pass `train_dataset=`/`eval_dataset=` directly to skip
loading entirely.

## See also

- [Algorithms overview](overview.md)
- [DPO](dpo.md), [PPO](ppo.md)
- [Known issues](../community/known-issues.md)
