# Algorithms Overview

AgentTune's agentic RL training runs through a single entry point:
[`create_agentic_trainer(algorithm, **kwargs)`](../reference/core-api.md) in
`core/backend_factory.py`. The `AgenticAlgorithm` enum it dispatches on has
exactly five members (`grpo`, `dpo`, `ppo`, `rloo`, `bco`), each backed by a
`TrlAgentic*` wrapper class under `backends/trl/agentic/`. The TRL
backend is the one wired into the factory (see
[Known Issues](../community/known-issues.md)).

## Summary table

| Algorithm | Wired into `create_agentic_trainer`? | Needs preference pairs or a reward function? | Notebook |
|---|---|---|---|
| [GRPO](grpo.md) | Yes | Reward function | 15_agentic_grpo_real |
| [DPO](dpo.md) | Yes | Preference pairs (`chosen`/`rejected`); or a reward function in rollout mode | 22_self_heal_dpo_real |
| [PPO](ppo.md) | Yes | Reward function/model **and** a value function/model | 20_ppo_real |
| [RLOO](../algorithms/rloo.md) | Yes | Reward function | 21_rloo_real |
| [BCO](bco.md) | Yes | Binary desirable/undesirable labels (or a reward function thresholded into labels) | 17_bco_real |

See the [Local Notebooks](../notebooks/local-notebook.md) index for links to each notebook above.

## GRPO: Group Relative Policy Optimization

Samples several completions per prompt, scores every one with your reward
function(s), and normalizes the reward *within that group* before updating
the policy, no reward model, no value model, just a programmatic scorer.
Pick GRPO when you have (or can write) a reward function (a correctness
checker, a format checker, a retrieval-quality scorer) and want on-policy RL
without the overhead of PPO's actor-critic setup. `TrlAgenticGrpo` also
supports an agentic mode where `tools=` drives a real multi-turn tool-calling
rollout. See [GRPO](grpo.md).

## DPO: Direct Preference Optimization

Trains directly on `prompt`/`chosen`/`rejected` triples; no reward model, no
rollout loop, no RL machinery, just a classification-style loss over pairs.
Pick DPO when you already have (or can generate) ranked pairs of good/bad
completions. `TrlAgenticDPO` also has two live-rollout modes that regenerate
fresh pairs every training step from a reward function, with or without
tools. See [DPO](dpo.md).

## PPO: Proximal Policy Optimization

The classic actor-critic RLHF algorithm: a policy generates, a reward
model or function scores, and a *separate* value model estimates the
baseline used for the advantage calculation. It's the only one of these five
that strictly needs both a reward signal and a value estimator. Pick PPO when
you need a learned reward model, or want the flexibility of plugging in
arbitrary reward/value callables (`TrlAgenticPPO`'s `RewardFnWrapper` /
`ValueFnWrapper`), and can tolerate more compute and more instability than
GRPO. See [PPO](ppo.md).

## RLOO: REINFORCE Leave-One-Out

Reward-function driven, like GRPO: multiple completions per prompt, scored by
your reward function. The difference is the baseline: RLOO uses the mean
reward of the *other* samples for a given prompt (leave-one-out) instead of
GRPO's group normalization. Simpler than GRPO, no `beta`/KL term, no group
statistics to tune. Pick RLOO when you want GRPO's reward-function-driven
setup with a cheaper, more direct baseline. See
[RLOO](../algorithms/rloo.md).

## BCO: Binary Classifier Optimization

Doesn't need paired chosen/rejected data; it works off a
`prompt`/`completion`/`label` dataset where `label` is a desirable/undesirable
bool, or generates that label live by thresholding a reward function
(`score_threshold`) during rollouts. Uses TRL's `BCOTrainer` from
`trl.experimental.bco`. Pick BCO when your feedback signal is naturally
binary (thumbs up/down, pass/fail) rather than pairwise comparisons. See
[BCO](bco.md).

## Choosing between them

- **Have a reward function, no reward model?** GRPO or RLOO. GRPO if you want
  group-relative normalization and are fine tuning `beta`/`num_generations`;
  RLOO if you want the simpler leave-one-out baseline.
- **Have ranked pairs of completions?** DPO: no rollout loop required.
- **Have binary desirable/undesirable labels instead of pairs?** BCO.
- **Need a learned reward model or a value function, not just a scoring
  function?** PPO: the only algorithm here with a real actor-critic loop.

For the mechanics of wiring any of these into a full training run, see the
[RL training guide](../user-guide/rl-training.md) and the
[Core API reference](../reference/core-api.md).
