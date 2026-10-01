# Basic Concepts

The vocabulary you need before the User Guide starts throwing code at you.

## `EventLog`: the one trajectory format everything shares

Every agent run (a spine episode, an RL rollout, a DECIDE pipeline execution, an eval)
gets normalized into an `EventLog`: a list of typed `Event`s (`TEXT`, `REASONING`,
`TOOL_CALL`, `TOOL_RESULT`, `OBSERVATION`, `TURN_COMPLETE`, `REWARD`, `MEMORY_OP`). This is
the thing that lets one agent artifact flow through evaluation, training, distillation,
and self-healing without re-instrumenting anything for each stage. Two tiers exist:
"light" (observational, cheap) and "full" (carries token spans/logprobs, trainable).

## `Harness`: the environment an agent acts in

A `Harness` is a Gym-style `reset()`/`step(action)` interface. `DictToolHarness` is the
built-in one; it runs plain Python callables as tools, in-process, no sandbox. Swap in
`OpenEnvHarness` to run tool calls in an isolated remote environment instead.

## `AgentStrategy`: the agent's policy

A strategy decides what to do at each step: `init()` sets up state, `propose()` picks the
next action, `observe()` incorporates the result, `is_done()` says when to stop. AgentTune
ships 5: ReAct, Plan-and-Solve, Reflexion, MemoryReAct, Tree-of-Thoughts. All take an
injected `policy` callable, so a strategy is model-agnostic; you can drive it with a
scripted function (as in the [Quick Start](quickstart.md)) or a real model.

## `run_episode`: the driver that ties strategy + harness together

```python
run_episode(strategy, harness, task) -> EventLog
```

This is the actual loop: call the strategy's `propose()`, step the harness, feed the
result back via `observe()`, repeat until `is_done()`. Every strategy plugs into this same
function.

## `Project`: the lifecycle object

`Project(strategy=..., harness=...)` is the thing you actually instantiate. It wraps
`run_episode` and adds the rest of the lifecycle: `.infer()` (one episode), `.evaluate()`/
`.evaluate_agentic()` (score trajectories), `.collect()`/`.collect_rollout()` (gather many),
`.train()` (SFT or GRPO), `.distill()` (teacher→student), `.heal()` (detect failures and
feed the self-healing loop).

## The lifecycle, end to end

```
build (Strategy + Harness) → collect (rollouts) → evaluate → train → distill → heal
                                                                          │
                                                                          └──▶ retrain
```

This is the "agentic spine." It's one of two systems in AgentTune; the other is
**DECIDE**, a separate YAML-defined decision-workflow engine for running a pipeline in
production and logging every decision to an audit trail. They connect through the
**self-healing closed loop**, which watches that audit trail and retrains via the spine's
real trainers. See [Architecture](../reference/architecture.md) for the full picture, and
[User Guide: DECIDE Workflows](../user-guide/decide-workflows.md) /
[User Guide: Self-Healing](../user-guide/self-healing.md) for how to actually use that
side.

## Where to go next

- **[Quick Start](quickstart.md)** if you haven't run anything yet.
- **[User Guide: Agentic Spine](../user-guide/agentic-spine.md)** for the 5 strategies, 4
  memory backends, and tool library in full.
- **[User Guide: RL Training](../user-guide/rl-training.md)** to actually train a model.
