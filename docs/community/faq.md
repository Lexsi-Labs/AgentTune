# FAQ

## What is AgentTune, in one sentence?

An RL training layer built on TRL for LLM agents that use tools, plus an
agentic spine on top that carries one agent artifact through **build → collect → evaluate
→ train → distill → heal** via a single normalized trajectory format, `EventLog`. See the
[Home page](../README.md) and [Features](../features.md).

## Do I need a GPU to try it?

No. The agentic-spine core (`agenttune.agentic`) is pure Python; you can build a
strategy + harness, run episodes, and evaluate them with no model, no network, and no GPU.
See [Getting Started](../getting-started/installation.md). A GPU is
only needed once you train a real model with GRPO/PPO/DPO/RLOO/BCO.

## What's the difference between the agentic spine and DECIDE?

The spine (`agenttune.agentic`) is a Python API for building, running, and training an
agent design against the `EventLog` schema. DECIDE (`agenttune.decide`) is a separate YAML-
defined decision-workflow engine for running a pipeline in production and logging every
decision to an audit trail. The closed loop connects the two: it watches DECIDE's audit
log and retrains via the spine's real trainers. See
[Architecture](../reference/architecture.md) and
[Concepts: DECIDE & the closed loop](../concepts/decide-and-closed-loop.md).

## Which RL algorithms are fully wired into `create_agentic_trainer(...)`?

GRPO, PPO, DPO, RLOO, and BCO, through one entry point. See [Features](../features.md) and
the [Roadmap](roadmap.md).

## Does this need API keys for OpenAI/Anthropic/Groq?

No. AgentTune Decide runs open-source models only (Qwen, LLaMA, and similar); the
`api_keys` block in [`config.yaml`](../reference/configuration.md) is a deprecated
placeholder kept for backward compatibility, not a required credential.

## What Python version does it need?

Python 3.12+. See [Getting Started](../getting-started/installation.md) for the install command
(OpenEnv sandboxing is a base dependency) and the `[service]` optional extra.

## Where is the AgentTune repository?

This repo is the canonical AgentTune repository. See the
[root README](https://github.com/Lexsi-Labs/AgentTune) for the full project
overview, or clone this repo directly to run everything linked from these docs. All 30
notebooks under [Local Notebooks](../notebooks/local-notebook.md) were executed fresh in this
environment: real models, real data, zero error cells.

## Where do I report a bug or request a feature?

[GitHub Issues](https://github.com/Lexsi-Labs/AgentTune/issues) for the repo, or
[GitHub Discussions](https://github.com/Lexsi-Labs/AgentTune/discussions) for open-ended
questions. See [Contributing](contributing.md) for the full workflow, and
[Security](security.md) if what you found is a vulnerability rather than a bug.
