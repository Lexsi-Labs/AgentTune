# Changelog

All notable changes to AgentTune will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-28

### Added

- Cohere tool calling: the rollout parses Command R7B
  (`<|START_ACTION|>[{"tool_name", "parameters"}]<|END_ACTION|>`, also with the markers stripped)
  and Command-R / Aya Expanse (`Action: ```json [...]````) calls, and the plain
  `[{"tool_name", "parameters"}]` list the fallback prompt asks for.
- `tools_fallback_prompt` and `tool_result_format` on `create_rollout_fn` and
  `create_agentic_trainer("grpo", ...)`: for templates that ignore `tools` (every Cohere
  hackathon model: Tiny Aya, Aya Expanse, Aya Vision, North) the schemas go into the system
  prompt and tool results come back as user turns, with a warning. `None` raises instead.
- `agenttune[vllm]` extra; `require_vllm()` gives an install hint where vLLM is needed.
- GRPO and RLOO runs with tools write `<output_dir>/trajectories.jsonl`, one
  `Trajectory.to_dict()` record per rollout, with tool calls as
  `{"id", "name", "arguments": dict}` for every model family. AuditKIT's
  `episodes_from_agenttune` reads it directly. `trajectories_file=` renames it, `None` turns it off.
- `Trajectory.to_dict()` and `trajectory_writer(path)` (an `on_trajectory_end` hook).
- `lexsi_provenance.json` (`lexsi.provenance/1`) in every model/run directory saved by the TRL
  GRPO, RLOO, DPO, BCO and PPO trainers, embedding the training dataset folder's own provenance
  when it has one (`agenttune.utils.provenance`).

### Changed

- `transformers>=5.15,<6` (resolves to 5.17.0); base install uses `trl==1.7.1` and no longer
  pulls vLLM; `torchvision` added for `cohere_compass`.
- `agenttune.__version__` is read from the installed package metadata.

### Fixed

- Tool-using rollouts executed 0 tool calls on every Cohere model, and crashed on turn 2 with
  Aya Expanse and Aya Vision.
- Tool-call arguments were double-encoded as a JSON string on templates that apply `tojson`.
- The package did not install on macOS (vLLM in the base requirements).
- A reply that is a JSON list of non-objects (e.g. `[2, 3]`) raised `TypeError` in the
  tool-call parser and aborted the GRPO step.
- `final_response` (and the assistant turn sent back to the model) kept special tokens such as
  `<|START_RESPONSE|>` and `<EOS_TOKEN>`.
- A template that renders a `tool` turn empty (Tiny Aya, North) dropped the tool result
  silently; the result is now folded into a user turn, with a warning.

## [1.0.0] - 2026-09-01

First public release. The `agenttune.agentic` public API is frozen under semver —
exports, signatures, `EventKind` / `MemoryKind` values, `EventLog` projection schemas,
and documented return-dict shapes will not change without a major version bump.

### RL training

- `create_agentic_trainer(algorithm=...)` — one entry point for `grpo`, `ppo`, `dpo`,
  `rloo`, and `bco`, each running real tool-calling rollouts. Built on TRL.
- Rollout engines: `transformers` (local), `vllm` (batched), `api` (any LiteLLM provider).
- `REWARD_REGISTRY` — 26 named reward functions with `combine_rewards(...)`; `LLMJudge`
  with hosted-API, local `transformers`, and local `vllm` backends.
- Optional, unofficial Unsloth backend (`backend="unsloth"` / `"auto"`), installed
  separately.

### Agentic spine

- Two-tier `EventLog` trajectory schema — `light` (observational) and `full`
  (trainable: token spans + logprobs).
- Agent strategies: ReAct, Plan-and-Solve, Reflexion, MemoryReAct, Tree-of-Thoughts.
- Memory: InContext, Vector, Graph, plus `TrajectoryStore`.
- Harness: `DictToolHarness` (pure-Python) and `OpenEnvHarness` (sandboxed OpenEnv).
- `Project` lifecycle: `infer` / `collect` / `evaluate` / `evaluate_agentic` / `train` /
  `distill` / `heal`.
- Agentic distillation — behavior-clone a teacher's trajectories into a small student.

### DECIDE

- YAML decision-workflow engine (`llm_call`, `llm_judge`, `router`, `rules`, `parallel`,
  `human_review`, `tool_call`, `output`) compiled to a `langgraph` `StateGraph`, with
  audit logging and step-limit enforcement.
- BFSI and generic templates; `FileWriter` / `PostgresWriter` / `WebhookSender`
  destinations.
- Self-healing closed loop: detect → classify → generate → retrain → gate → redeploy.

### Agentic RAG

- Retrieval backends (`SQLiteFTSBackend`, `ChromaBackend`), a `search_corpus` tool, and
  GRPO training against a retrieval reward.
- A 6-stage docs-to-training-data synthesis pipeline.

### Multi-agent orchestration

- `AgentTuneGraph` (LangGraph) composes rollout nodes and LLM judges into one
  GRPO-compatible reward / rollout function.

### Packaging

- Python 3.12+. `pip install -e .` covers the spine, RL training, RAG, extra eval
  metrics, the PostgreSQL destination, and the OpenEnv sandbox. Extras: `[service]`,
  `[docs]`, `[dev]`, `[flash-attn]`.
- Source-available under the Lexsi Labs Source Available License (LSAL) v1.1.

---

Development prior to 1.0.0 was internal and is not tracked here.

[1.0.0]: https://github.com/Lexsi-Labs/AgentTune/releases/tag/v1.0.0
