# Features

AgentTune is two things that fit together:

1. **An RL training layer built on TRL** that trains LLM agents
   which *actually use tools*: GRPO, PPO, DPO, RLOO, and BCO through one entry point,
   `create_agentic_trainer(...)`.
2. **The agentic spine** on top of it: a single normalized trajectory format (`EventLog`)
   that carries one agent artifact through its entire lifecycle: **build → collect →
   evaluate → train → distill → heal.**

Everywhere else you can *run* an agent design. AgentTune lets you **train, evaluate, distill,
and self-heal** that design, through one shared data schema, not five disconnected tools.

Every row below is demonstrated in one of the [Local Notebooks](notebooks/local-notebook.md),
run for real: real models, real data, real output, executed fresh.

## Core capabilities

| Feature | What it does | See it run |
|---|---|---|
| **Agent design layer** | 5 swappable agent strategies (ReAct, Plan-and-Solve, Reflexion, MemoryReAct, Tree-of-Thoughts) behind one interface. | Spine lifecycle, Strategies |
| **Memory layer** | 3 pluggable memory backends: recent-k in-context, semantic vector recall, and relational/temporal graph memory. | Memory & strategies, Vector memory |
| **Tool library** | Built-in tools for SQL databases, web search, file I/O, code execution, GitHub, Slack, and browser automation (Playwright). | Tool library showcase |
| **Sandboxed execution** | Run an agent's tool calls inside an isolated `OpenEnv` sandbox instead of directly in-process. | OpenEnv sandbox |
| **Trajectory schema (EventLog)** | One event-log format that carries an agent's full run through every later stage (evaluation, training, distillation, healing) without re-instrumenting. | Spine lifecycle |
| **RL trainer** | Train tool-using agents with GRPO, PPO, DPO, RLOO, or BCO through one `create_agentic_trainer(...)` interface. | GRPO, PPO, RLOO, BCO |
| **Agentic distillation** | Copy a strong teacher agent's behavior into a small, cheaper student model via SFT on its captured trajectories; one call via `create_distill_trainer(...)`, with a built-in LoRA+TRL factory. | Agentic distillation |
| **Reward-model distillation** | Train a compact reward model (TRL `RewardTrainer`) that scores completions inside a training loop, cheaper than an LLM judge. | RL training & rewards |
| **RAG agent training** | Train a tool-using search agent end-to-end with one call, `create_rag_trainer(...)`: wires a retrieval backend, a search tool, and a search/correctness reward into the RL trainer (bring your own reward/tool to override the defaults). | Agentic RAG training |
| **Docs-to-training-data generator** | Turn a document corpus into a difficulty-tagged, grounded multi-hop QA training set. | Docs-to-training-data, Full synthesis pipeline |
| **Groundedness scoring** | Checks whether an answer's claims are actually supported by retrieved passages, via an LLM-judge rubric. | Reward defenses + groundedness |
| **Reward-hacking defenses** | Score clamping on over-retrieval, judge prompt rotation, and relative (batch) scoring, so a policy can't game one fixed prompt or a single absolute score. | Reward defenses + groundedness |
| **Trajectory evaluation** | Automatic, rule-based scoring of *how* an agent performed (tool-argument correctness, errors, loops), plus optional LLM-as-judge scoring. | Spine lifecycle |
| **Self-healing loop** | Detect a failing/looping agent, classify why, generate a corrective training example, retrain (real DPO), and gate the redeployment on real accuracy. See [Concepts: DECIDE & the closed loop](concepts/decide-and-closed-loop.md) for how the two stages fit together. | Self-healing & closed loop, Live-LLM failure detection & example generation |
| **Data collection** | Log real agent rollouts in a structured, replayable format for later training or analysis. | Spine lifecycle |
| **DECIDE** | Define a decision workflow as a YAML config file instead of code. | DECIDE business rules, Concepts: DECIDE & the closed loop |
| **Multi-agent orchestration** | Connect multiple agents/scoring steps into one pipeline via LangGraph. | LangGraph orchestration |
| **Standardized benchmarking** | Run an agent through EleutherAI's real `lm-eval` CLI (e.g. `arc_challenge`) instead of a bespoke eval script. | lm-eval integration |
| **SDK, CLI, and web service** | Three ways to use AgentTune directly: Python SDK, command-line tool, and a FastAPI web service. | API & CLI usage |
| **Industry examples** | Sample agents for BFSI, healthcare, legal, retail, and general support use cases. | BFSI, Other industries |

See [Local Notebooks](notebooks/local-notebook.md) for the full list, with descriptions and links.

## Possible use cases

| Use case | What it does |
|---|---|
| Train a tool-using agent's policy | Train an agent on a custom task and reward function using GRPO, PPO, DPO, RLOO, or BCO. |
| Train a document-search agent (RAG) | Train an agent that searches a document corpus and answers questions grounded in what it retrieves. |
| Compress a large agent into a small one (distillation) | Distill a strong teacher agent's behavior into a small, cheaper-to-run student model. |
| Automate KYC / account-opening checks | An agent pulls applicant data, runs screening checks, and auto-clears low-risk cases. |
| Automate contract review | An agent flags contract clauses that deviate from a standard playbook for human review. |
| Benchmark an agent against standard tasks | Run `arc_challenge`, `winogrande`, and other `lm-eval` tasks against your model/agent for comparable numbers. |

## Honest notes

- **Tool library**: `GitHubTool`, `SlackTool`, and `PlaywrightTool` exist in the builtin tool
  library but need credentials (GitHub/Slack tokens) or packages (`playwright` + browser
  binaries) not exercised in the linked notebook.
- **Reward-hacking defenses**: prompt rotation is a real random draw each call; small samples
  can land on the same rotation id by chance, that's expected variance, not a bug.
- **Hybrid PRM**: a real, reproducible gap exists: the default reward path currently scores a
  well-formed, a malformed, and a missing tool call identically. Documented, not hidden, in its
  notebook; see [Local Notebooks](notebooks/local-notebook.md).
- **Multi-agent orchestration** and the **RAG synthesis pipeline** each needed a real bug fixed
  (a `transformers`-version incompatibility, and a stale `TypedDict` vs. attribute-access
  mismatch respectively) before they would run for real; both fixed upstream; see each
  notebook's own notes for detail.
