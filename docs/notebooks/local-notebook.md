# Local Notebooks

43 notebooks total, real models and real data throughout, zero error cells, no
`!python script.py` shell-outs. Every link below points at the notebook's file on GitHub
(open, read, or download from there; `git clone` the repo to run one locally).

- **`docs/notebooks/` (28)**: 1-9 are the core use-case walkthroughs, 10-28 cover the rest
  of the feature surface (every RL algorithm, sandboxed execution, the tool library, reward
  defenses, and more). See [Features](../features.md) for the full feature-to-notebook map.
- **`examples/USECASES/` (15)**: real end-to-end use-case notebooks: real small open-source
  models (≤3B params), real datasets, real training/inference, no API-based models anywhere.

Nine of these are also mirrored on Google Colab; see [Sample Notebooks](sample-notebook.md).

## docs/notebooks: Use Case Notebooks (1-9)

The core notebooks demonstrating the agentic spine end to end.

| # | Notebook | What it shows |
|---|---|---|
| 1 | [Agentic Spine: Core Lifecycle](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/01_agentic_spine_core_lifecycle.ipynb) | Build → collect rollouts → evaluate → distill → heal, all on one `Project` |
| 2 | [Agentic Distillation](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/02_agentic_distillation_real.ipynb) | Teacher → student distillation: a model learns a new output contract |
| 3 | [Self-Healing & Closed Loop](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/03_self_healing_and_closed_loop.ipynb) | Real TRL `DPOTrainer` retrain on real submitted failures, deployment gate decides on real accuracy (0.33→1.00), real adapter deployed and reloaded as a `PeftModel` |
| 4 | [Memory & Agent Strategies](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/04_memory_and_agent_strategies.ipynb) | Semantic (vector) + graph memory recall, ReAct/Reflexion strategies |
| 5 | [DECIDE Business Rules](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/05_decide_business_rules.ipynb) | YAML-defined decision pipeline producing real verdicts |
| 6 | [API & CLI Usage](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/06_api_and_cli_usage.ipynb) | Public SDK and CLI exercised end-to-end |
| 7 | [BFSI Industry Use Cases](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/07_bfsi_industry_usecases.ipynb) | Fraud triage, KYC distillation, compliance graph memory |
| 8 | [Industry Showcase: Other Domains](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/08_industry_showcase_other_domains.ipynb) | Healthcare, legal, retail, general: 8 more domain agents |
| 9 | [RL Training, Rewards & Eval](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/09_rl_training_rewards_and_eval.ipynb) | Reward shaping, trajectory eval metrics, reward-model training |

RAG is covered by `examples/USECASES/` notebooks 05 and 15 below rather than by a separate
`docs/notebooks` entry.

## docs/notebooks: Feature Notebooks (10-28)

The rest of the feature surface: every RL algorithm for real, sandboxed execution, the tool
library, reward defenses, and more.

| # | Notebook | Feature | What it shows |
|---|---|---|---|
| 10 | [Sandboxed Execution](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/10_sandboxed_execution_openenv.ipynb) | Sandboxed execution (OpenEnv) | A real `openenv.core.Environment` (Observation/Action/Rubric) driven through the spine's harness, conformance check + replay included. CPU-only. |
| 11 | [Docs-to-Training-Data Generator](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/11_docs_to_training_data_generator.ipynb) | Docs-to-training-data generator | Turns a document corpus into a difficulty-tagged, grounded QA set: real `generate_qa_from_corpus` → `label_difficulty` → curriculum balance, over a live Qwen2.5-3B. |
| 12 | [Multi-Agent Orchestration](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/12_multiagent_orchestration_langgraph.ipynb) | Multi-agent orchestration (LangGraph) | `AgentTuneGraph` composes a real rollout node + two divergent LLM-judge nodes into a reward function driving a real GRPO LoRA step. |
| 13 | [Tool Library Showcase](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/13_tool_library_showcase.ipynb) | Tool library | Real calls to the builtin SQL, code-execution, file I/O, and search tools, plus a small agent episode wired to two of them. |
| 14 | [Reward Defenses + Groundedness](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/14_reward_defenses_and_groundedness.ipynb) | Reward-hacking defenses, groundedness scoring | Score clamping on over-retrieval, judge prompt rotation, relative (batch) vs absolute scoring, and groundedness scoring, all against a real local model. |
| 15 | [Core GRPO Training](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/15_agentic_grpo_real.ipynb) | RL training: GRPO (real GPU) | Real `TrlAgenticGrpo` (TRL `GRPOTrainer`, LoRA) fine-tunes SmolLM2-360M on a strict-contract reward; generations compared before/after to show the policy actually moved. |
| 16 | [Real Model Driving ReAct Strategy](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/16_agentic_strategy_real.ipynb) | Agent design: ReAct (real model) | A live SmolLM2-360M (not a scripted policy) drives `ReActStrategy` + `run_episode`, scored with the real agentic metrics (tac/ter/arr/scsr/rad/lcf). |
| 17 | [BCO Training](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/17_bco_real.ipynb) | RL training: BCO (real GPU) | Real `trl.experimental.bco.BCOTrainer` (LoRA) learns from unpaired desirable/undesirable labels (no chosen/rejected pairs needed), on SmolLM2-360M. |
| 18 | [GSPO Training](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/18_gspo_real.ipynb) | RL training: GSPO (real GPU) | Real TRL `GRPOTrainer` run in sequence-level importance-sampling mode (LoRA), on the same task/reward as the GRPO notebook so the two are directly comparable. |
| 19 | [KTO Training](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/19_kto_real.ipynb) | RL training: KTO (real GPU) | Real TRL `KTOTrainer` (LoRA) learns from unpaired binary desirable/undesirable labels instead of chosen/rejected pairs, on SmolLM2-360M. |
| 20 | [PPO Training](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/20_ppo_real.ipynb) | RL training: PPO (real GPU) | Trains a real reward model (TRL `RewardTrainer`) on preference data, then runs real `trl.experimental.ppo.PPOTrainer` (LoRA): actor-critic RL, unlike GRPO/RLOO's reward-function-only setup. |
| 21 | [RLOO Training](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/21_rloo_real.ipynb) | RL training: RLOO (real GPU) | Real TRL `RLOOTrainer` (LoRA), on-policy like GRPO but with a leave-one-out baseline instead of group-normalized advantage; same task/reward as notebook 15. |
| 22 | [Self-Heal DPO Training](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/22_self_heal_dpo_real.ipynb) | Self-healing → DPO (real GPU) | Closes the self-heal loop: builds `{prompt, chosen, rejected}` preference data via the real spine (`build_dataset`) from a detected failure, then runs real TRL `DPOTrainer` (LoRA) to actually correct the policy. |
| 23 | [Self-Heal TAC/TER Reward Ranking](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/23_self_heal_reward_real.ipynb) | Self-healing: reward ranking (real) | Runs the closed loop's TAC (Tool Argument Correctness) + TER (Tool Efficacy Reward) scoring for real (previously only exercised via a mock trajectory), to pick chosen vs rejected corrections. |
| 24 | [Vector Memory (real embedder)](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/24_vector_memory_real.ipynb) | Memory: VectorMemory (real embedder) | Swaps the GPU-free case study's bag-of-words `embed` for a real sentence-transformer (`all-MiniLM-L6-v2`); shows genuine semantic recall (matches by meaning, not shared words) versus a lexical-overlap baseline. |
| 25 | [lm-eval Integration](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/25_lm_eval_real.ipynb) | Standardized benchmarking (lm-eval), **not listed in the main features doc at all** | `agenttune.eval`'s `LMEvalConfig`/`LMEvalTask`/`LMEvalRunner` shells out to EleutherAI's real `lm_eval` CLI (`--model hf ...`) and parses results, standardized benchmark tasks (e.g. `arc_challenge`), not a bespoke eval. |
| 26 | [Live-LLM Failure Detection & Example Generation](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/26_self_heal_llm_real.ipynb) | Self-healing: failure detection & example generation (real) | Runs the failure classifier and corrective-example generator for real (unmodified production path, not the injected fakes the case studies use) by standing up a local OpenAI-compatible server backed by real Qwen2.5-3B-Instruct, then a real DPO retrain closes the loop. |
| 27 | [RAG Synthesis Pipeline](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/27_rag_synthesis_real.ipynb) | Docs-to-multi-hop-QA synthesis (real) | The full 6-stage `agenttune.rag.synthesis` pipeline (typed entity graph → path sampling → answer-first generation → chain-dependency verification → 2D difficulty balance → GRPO-ready split) run for real against a local LLM client instead of the CLI's hardcoded Groq client, found and fixed a real bug along the way (see below). |
| 28 | [Hybrid PRM Reward](https://github.com/Lexsi-Labs/AgentTune/blob/main/docs/notebooks/28_hybrid_prm_real.ipynb) | Reward-model distillation / Hybrid PRM (real, bug documented not fixed) | Runs the real, unmodified `hybrid_prm_reward` function on a well-formed tool call, a malformed one, and a no-tool-call response, documents an honest finding that it currently scores all three identically (1.0), not fixed. |

## examples/USECASES: Real Use-Case Notebooks (15)

Each run end-to-end for real: real small open-source models (≤3B params, fit comfortably on a
single 48GB GPU), real datasets, real training/inference, real evaluation sections. No
API-based models anywhere. See [examples/USECASES/README.md](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/README.md)
for the full write-up of each.

| # | Notebook | Real dataset | Real model(s) |
|---|---|---|---|
| 01 | [Agentic Spine Core Lifecycle](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/01_agentic_spine_core_lifecycle.ipynb) | GSM8K (single-hop filtered) | SmolLM2-360M-Instruct |
| 02 | [Agentic Distillation](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/02_agentic_distillation.ipynb) | SST-2 | SmolLM2-360M-Instruct |
| 03 | [Self-Heal Closed-Loop Distillation](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/03_selfheal_closedloop_distillation.ipynb) | NousResearch/hermes-function-calling-v1 | Qwen2.5-0.5B-Instruct + Qwen2.5-3B-Instruct |
| 04 | [Memory & Agent Strategies](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/04_memory_and_agent_strategies.ipynb) | MRPC, Zachary's Karate Club, GSM8K | SmolLM2-360M-Instruct + all-MiniLM-L6-v2 |
| 05 | [RAG Synthesis Pipeline](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/05_rag_synthesis_pipeline.ipynb) | HotpotQA | Qwen2.5-3B-Instruct + BAAI/bge-m3 |
| 06 | [DECIDE Business Rules](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/06_decide_business_rules.ipynb) | SST-2 | Qwen2.5-1.5B-Instruct |
| 07 | [RL Training Algorithms Zoo](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/07_rl_training_algorithms_zoo.ipynb) | SST-2 | SmolLM2-360M-Instruct |
| 08 | [lm-eval Standardized Benchmarks](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/08_lm_eval_standardized_benchmarks.ipynb) | ARC-Challenge, WinoGrande | Qwen3-0.6B |
| 09 | [Sandboxed Execution (OpenEnv)](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/09_sandboxed_execution_openenv.ipynb) | — (real `openenv` library types) | none (CPU-only) |
| 10 | [Multi-Agent Orchestration (LangGraph)](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/10_multiagent_orchestration_langgraph.ipynb) | hand-picked math problems (disclosed) | Qwen2.5-3B-Instruct ×3 roles |
| 11 | [Tool Library Showcase](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/11_tool_library_showcase.ipynb) | Enron emails (corbt/enron-emails) | SmolLM2-360M-Instruct |
| 12 | [Reward Defenses & Groundedness](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/12_reward_defenses_and_groundedness.ipynb) | — | Qwen2.5-0.5B-Instruct |
| 13 | [Industry Domain Agents](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/13_industry_domain_agents.ipynb) | — (disclosed constructed scenarios) | Qwen2.5-0.5B-Instruct |
| 14 | [Hybrid PRM Reward Bug](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/14_hybrid_prm_reward_bug.ipynb) | — | none (CPU-only) |
| 15 | [Financial RAG Agent (FinBench)](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/USECASES/15_financial_rag_agent.ipynb) | FinDER (real SEC 10-K filings, 490 S&P 500 companies) | Qwen2.5-3B-Instruct |

## Bugs found (and fixed) while running everything for real

Three genuine, reproducible bugs surfaced only by actually executing this code for real,
not visible from reading the source, and not caught by the existing test suite because the
tests mock the pieces that broke:

- **`examples/ppo_real.py`**: `apply_chat_template()` returns a `BatchEncoding` under
  the installed `transformers` version, which is *not* a `dict` subclass, so an
  `isinstance(x, dict)` guard silently failed to unwrap it; a whole `BatchEncoding` got
  nested into the PPO dataset's `input_ids` column. Fixed with `isinstance(x, list)` instead.
- **`src/agenttune/rag/synthesis/verify.py`**: `check_retrieval_necessity()` accessed
  `r.metadata` on search results, but `SearchResult` is a `TypedDict` (a plain `dict` at
  runtime, no attribute access). The existing 18 synthesis tests never caught this because
  they mock the search backend with attribute-style fake objects instead of real dicts;
  a real integration gap between the test doubles and the real `SQLiteFTSBackend`. Fixed to
  `r["metadata"]`.
- **`agenttune.agentic.rewards.builtin_rewards.hybrid_prm.hybrid_prm_reward`**: documented,
  not fixed (notebook 28): with the default `use_llm_judge=False`, the function still
  attempts an LLM call with `model_name="offline-deterministic-only"` (not a real provider),
  which is silently caught, and the fallback scoring path gives a well-formed tool call, a
  malformed one, and a missing one all the same reward (1.0); it does not currently rank
  them.

## Honest notes

- **Tool library**: `GitHubTool`, `SlackTool`, and `PlaywrightTool` exist in the same
  builtin tool library but aren't exercised in notebook 13; they need credentials
  (GitHub/Slack tokens) or packages (`playwright` + browser binaries) not present in
  this environment. That's an environment limitation, not a code gap.
- **Prompt rotation**: rotation is a real `random.randint` draw on every call; this
  run happened to land on the same rotation id across a small 4-call sample (disclosed
  in the notebook's own output), which is expected variance at that sample size, not a
  bug.
- **Relative scoring**: in this run, absolute-mode scoring gave both trajectories the
  same score (0.8/0.8) while relative mode told them apart (0.95/0.85), a real,
  observed instance of the effect this mode exists for, on one example.

## How to re-run

```bash
cd docs/notebooks
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

```bash
cd examples/USECASES
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

`_real_backends.py` (in `docs/notebooks/`) is required by notebook 14 and several of the
use-case notebooks. Notebooks 23 and 26 need `nest_asyncio` (they call `asyncio.run()` deep
inside production code that assumes a plain script, not a Jupyter kernel's already-running
event loop).
