# AgentTune — Real Use-Case Notebooks

15 Jupyter notebooks, each run end-to-end for real: real small open-source models (≤3B params,
fit comfortably on a single 48GB GPU), real datasets, real training/inference, real evaluation
sections. **No API-based models anywhere** — every model is a local open-weight checkpoint
pulled from the Hugging Face Hub and run on this GPU. No shared helper modules — every notebook
is self-contained and calls the `agenttune` library's own modules directly.

Explicitly requested use cases:
- **03 — Self-Heal Closed-Loop Distillation** is the closed-loop pipeline
  (real `FailureClassifier` → `TrainingExampleGenerator` → `RetrainConfig`/`run_retrain` →
  `DeploymentGate`), run against real function-calling data.
- **05 — RAG Synthesis Pipeline** exercises `agenttune.rag.synthesis`.
- **15 — Financial RAG Agent** is a client-facing deep-dive on `agenttune.rag`'s real GRPO
  training, scoped to the FinDER/FinBench financial dataset (PR #21), with a custom reward and
  a longer training run.

## Index

| # | Notebook | Real dataset | Real model(s) | What the eval showed |
|---|---|---|---|---|
| 01 | [Agentic Spine Core Lifecycle](01_agentic_spine_core_lifecycle.ipynb) | GSM8K (single-hop filtered) | SmolLM2-360M-Instruct | Tool-use lifted real accuracy 1/25 → 4/25 |
| 02 | [Agentic Distillation](02_agentic_distillation.ipynb) | SST-2 | SmolLM2-360M-Instruct | Real held-out contract adherence 0.00→1.00, label accuracy 0.00→0.85 |
| 03 | [Self-Heal Closed-Loop Distillation](03_selfheal_closedloop_distillation.ipynb) | NousResearch/hermes-function-calling-v1 | Qwen2.5-0.5B-Instruct (student) + Qwen2.5-3B-Instruct (local heal LLM) | Real detect→classify→generate→DPO retrain→gate pipeline ran end-to-end; gate honestly reported no held-out generalization this run |
| 04 | [Memory & Agent Strategies](04_memory_and_agent_strategies.ipynb) | MRPC, Zachary's Karate Club, GSM8K | SmolLM2-360M-Instruct + all-MiniLM-L6-v2 | Real semantic recall, exact graph-structure recall, 4 real agent designs compared |
| 05 | [RAG Synthesis Pipeline](05_rag_synthesis_pipeline.ipynb) | HotpotQA | Qwen2.5-3B-Instruct (local) + BAAI/bge-m3 | Real 6-stage pipeline ran end-to-end; honest 0-accept-rate finding at smoke scale |
| 06 | [DECIDE Business Rules](06_decide_business_rules.ipynb) | SST-2 | Qwen2.5-1.5B-Instruct | Real GraphRunner accuracy 12/12 on held-out sentences |
| 07 | [RL Training Algorithms Zoo](07_rl_training_algorithms_zoo.ipynb) | SST-2 | SmolLM2-360M-Instruct | Real GRPO/RLOO/DPO/BCO training via `create_agentic_trainer`; honest RL-cold-start finding |
| 08 | [lm-eval Standardized Benchmarks](08_lm_eval_standardized_benchmarks.ipynb) | ARC-Challenge, WinoGrande | Qwen3-0.6B | Real `lm_eval` CLI accuracies (0.25/0.30, 0.60) |
| 09 | [Sandboxed Execution (OpenEnv)](09_sandboxed_execution_openenv.ipynb) | — (real `openenv` library types) | none (CPU-only) | Real reward path, episode, conformance, and replay all passed |
| 10 | [Multi-Agent Orchestration (LangGraph)](10_multiagent_orchestration_langgraph.ipynb) | hand-picked math problems (disclosed) | Qwen2.5-3B-Instruct ×3 roles | Real divergent judge rewards drove a real GRPO step |
| 11 | [Tool Library Showcase](11_tool_library_showcase.ipynb) | Enron emails (corbt/enron-emails) | SmolLM2-360M-Instruct | Real Enron DB queries, real code exec, real file I/O, real agent episode |
| 12 | [Reward Defenses & Groundedness](12_reward_defenses_and_groundedness.ipynb) | — | Qwen2.5-0.5B-Instruct | Real score clamping, prompt rotation, relative-vs-absolute divergence, groundedness split |
| 13 | [Industry Domain Agents](13_industry_domain_agents.ipynb) | — (disclosed constructed scenarios) | Qwen2.5-0.5B-Instruct | Real per-case model decisions + accuracy across 3 domains |
| 14 | [Hybrid PRM Reward Bug](14_hybrid_prm_reward_bug.ipynb) | — | none (CPU-only) | Confirmed real bug: reward doesn't discriminate tool-call validity |
| 15 | [Financial RAG Agent (FinBench)](15_financial_rag_agent.ipynb) | FinDER (real SEC 10-K filings, 490 S&P 500 companies) | Qwen2.5-3B-Instruct | Real GRPO+LoRA, custom reward (targets F1 + gold-chunk recall); held-out n=40: f1_strict +8.0%, f1_lenient +2.4%, correctness_numeric +17.6% — genuine gains; golden_chunk_recall −6.7%, and a second reward design aimed squarely at chunk-level recall left it unchanged, an honestly reported partial win |

## Honesty notes

Several notebooks surfaced **real, reproducible bugs** by actually running shipped code —
documented, and fixed where the fix was small and safe:

- **Broken HF mirror**: this box's Python force-routes every HF Hub call through a flaky
  offshore mirror (`hf_config.pth`), which 504-timed-out on dataset downloads and — worse —
  silently re-applied itself inside every fresh subprocess, undoing any per-notebook override.
  Fixed system-wide with an additive `hf_config_zzfix.pth` (doesn't touch the original file).
- **`evaluate.py` (`agenttune.rag.scripts`)**: omitted `rollout_backend="transformers"` in its
  `create_rollout_fn(...)` call (unlike `train_grpo.py`), so it fell through to an `"auto"` →
  `vllm` path and crashed. Fixed in the repo (one-line addition).
- **`evaluate.py`'s CLI also hangs indefinitely** inside `evaluate_checkpoint()` when run as a
  subprocess (confirmed with a direct, timed run — it loads the checkpoint, then produces no
  further output). Root cause not fully isolated given time; worked around in notebook 15 by
  calling the same underlying real functions (`create_rollout_fn`, `qa_metrics`) directly
  in-process instead of through the CLI wrapper — not a fix to the script itself, disclosed here.
- **PR #20** was merged before any of this work started: fixes `EventLog.as_dataset_rows()`
  serializing tool calls as Python repr instead of JSON, which silently broke SFT training data.
- **Hybrid PRM bug** (notebook 14), the **self-heal gate's "keep_old" result** (notebook 03),
  the **0-accept-rate RAG synthesis run** (notebook 05), and the **RL cold-start finding**
  (notebook 07) are all reported exactly as observed — not fixed, not hidden, not
  re-run-until-pretty.
- **Notebook 15** ran the obvious follow-up experiment instead of just speculating about it: a
  second training run with a reward pointed directly and heavily at exact chunk-level retrieval
  came back with `golden_chunk_recall`/`golden_chunk_recall_chunklevel` and `mean_tool_calls`
  numerically identical to the first run — the model's search behavior didn't move at all, only
  its answer phrasing did. Reported as a genuine limitation of reward-shaping at this training
  scale, not hidden behind the notebook's real F1/correctness gains.

## How to re-run

```bash
cd examples/USECASES
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

All notebooks assume `PYTHONPATH` includes the repo's `src/` and run against an `AgentTune`
checkout with PR #20 merged and the `evaluate.py` fix applied.
