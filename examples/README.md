# Examples

An index of everything under `examples/` — one flat folder, no subfolders, every script a
runnable `.py` (notebook versions are being added alongside). Everything here assumes you've
already run `pip install -e .` from the repo root — see
[`docs/getting-started/installation.md`](../docs/getting-started/installation.md) if not.

## Where to start

| Doc | Covers |
|---|---|
| **[`CASE_STUDIES.md`](CASE_STUDIES.md)** | **Start here.** 19 scripts covering the agentic spine lifecycle (build→train, distillation, self-heal, memory, strategy comparison) plus 5 industry domains (BFSI, healthcare, legal, retail, general). Being converted from injected/scripted stand-ins to real open-source models — each script's docstring says which state it's in. |
| [`REAL_EXAMPLES.md`](REAL_EXAMPLES.md) | Real models on a real GPU: every RL algorithm (GRPO/PPO/DPO/RLOO/BCO), real closed-loop retrain+deploy, real RAG training, real service/transport. Captured output and honest caveats for each. |
| [`DECIDE_EXAMPLES.md`](DECIDE_EXAMPLES.md) | DECIDE pipeline examples — sync vs. async execution, single-stage and multi-judge templates. |
| [`COVERAGE.md`](COVERAGE.md) | Maps every public `agenttune.agentic` capability to the example that exercises it. |
| [`USECASES/`](USECASES/README.md) | *Kept as its own folder.* 15 real, end-to-end use-case notebooks — small open-source models (≤3B params), real datasets, real training/inference/eval, no shared helpers. |
| [`USE_CASES_decide.ipynb`](USE_CASES_decide.ipynb) | 10 real business decisions (support routing, spam/toxicity detection, resume screening, etc.) made live by a real local open-source model — one self-contained notebook, no external files. |

## Other root-level scripts

Scripts bridging agentic-spine training with DECIDE validation/rewards: `bridge_quickstart.py`
plus 3 worked use cases (`use_case_agentic_1_sql_agent_with_decide_validation.py`,
`use_case_agentic_2_multi_tool_routing_with_decide.py`,
`use_case_agentic_3_agent_training_with_decide_rewards.py`).

| Script | What it does |
|---|---|
| `basic_usage.py` | Load a config, build a `GraphRunner` from a DECIDE template, run it. |
| `bfsi_kyc.py` | Full KYC triage pipeline — runs the `bfsi/kyc_triage` template, inspects the audit trail, extracts DPO training pairs. |
| `training_pipeline.py` | The full flywheel: run DECIDE pipelines → extract training data (DPO/BCO/trajectories) → train → deploy. |
| `episode_collect_eval.py` | Episode loops, collect mode, and eval mode — DECIDE's three execution modes. |
| `stage_wise_training_example.py` | Epoch-based agentic training using DECIDE pipelines for routing and evaluation. |
| `path_a_failure_replay_exgen.py` | Failure detection → replay → corrective training-example generation, standalone (see [Concepts: DECIDE & the closed loop](../docs/concepts/decide-and-closed-loop.md)). |
| `grpo_with_openenv_sandbox.py` | Safety demo — the security difference between running model-generated code locally vs. inside the sandboxed OpenEnv tool adapter. |
| `openenv_workflow_demo.py` | Minimal `create_openenv_tools()` usage. |
| `dummy_sandbox.py` | A toy in-process sandbox stand-in used by other examples/tests. |
| `real_evaluation_demo.py` | Trajectory evaluation over a DECIDE-run agent. |
| `presentation_demo.py` | Scripted walkthrough of the closed-loop failure/classification contracts. |
| `test_hybrid_prm.py` | Exercises the hybrid process-reward-model reward path — see the [Hybrid PRM notebook](../docs/notebooks/28_hybrid_prm_real.ipynb) for the documented gap in this path. |
| `trainer_config_openenv_example.yaml` | Sample trainer config wiring in the OpenEnv sandbox. |

## Where to next

- **Every capability, linked to a notebook that runs it for real**: [`docs/features.md`](../docs/features.md)
- **DECIDE & the closed loop, explained**: [`docs/concepts/decide-and-closed-loop.md`](../docs/concepts/decide-and-closed-loop.md)
- **The agentic spine's own reference**: [`src/agenttune/agentic/README.md`](../src/agenttune/agentic/README.md)
