# Scripts

Standalone operational scripts for the teacher→student distillation workflow (collect real
teacher rollouts, correct a student against them, train, evaluate, monitor, serve), plus a
couple of repo-maintenance tools. These aren't wired into the `agenttune` CLI or any package
entry point — run them directly with `python scripts/<name>.py [args]` (most take `--help`).

## Distillation pipeline

Roughly in pipeline order:

1. **`collect_teacher_rollouts.py`** — runs a batch of teacher episodes (via `GraphRunner` /
   `APIEngine` / `OfflineVLLMEngine`) using a "First-Thought Prefix" prompting strategy, to
   produce reference trajectories a student can be corrected against.
2. **`dagger_correction_loop.py`** — batch DAgger (Dataset Aggregation) loop: runs a student
   model against a prompt set, compares it to the teacher, and emits corrective
   `{prompt, chosen, rejected}`-style training data.
3. **`train_agentic_dpo.py`** — trains a student model with DPO (TRL) on the DAgger output
   (`--dataset_path`, `--model_name`, `--output_dir`, standard TRL hyperparameters).
4. **`train_neural_reward_model.py`** — trains a Bradley-Terry neural reward model (TRL
   `RewardTrainer`) on a `{prompt, chosen, rejected}` preference dataset.
5. **`eval_agentic_distillation.py`** — compares the distilled student against the teacher on
   agentic metrics (tool-call accuracy, search-count similarity) to check the student actually
   *acts* like the teacher, not just scores well.
6. **`monitor_judge_agreement.py`** — checks how often a small distilled judge/reward model
   agrees with the original teacher judgments it was trained to approximate.
7. **`analyze_results.py`** — reads a bulk classified-failures log and prints an error-taxonomy
   report (root causes, failure types, tools that crashed).
8. **`fastapi_app.py`** — a minimal FastAPI app that serves a distilled student model
   (`POST /query`) for manual testing; separate from the package's own
   `agenttune.agentic.service` app.
9. **`run_end_to_end.sh`** — orchestrates steps 2-4 (DAgger correction → distillation → reward
   model training) as one shell pipeline: `./run_end_to_end.sh [dataset_path] [student_model]
   [teacher_model]`.

## Self-healing / closed-loop tooling

- **`run_path_a.py`** — CLI runner for the failure-detection self-healing pipeline: scans a DECIDE audit
  log, classifies failures, and writes them out (`--audit-log`, `--continuous` for polling
  mode). See [Concepts: DECIDE & the closed loop](../docs/concepts/decide-and-closed-loop.md).
- **`generate_bulk_experiments.py`** — generates a synthetic bulk audit-log fixture (100
  randomized failure scenarios across text2sql/math domains) for exercising the closed-loop
  pipeline at scale without a real model.

## Repo maintenance

- **`check_secrets.sh`** — scans staged files (pre-commit hook) or the whole tree (`--all`)
  for hardcoded credential patterns. Wired into `.pre-commit-config.yaml`.
- **`verify.sh`** — re-checks the repo's "green state" by hand: runs the GPU-free case studies
  and a strict `mkdocs build`. `./verify.sh`, `./verify.sh studies`, or `./verify.sh docs`.
