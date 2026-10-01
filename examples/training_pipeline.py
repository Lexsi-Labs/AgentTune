"""
Example 3: Full Training Pipeline — Decide → Extract → Train → Deploy

Demonstrates the complete AgentTune flywheel:
  1. Run Decide pipelines to generate audit data
  2. Extract training data from audit (DPO, BCO, or trajectories)
  3. Train a fine-tuned model (DPO or GRPO)
  4. Deploy the trained model back to Decide config
  5. (Optional) Build a multi-agent AgentTuneGraph from YAML config

Usage:
    python examples/training_pipeline.py

Set AGENTTUNE_MOCK=1 to skip real training (inspect structure only).
Set ALGORITHM=grpo to use GRPO instead of DPO.
"""

import json
import os
from pathlib import Path

import yaml

MOCK_MODE = os.environ.get("AGENTTUNE_MOCK", "1") == "1"
ALGORITHM = os.environ.get("ALGORITHM", "dpo")

AUDIT_PATH = "./audit_training.jsonl"
CONFIG_PATH = "./config.yaml"
TRAINER_CONFIG_PATH = "./trainer_config.yaml"
OUTPUT_DIR = f"./output/{ALGORITHM}_run"
TRAINED_MODEL_PATH = f"{OUTPUT_DIR}/checkpoint-final"


def print_step(n: int, title: str):
    print(f"\n{'─' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Step 1: Generate audit data from Decide pipelines
# ---------------------------------------------------------------------------


def generate_audit_data():
    """Simulate Decide pipeline runs writing audit data."""
    print("  Generating synthetic audit data for training...")

    audit_entries = []

    # Simulate 20 pipeline runs
    for i in range(20):
        pipeline_id = f"train-pipe-{i:03d}"
        verdict = "APPROVE" if i % 3 != 0 else "DENY"
        score = 8 if verdict == "APPROVE" else 3

        entries = [
            {
                "pipeline_id": pipeline_id,
                "template_id": "bfsi/kyc_triage",
                "stage_id": "extract",
                "stage_type": "llm_call",
                "input": f"Customer document #{i}",
                "output": {"income": 50000 + i * 1000, "dob": "1990-01-01"},
                "iteration": 1,
                "latency_ms": 300,
                "cost_usd": 0.001,
            },
            {
                "pipeline_id": pipeline_id,
                "template_id": "bfsi/kyc_triage",
                "stage_id": "decision_judge",
                "stage_type": "llm_judge",
                "input": f"Rate customer #{i}",
                "output": {"score": score, "decision": verdict, "explanation": f"Analysis {i}"},
                "iteration": 1,
                "latency_ms": 500,
                "cost_usd": 0.005,
            },
        ]

        # Add human correction for every 5th pipeline
        if i % 5 == 0:
            entries.append(
                {
                    "pipeline_id": pipeline_id,
                    "stage_id": "decision_judge",
                    "stage_type": "llm_judge",
                    "human_feedback": "rejected",
                    "model_output": json.dumps({"decision": verdict, "score": score}),
                    "human_output": json.dumps({"decision": "DENY", "score": 1}),
                    "human_explanation": f"Undisclosed risk factor for customer #{i}",
                    "input": f"Rate customer #{i}",
                }
            )

        verdict_stage = "approve" if verdict == "APPROVE" else "deny"
        entries.append(
            {
                "pipeline_id": pipeline_id,
                "stage_id": verdict_stage,
                "stage_type": "output",
                "verdict": verdict,
                "verdict_label": verdict.lower(),
                "input": "Final verdict",
                "output": {"verdict": verdict},
            }
        )

        audit_entries.extend(entries)

    with open(AUDIT_PATH, "w") as f:
        for entry in audit_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"  Generated {len(audit_entries)} audit entries for {20} pipelines")
    print(f"  Audit written to: {AUDIT_PATH}")


# ---------------------------------------------------------------------------
# Step 2: Extract training data
# ---------------------------------------------------------------------------


def extract_training_data():
    """Extract training data from the audit log."""
    from agenttune.decide.training_bridge import DecideToTrainerBridge

    bridge = DecideToTrainerBridge(AUDIT_PATH)

    if ALGORITHM == "dpo":
        data = bridge.extract_dpo_pairs("decision_judge")
        print(f"  DPO pairs extracted:       {len(data)}")
        if data:
            pair = data[0]
            print(
                f"  Sample rejected output:    {str(pair.get('model_output', pair.get('rejected_output', '')))[:60]}"
            )
            print(
                f"  Sample chosen output:      {str(pair.get('human_output', pair.get('chosen_output', '')))[:60]}"
            )
        return data

    elif ALGORITHM == "bco":
        data = bridge.extract_bco_labels("approve")
        print(f"  BCO labels extracted:      {len(data)}")
        approve = sum(1 for l in data if l["label"] == 1)
        deny = sum(1 for l in data if l["label"] == 0)
        print(f"  APPROVE labels (1):        {approve}")
        print(f"  DENY labels (0):           {deny}")
        return data

    else:  # grpo / ppo / rloo
        ds = bridge.extract_trajectories("decision_judge")
        print(f"  Trajectories extracted:    {len(ds)}")
        return ds


# ---------------------------------------------------------------------------
# Step 3: Create trainer
# ---------------------------------------------------------------------------


def create_trainer(training_data):
    """Build the agentic trainer for the chosen algorithm."""
    if MOCK_MODE:
        print(
            f"  [MOCK] Would build {ALGORITHM.upper()} trainer with {len(training_data) if hasattr(training_data, '__len__') else '?'} examples"
        )
        print("  [MOCK] Model: Qwen/Qwen2.5-0.5B-Instruct")
        print(f"  [MOCK] Output: {OUTPUT_DIR}")
        return None

    from agenttune.core.backend_factory import create_agentic_trainer

    trainer = create_agentic_trainer(
        algorithm=ALGORITHM,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=training_data,
        output_dir=OUTPUT_DIR,
        max_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        report_to="none",
    )
    return trainer


# ---------------------------------------------------------------------------
# Step 4: Train
# ---------------------------------------------------------------------------


def run_training(trainer):
    """Execute training loop."""
    if MOCK_MODE:
        print("  [MOCK] Skipping training — set AGENTTUNE_MOCK=0 to train")
        print(f"  [MOCK] Model would be saved to: {TRAINED_MODEL_PATH}")
        return {"total_steps": 5, "final_loss": 0.42}

    print(f"  Starting {ALGORITHM.upper()} training...")
    results = trainer.train()
    print(
        f"  Training complete — steps: {results.get('total_steps')}  loss: {results.get('final_loss'):.4f}"
    )
    return results


# ---------------------------------------------------------------------------
# Step 5: Deploy trained model back to Decide
# ---------------------------------------------------------------------------


def deploy_model():
    """Update config.yaml to use the trained model for the judge stage."""
    from agenttune.decide.model_deployment import ModelDeploymentBridge

    if MOCK_MODE:
        print(
            f"  [MOCK] Would deploy {TRAINED_MODEL_PATH} to stage 'decision_judge' in {CONFIG_PATH}"
        )
        return

    if not Path(CONFIG_PATH).exists():
        print("  config.yaml not found — skipping deployment")
        return

    bridge = ModelDeploymentBridge()
    bridge.deploy_trained_model(
        trained_model_path=TRAINED_MODEL_PATH,
        config_path=CONFIG_PATH,
        backend="transformers",
        stage_model_map={"decision_judge": TRAINED_MODEL_PATH},
    )
    print(f"  Deployed to: {CONFIG_PATH}")
    print(f"  Backup at:   {CONFIG_PATH}.backup")


# ---------------------------------------------------------------------------
# Step 6 (bonus): Show YAML-driven trainer config
# ---------------------------------------------------------------------------


def show_yaml_trainer_config():
    """Show how TrainerConfigBridge lets you configure everything from YAML."""
    from agenttune.decide.trainer_config_bridge import TrainerConfigBridge

    # Create a minimal trainer config YAML
    cfg = {
        "training": {
            "algorithm": ALGORITHM,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": OUTPUT_DIR,
            "max_steps": 5,
            "rollout": {
                "backend": "api",
                "api_model": "llama-3.1-8b-instant",
                "api_base_url": "https://api.groq.com/openai/v1",
                "api_key": "your-groq-key",
                "max_steps": 2,
            },
            "reward_funcs": ["correctness_reward"],
            "multi_agent": {"enabled": False},
        },
        "decide_bridge": {
            "enabled": True,
            "audit_path": AUDIT_PATH,
            "stage_id": "decision_judge",
        },
    }

    config_path = Path(TRAINER_CONFIG_PATH)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"  Written: {TRAINER_CONFIG_PATH}")
    bridge = TrainerConfigBridge(TRAINER_CONFIG_PATH)
    print(f"  Algorithm:     {bridge.algorithm}")
    print(f"  Reward funcs:  {bridge.build_reward_funcs()}")
    print(f"  PEFT config:   {bridge.get_peft_config()}")
    print()
    print("  With TrainerConfigBridge you can:")
    print("  • bridge.step_build_rollout_engine()   → Phase 2")
    print("  • bridge.step_build_tools()            → Phase 3")
    print("  • bridge.step_build_reward_funcs()     → Phase 4")
    print("  • bridge.step_build_judge()            → Phase 4")
    print("  • bridge.step_build_graph()            → Phase 5 (multi-agent)")
    print("  • bridge.step_get_peft_config()        → Phase 6")
    print("  • bridge.step_build_trainer(dataset)   → Phase 7")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  AgentTune — Full Training Pipeline Example")
    print("=" * 60)
    print(f"  Algorithm:  {ALGORITHM.upper()}")
    print(f"  Mock mode:  {'ON (no real training)' if MOCK_MODE else 'OFF (real training)'}")

    print_step(1, "Generate Audit Data from Decide Pipelines")
    generate_audit_data()

    print_step(2, "Extract Training Data from Audit")
    training_data = extract_training_data()

    print_step(3, "Create Trainer")
    trainer = create_trainer(training_data)

    print_step(4, "Run Training")
    results = run_training(trainer)
    print(f"  Results: {results}")

    print_step(5, "Deploy Trained Model")
    deploy_model()

    print_step(6, "YAML-Driven Trainer Config (TrainerConfigBridge)")
    show_yaml_trainer_config()

    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"  Audit log:      {AUDIT_PATH}")
    print(f"  Model output:   {OUTPUT_DIR}")
    print(f"  Trainer config: {TRAINER_CONFIG_PATH}")
    print()
    print("  Next steps:")
    print("  • Review audit_training.jsonl to see all pipeline executions")
    print("  • Open notebooks/04_training_and_deployment.ipynb for more detail")
    print("  • Set AGENTTUNE_MOCK=0 to run real training")
    print()


if __name__ == "__main__":
    main()
