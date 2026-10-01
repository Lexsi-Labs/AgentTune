"""
Quick start example: Decide → Train → Deploy complete flywheel.

This example shows:
1. Generate audit data from Decide inference
2. Extract and review training data
3. Train with all 5 algorithms
4. Deploy trained models back to Decide
5. Monitor improvements

NOTE: This is pseudo-code showing the flow. Actual execution requires:
- Running Decide pipeline (generates audit.jsonl)
- Human review stage to mark rejected outputs (for DPO)
- Configured rollout engine for agentic mode

The `bfsi/kyc_triage` template's `output_human_review` stage is a real pause-and-resume
stage (src/agenttune/decide/stages/human_review.py): it writes the reviewer prompt to
./pending_reviews/<review_id>.json and polls ./review_decisions/<review_id>.txt for up
to timeout_seconds (24h in the template) before giving up. There's no real reviewer in
this unattended demo, so `_simulated_human_reviewer` below stands in for one: it watches
for pending reviews, prints them (so you can see exactly what a human would see), and
submits a decision through that same file-based mechanism -- deny/approve based on the
judge score, the same signal a real reviewer would be shown. Swap it for a real UI/queue
in production; the pipeline itself doesn't need to change.
"""

import asyncio
import json
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Stand-in for a human reviewer (see module docstring)
# ─────────────────────────────────────────────────────────────────


async def _simulated_human_reviewer(poll_interval: float = 0.5) -> "asyncio.Task":
    """Watch ./pending_reviews for new requests and auto-decide them.

    Runs until cancelled. This stage only ever sees judge scores 5-7 (that's exactly
    the band decision_judge routes to REVIEW instead of auto-approve/auto-deny), so
    a 6/10 cutoff -- not 7 -- is what actually splits that band into both chosen
    (approved) and rejected (denied) outcomes; a stand-in for what a compliance
    officer would actually judge.
    """
    review_dir = Path("./pending_reviews")
    decision_dir = Path("./review_decisions")
    decision_dir.mkdir(exist_ok=True)
    seen = set()

    while True:
        if review_dir.is_dir():
            for path in review_dir.glob("*.json"):
                review_id = path.stem
                if review_id in seen:
                    continue
                seen.add(review_id)
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                prompt = data.get("prompt", "")
                print("\n--- Simulated human review (stand-in; see module docstring) ---")
                print(prompt.strip()[:500])

                m = re.search(r"Judge Score:\s*(\d+)/10", prompt)
                score = int(m.group(1)) if m else 0
                decision = "approved" if score >= 6 else "denied"
                (decision_dir / f"{review_id}.txt").write_text(decision)
                print(f"[simulated reviewer] decision={decision} (judge score={score}/10)")

                # The approve/deny file above only routes the *pipeline* (it's what
                # HumanReviewStage polls for). DecideToTrainerBridge.extract_dpo_pairs()
                # reads a different, separate shape straight off the audit log --
                # {stage_id, human_feedback: "rejected", model_output, human_output} --
                # which nothing else writes. Every case that reaches human review does
                # so *because* decision_judge itself couldn't confidently decide, so the
                # human's firm APPROVE/DENY is a real preference signal over that
                # uncertain call: write the pair ourselves.
                model_output = data.get("stage_outputs", {}).get("decision_judge", {})
                human_output = dict(model_output)
                human_output["decision"] = "APPROVE" if decision == "approved" else "DENY"
                human_output["score"] = 9 if decision == "approved" else 2
                human_output["explanation"] = (
                    f"Compliance officer {decision} this case on manual review."
                )
                dpo_entry = {
                    "stage_id": "decision_judge",
                    "human_feedback": "rejected",
                    "input": prompt,
                    # DPO needs chosen/rejected as text completions, not JSON objects --
                    # serialize decision_judge's structured output to a string.
                    "model_output": json.dumps(model_output),
                    "human_output": json.dumps(human_output),
                    "human_explanation": human_output["explanation"],
                }
                with open("./audit.jsonl", "a") as f:
                    f.write(json.dumps(dpo_entry) + "\n")
        await asyncio.sleep(poll_interval)


# ─────────────────────────────────────────────────────────────────
# Step 1: Generate audit data (run Decide inference)
# ─────────────────────────────────────────────────────────────────


async def generate_audit_data():
    """Run Decide pipeline, generating audit.jsonl with execution traces."""
    from agenttune.decide import GraphRunner

    # Load template
    runner = GraphRunner.from_template(template_id="bfsi/kyc_triage", config_path="./config.yaml")

    # A few complete applications, deliberately spanning the score range so some
    # land in output_human_review (5 <= score < 8) rather than all auto-approving
    # or auto-denying -- that's what actually exercises the reviewer above.
    customer_applications = [
        "Name: John Doe, DOB: 1989-03-14, SSN: 123-45-6789, "
        "Address: 42 Maple St, Springfield, Income: $120000, Employment: employed",
        "Name: Jane Smith, DOB: 1996-07-22, SSN: 987-65-4321, "
        "Address: 9 Birch Ave, Rivertown, Income: $95000, Employment: employed",
        "Name: Carlos Mendez, DOB: 1975-11-02, SSN: 555-12-3456, "
        "Address: 17 Cedar Ln, Lakeside, Income: $38000, Employment: self_employed",
    ]

    reviewer_task = asyncio.create_task(_simulated_human_reviewer())
    try:
        for application in customer_applications:
            await runner.run(input_text=application)
            # Audit log entry written automatically
            # Look for: ./audit.jsonl
    finally:
        reviewer_task.cancel()

    print("✓ Generated audit data: ./audit.jsonl")


# ─────────────────────────────────────────────────────────────────
# Step 2: Extract and review training data
# ─────────────────────────────────────────────────────────────────


def extract_training_data():
    """Inspect training data before training."""
    from agenttune.decide import DecideToTrainerBridge

    bridge = DecideToTrainerBridge("./audit.jsonl")

    # DPO pairs (from human rejections). "decision_judge" is the stage
    # routed to human_review in bfsi/kyc_triage.yaml (its own header comment
    # documents this as the DPO target) — "income_agent" is just a parallel
    # sub-stage that never reaches human_review, so it always yields 0 pairs.
    dpo_pairs = bridge.extract_dpo_pairs(stage_id="decision_judge")
    print(f"✓ Found {len(dpo_pairs)} DPO pairs")
    if dpo_pairs:
        print(f"  Example: {dpo_pairs[0]}")

    # BCO labels (from APPROVE/DENY verdicts)
    bco_labels = bridge.extract_bco_labels(output_stage_id="output")
    print(f"✓ Found {len(bco_labels)} BCO labels")

    # RL trajectories (for GRPO/PPO/RLOO)
    trajectories = bridge.extract_trajectories(stage_id="risk_assessment")
    print(f"✓ Found {len(trajectories)} trajectories for RL training")

    return dpo_pairs, bco_labels, trajectories


# ─────────────────────────────────────────────────────────────────
# Step 3: Train with different algorithms
# ─────────────────────────────────────────────────────────────────


def train_dpo_model():
    """Train with DPO (Direct Preference Optimization)."""
    from agenttune.decide import train_from_audit

    print("\n🔵 Training DPO model...")
    trainer = train_from_audit(
        audit_path="./audit.jsonl",
        stage_id="decision_judge",
        algorithm="dpo",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="./runs/dpo_income_v1",
        num_epochs=3,
        learning_rate=1e-5,
        batch_size=8,
    )

    results = trainer.train()
    print("✓ DPO training complete: ./runs/dpo_income_v1/")
    print(f"  Loss: {results.get('loss', 'N/A')}")
    return results


def train_bco_model():
    """Train with BCO (Binary Classification Optimization)."""
    from agenttune.decide import train_from_audit

    print("\n🔵 Training BCO model...")
    trainer = train_from_audit(
        audit_path="./audit.jsonl",
        stage_id="output",
        algorithm="bco",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="./runs/bco_verdict_v1",
        num_epochs=5,
    )

    results = trainer.train()
    print("✓ BCO training complete: ./runs/bco_verdict_v1/")
    return results


def train_grpo_model():
    """Train with GRPO (Group Relative Policy Optimization)."""
    from agenttune.decide import train_from_audit

    print("\n🔵 Training GRPO model...")
    trainer = train_from_audit(
        audit_path="./audit.jsonl",
        stage_id="risk_assessment",
        algorithm="grpo",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="./runs/grpo_risk_v1",
        tools=[
            {"name": "sql_query"},
            {"name": "web_search"},
        ],
        reward_funcs=[],  # Use judge scores from audit
        num_epochs=3,
    )

    results = trainer.train()
    print("✓ GRPO training complete: ./runs/grpo_risk_v1/")
    return results


def train_ppo_model():
    """Train with PPO (Proximal Policy Optimization)."""
    from agenttune.decide import train_from_audit

    print("\n🔵 Training PPO model...")
    trainer = train_from_audit(
        audit_path="./audit.jsonl",
        stage_id="extraction",
        algorithm="ppo",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="./runs/ppo_extract_v1",
    )

    results = trainer.train()
    print("✓ PPO training complete: ./runs/ppo_extract_v1/")
    return results


def train_rloo_model():
    """Train with RLOO (Leave-One-Out Baseline)."""
    from agenttune.decide import train_from_audit

    print("\n🔵 Training RLOO model...")
    trainer = train_from_audit(
        audit_path="./audit.jsonl",
        stage_id="research",
        algorithm="rloo",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="./runs/rloo_research_v1",
    )

    results = trainer.train()
    print("✓ RLOO training complete: ./runs/rloo_research_v1/")
    return results


# ─────────────────────────────────────────────────────────────────
# Step 4: Deploy trained models back to Decide
# ─────────────────────────────────────────────────────────────────


def deploy_models():
    """Deploy trained models to Decide for improved inference."""
    from agenttune.decide import deploy_trained_model

    # Global deployment (one model for all stages)
    print("\n🟢 Deploying trained model...")
    deploy_trained_model(
        trained_model_path="./runs/dpo_income_v1/checkpoint-final",
        config_path="./config.yaml",
        backend="transformers",  # Local inference
        backup=True,  # Auto-backup original config
    )
    print("✓ Model deployed: ./runs/dpo_income_v1/checkpoint-final")
    print("  Backend: transformers (local)")
    print("  Backup: ./config.yaml.backup")


def deploy_models_ab_test():
    """Deploy with A/B testing (stage-specific models)."""
    from agenttune.decide import deploy_trained_model

    # Different models per stage
    print("\n🟢 Deploying with A/B testing...")
    deploy_trained_model(
        trained_model_path="./runs/dpo_income_v1/checkpoint-final",
        config_path="./config.yaml",
        backend="transformers",
        stage_model_map={
            # New trained models (canary)
            "income_agent": "./runs/dpo_income_v1/checkpoint-final",
            # Keep old models (baseline)
            "fraud_check": "gpt-4o",
            "kyc_extract": "gpt-4-turbo",
        },
    )
    print("✓ A/B test deployed")
    print("  income_agent: local fine-tuned (canary)")
    print("  fraud_check: gpt-4o (baseline)")
    print("  kyc_extract: gpt-4-turbo (baseline)")


# ─────────────────────────────────────────────────────────────────
# Step 5: Monitor & rollback if needed
# ─────────────────────────────────────────────────────────────────


def check_deployment_status():
    """Check current deployment status."""
    from agenttune.decide import ModelDeploymentBridge

    status = ModelDeploymentBridge.get_deployment_status("./config.yaml")
    print("\n📊 Deployment Status:")
    print(f"  Model: {status['default_model']}")
    print(f"  Backend: {status['backend']}")
    print(f"  Model exists: {status['model_exists']}")
    if status.get("stage_models"):
        print(f"  Stage-specific models: {len(status['stage_models'])}")
    return status


def rollback_if_issues():
    """Rollback deployment if issues detected."""
    from agenttune.decide import ModelDeploymentBridge

    print("\n🔴 Rolling back deployment...")
    ModelDeploymentBridge.rollback_deployment("./config.yaml")
    print("✓ Rollback complete")
    print("  Config restored from: ./config.yaml.backup")


# ─────────────────────────────────────────────────────────────────
# Complete end-to-end pipeline
# ─────────────────────────────────────────────────────────────────


async def main():
    """Run complete Decide → Train → Deploy pipeline."""
    print("=" * 70)
    print("AgentTune Decide ↔ Trainer Bridge — Quick Start")
    print("=" * 70)

    # Step 1: Generate audit data
    print("\n[1/6] Generate audit data from Decide inference...")
    await generate_audit_data()

    # Step 2: Extract training data
    print("\n[2/6] Extract and review training data...")
    dpo_pairs, bco_labels, trajectories = extract_training_data()

    # Step 3a: Train DPO
    print("\n[3/6] Train all 5 algorithms...")
    train_dpo_model()

    # Step 3b: Train BCO
    train_bco_model()

    # Step 3c: Train GRPO
    train_grpo_model()

    # Step 3d: Train PPO
    train_ppo_model()

    # Step 3e: Train RLOO
    train_rloo_model()

    # Step 4a: Deploy (simple)
    print("\n[4/6] Deploy trained models...")
    deploy_models()

    # Step 4b: Or deploy with A/B testing
    # deploy_models_ab_test()

    # Step 5: Monitor
    print("\n[5/6] Monitor deployment...")
    check_deployment_status()

    # Step 5b: Rollback if needed
    # rollback_if_issues()

    # Step 6: Next iteration
    print("\n[6/6] Generate new audit data with improved model...")
    print("  ✓ Run Decide again to see better decisions")
    print("  ✓ Collect human feedback on new outputs")
    print("  ✓ Train again on expanded audit data")
    print("  ✓ Deploy again for continuous improvement")

    print("\n" + "=" * 70)
    print("✅ Complete pipeline executed!")
    print("=" * 70)
    print("\nFlywheel Status:")
    print("  ✓ Audit data generated")
    print("  ✓ Training data extracted")
    print("  ✓ 5 algorithms trained (DPO, BCO, GRPO, PPO, RLOO)")
    print("  ✓ Models deployed to Decide")
    print("  ✓ Ready for next iteration")


# ─────────────────────────────────────────────────────────────────
# Run examples
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run full pipeline
    asyncio.run(main())

    # Or run individual steps:
    # extract_training_data()
    # train_dpo_model()
    # deploy_models()
    # check_deployment_status()
