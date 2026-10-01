"""
Stage-Wise Training Example using AgentTune + DECIDE

Demonstrates how to run epoch-based agentic training
using DECIDE pipelines for routing and evaluation.

Features:
  ✅ YAML-driven configuration (no code needed)
  ✅ Stage-wise execution: COLLECT → EVALUATE → TRAIN
  ✅ Multi-epoch training loop
  ✅ DECIDE for decision making at each stage
  ✅ GRPO/DPO integration for model updates
  ✅ Automatic reward computation

Usage:
    python examples/stage_wise_training_example.py
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def example_stage_wise_training():
    """
    Example: Stage-wise training with DECIDE.

    This example shows:
    1. Load DECIDE config from YAML
    2. Define training inputs
    3. Run stage-wise training (multiple epochs)
    4. Print summary of results
    """
    from agenttune.decide.graph_runner import GraphRunner
    from agenttune.decide.stage_wise_runner import StageWiseRunner

    logger.info("=" * 70)
    logger.info("  STAGE-WISE TRAINING EXAMPLE")
    logger.info("  Using AgentTune + DECIDE for agentic training")
    logger.info("=" * 70)

    # =========================================================================
    # Step 1: Load DECIDE configuration from YAML
    # =========================================================================
    logger.info("\n[Step 1] Loading DECIDE configuration...")

    config_path = "STAGE_WISE_TRAINING_EXAMPLE.yaml"
    if not Path(config_path).exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using generic sentiment analysis template instead")
        runner = GraphRunner.from_template("generic/sentiment_analysis")
    else:
        runner = GraphRunner.from_config(config_path)

    logger.info("✅ Configuration loaded")
    logger.info(f"   Algorithm: {runner.config.get('collect', {}).get('algorithm')}")
    logger.info(f"   Epochs: {runner.config.get('num_epochs', 5)}")

    # =========================================================================
    # Step 2: Define training inputs (dataset)
    # =========================================================================
    logger.info("\n[Step 2] Preparing training dataset...")

    training_inputs = [
        "This product is amazing!",
        "I absolutely love this!",
        "Great quality and fast shipping",
        "Not happy with my purchase",
        "The item broke after 1 day",
        "Highly recommend to everyone",
        "Waste of money",
        "Worth every penny",
        "Customer service was helpful",
        "Poor packaging, arrived damaged",
    ]

    logger.info(f"✅ Dataset prepared: {len(training_inputs)} samples")

    # =========================================================================
    # Step 3: Initialize stage-wise runner
    # =========================================================================
    logger.info("\n[Step 3] Initializing stage-wise training runner...")

    stage_runner = StageWiseRunner(runner, training_inputs)
    logger.info("✅ Runner initialized")

    # =========================================================================
    # Step 4: Run stage-wise training
    # =========================================================================
    logger.info("\n[Step 4] Running stage-wise training (COLLECT → EVALUATE → TRAIN)...")

    num_epochs = runner.config.get("num_epochs", 5)
    results = await stage_runner.run_stage_wise(num_epochs=num_epochs)

    # =========================================================================
    # Step 5: Print summary
    # =========================================================================
    logger.info("\n[Step 5] Training complete!")
    stage_runner.print_summary()

    # =========================================================================
    # Step 6: Detailed results
    # =========================================================================
    logger.info("\n[Step 6] Detailed epoch results:")

    for epoch_result in results.get("epoch_results", []):
        epoch_num = epoch_result.get("epoch")
        success = epoch_result.get("success")

        if success:
            collect = epoch_result.get("collect", {})
            evaluate = epoch_result.get("evaluate", {})
            train = epoch_result.get("train", {})

            samples = collect.get("metrics", {}).get("samples_collected", 0)
            avg_reward = evaluate.get("avg_reward", 0.0)
            train_status = train.get("status", "unknown")

            logger.info(
                f"  Epoch {epoch_num}: "
                f"collect={samples} samples | "
                f"eval_reward={avg_reward:.3f} | "
                f"train={train_status}"
            )
        else:
            error = epoch_result.get("error", "Unknown error")
            logger.warning(f"  Epoch {epoch_num}: FAILED - {error}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ Stage-wise training example completed!")
    logger.info("=" * 70)

    return results


async def example_step_by_step():
    """
    Example: Step-by-step control (for advanced users).

    Shows how to control individual stages manually.
    """
    from agenttune.decide.graph_runner import GraphRunner
    from agenttune.decide.stage_wise_runner import StageWiseRunner

    logger.info("\n" + "=" * 70)
    logger.info("  STEP-BY-STEP CONTROL EXAMPLE")
    logger.info("=" * 70)

    # Load config
    runner = GraphRunner.from_template("generic/sentiment_analysis")
    inputs = ["Good product!", "Bad experience"]

    stage_runner = StageWiseRunner(runner, inputs)

    # Manual epoch control
    logger.info("\nManually executing one epoch:")

    epoch_result = await stage_runner._run_epoch(epoch=1)

    logger.info(f"  Epoch result: {epoch_result.get('success')}")
    if epoch_result.get("success"):
        collect_metrics = epoch_result.get("collect", {}).get("metrics", {})
        eval_metrics = epoch_result.get("evaluate", {})

        logger.info(f"    COLLECT: {collect_metrics.get('samples_collected')} samples")
        logger.info(f"    EVALUATE: {eval_metrics.get('avg_reward', 0):.3f} reward")

    logger.info("=" * 70)


async def main():
    """Run all examples."""
    try:
        # Run main example
        await example_stage_wise_training()

        # Run step-by-step example
        await example_step_by_step()

        logger.info("\n✅ All examples completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
