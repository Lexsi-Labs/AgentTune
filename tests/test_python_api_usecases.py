#!/usr/bin/env python
"""
Comprehensive tests for all Python API use cases with open-source models.

Tests:
1. Basic inference (GraphRunner.run)
2. Data collection (CollectRunner.run)
3. Evaluation (EvalRunner.run)
4. Stage-wise training (StageWiseRunner.run_stage_wise)
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agenttune.decide.collect_runner import CollectRunner
from agenttune.decide.eval_runner import EvalRunner
from agenttune.decide.graph_runner import GraphRunner
from agenttune.decide.stage_wise_runner import StageWiseRunner

# Runs the DECIDE templates against a real local model: 3.8GB peak RSS measured,
# the single largest contributor left on the fast gate (7.8GB CPU CI runner).
# Runs under -m qwen_e2e.
pytestmark = pytest.mark.qwen_e2e


@pytest.mark.asyncio
async def test_use_case_1_basic_inference():
    """Test Use Case 1: Basic Inference with GraphRunner"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Inference (GraphRunner)")
    print("=" * 70)

    try:
        # Create runner
        runner = GraphRunner.from_template("generic/text_classify")

        # Single inference
        input_text = "This product is amazing and works perfectly!"
        print(f"\nInput: {input_text}")

        state = await runner.run(input_text)

        # Validate result
        assert state.is_complete, "Pipeline should be complete"
        assert state.verdict is not None, "Verdict should not be None"
        if state.confidence is not None:
            assert 0 <= state.confidence <= 10, "Confidence should be 0-10"
        assert state.elapsed_seconds > 0, "Should have non-zero elapsed time"

        print("\n✅ Results:")
        print(f"   Verdict: {state.verdict}")
        print(f"   Confidence: {state.confidence or 'N/A'}/10")
        print(f"   Time: {state.elapsed_seconds:.2f}s")

        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_use_case_2_data_collection():
    """Test Use Case 2: Data Collection with CollectRunner"""
    print("\n" + "=" * 70)
    print("TEST 2: Data Collection (CollectRunner)")
    print("=" * 70)

    try:
        # Create runner
        runner = GraphRunner.from_template("generic/text_classify")
        collector = CollectRunner(runner)

        # Configure
        runner.config["collect"] = {
            "algorithm": "grpo",
            "stage_id": "process",
            "output_path": "./test_collected_data.jsonl",
            "min_samples": 10,
        }
        runner.config["episode"] = {
            "n_episodes": 10,
            "batch_size": 2,
            "shuffle_inputs": True,
            "seed": 42,
        }

        # Generate test inputs
        training_inputs = [
            "This is amazing!",
            "Terrible experience",
            "It works well",
        ]

        print(f"\nCollecting {10} episodes from {len(training_inputs)} inputs...")

        # Run collection
        n_records = await collector.run(training_inputs)

        # Validate
        assert n_records > 0, "Should have collected records"
        assert n_records <= 10, "Should not exceed min_samples"

        # Check JSONL format
        with open("./test_collected_data.jsonl") as f:
            records = [json.loads(line) for line in f.readlines()]

        assert len(records) == n_records, "Record count mismatch"
        assert all("input" in r for r in records), "All records should have input"
        assert all("verdict" in r for r in records), "All records should have verdict"
        assert all("reward" in r for r in records), "All records should have reward"

        print("\n✅ Results:")
        print(f"   Records collected: {len(records)}")
        print(f"   Sample record: {json.dumps(records[0], indent=2)[:200]}...")

        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_use_case_3_evaluation():
    """Test Use Case 3: Evaluation with EvalRunner"""
    print("\n" + "=" * 70)
    print("TEST 3: Evaluation (EvalRunner)")
    print("=" * 70)

    try:
        # Create test set
        test_set = [
            {"input": "This is amazing!", "expected_verdict": "APPROVE"},
            {"input": "Terrible experience", "expected_verdict": "DENY"},
            {"input": "It works well", "expected_verdict": "APPROVE"},
        ]

        # Save test set
        with open("./test_eval_set.jsonl", "w") as f:
            for record in test_set:
                f.write(json.dumps(record) + "\n")

        print(f"\nEvaluating on {len(test_set)} test cases...")

        # Create runner and evaluator
        runner = GraphRunner.from_template("generic/text_classify")
        evaluator = EvalRunner(runner)

        # Run evaluation
        results = await evaluator.run("./test_eval_set.jsonl")

        # Validate
        assert results["n_evaluated"] == len(test_set), "Should evaluate all cases"

        metrics = results.get("metrics", {})
        assert "accuracy" in metrics, "Should compute accuracy"
        assert 0 <= metrics["accuracy"] <= 1, "Accuracy should be 0-1"

        print("\n✅ Results:")
        print(f"   Test cases: {results['n_evaluated']}")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"   Precision: {metrics.get('precision', 0):.2%}")
        print(f"   Recall: {metrics.get('recall', 0):.2%}")
        print(f"   F1: {metrics.get('f1', 0):.2%}")

        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_use_case_4_stage_wise_training():
    """Test Use Case 4: Stage-Wise Training"""
    print("\n" + "=" * 70)
    print("TEST 4: Stage-Wise Training (StageWiseRunner)")
    print("=" * 70)

    try:
        # Create training inputs
        training_inputs = [
            "Sample positive 1",
            "Sample negative 1",
            "Sample positive 2",
        ]

        print(f"\nRunning stage-wise training with {len(training_inputs)} samples")
        print("   Epochs: 2")

        # Create runner
        runner = GraphRunner.from_template("generic/text_classify")

        # Configure for stage-wise training
        runner.config["run_mode"] = "train_stage_wise"
        runner.config["episode"] = {
            "n_episodes": 5,
            "batch_size": 2,
            "shuffle_inputs": True,
            "seed": 42,
        }
        runner.config["collect"] = {
            "algorithm": "grpo",
            "output_path": "./test_epochs/epoch_{epoch}/collected_data.jsonl",
            "min_samples": 5,
        }

        # Create stage-wise runner
        stage_runner = StageWiseRunner(runner, training_inputs)

        # Run training
        results = await stage_runner.run_stage_wise(num_epochs=2)

        # Validate results
        assert results["total_epochs"] == 2, "Should run 2 epochs"
        assert results["successful_epochs"] > 0, "Should have successful epochs"
        assert len(results["epoch_results"]) > 0, "Should have epoch results"

        print("\n✅ Results:")
        print(f"   Total epochs: {results['total_epochs']}")
        print(f"   Successful: {results['successful_epochs']}")
        print(f"   Overall avg reward: {results['overall_avg_reward']:.3f}")

        for epoch_result in results["epoch_results"]:
            if epoch_result["success"]:
                epoch_num = epoch_result["epoch"]
                avg_reward = epoch_result["evaluate"]["avg_reward"]
                samples = epoch_result["collect"]["metrics"]["samples_collected"]
                print(f"   Epoch {epoch_num}: {samples} samples, avg_reward={avg_reward:.3f}")

        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PYTHON API USE CASES - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Inference", test_use_case_1_basic_inference),
        ("Data Collection", test_use_case_2_data_collection),
        ("Evaluation", test_use_case_3_evaluation),
        ("Stage-Wise Training", test_use_case_4_stage_wise_training),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} passed")
    print(f"{'='*70}\n")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
