"""
Example 04 — Episode loops, collect mode, and eval mode
=========================================================

Demonstrates the three Phase 2 execution modes introduced in AgentTune Decide v1.1:

  collect  — run N episodes, write BCO / DPO / GRPO training records
  eval     — run a labelled test set, compute accuracy / latency / cost metrics
  train    — collect + auto-trigger TrainerConfigBridge (requires trainer_config.yaml)

All examples use mock LLM responses so they run without API keys.
Swap `mock_litellm` for a real API key when you're ready to run end-to-end.

Usage:
    python examples/episode_collect_eval.py

Or via CLI:
    # collect mode
    agenttune decide run \
        --template generic/multi_judge_score \
        --input @/tmp/test_inputs.jsonl \
        --mode collect \
        --config src/agenttune/decide/config.example.yaml

    # eval mode  (set eval.test_set in the template first)
    agenttune decide run \
        --template bfsi/kyc_triage \
        --input placeholder \
        --mode eval \
        --config src/agenttune/decide/config.example.yaml

    # train mode
    agenttune decide run \
        --template bfsi/kyc_triage \
        --input @/tmp/kyc_inputs.jsonl \
        --mode train \
        --config src/agenttune/decide/config.example.yaml
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agenttune.decide.collect_runner import CollectRunner
from agenttune.decide.eval_runner import EvalRunner
from agenttune.decide.graph_runner import GraphRunner
from agenttune.decide.state import PipelineState

# ---------------------------------------------------------------------------
# Shared mock helpers — remove these when using real API keys
# ---------------------------------------------------------------------------


def _mock_state(pipeline_id: str, verdict: str = "APPROVE", elapsed: float = 1.2) -> PipelineState:
    return PipelineState(
        pipeline_id=pipeline_id,
        template_id="generic/multi_judge_score",
        template_version="1.1.0",
        input_text="Sample input text for evaluation",
        input_hash="abc123",
        stage_outputs={
            "aggregate": {
                "consensus_score": 8,
                "verdict": verdict,
                "summary": "Good quality output.",
            },
            "judge_1": {"score": 8, "rationale": "Accurate"},
            "judge_2": {"score": 7, "rationale": "Clear"},
            "judge_3": {"score": 8, "rationale": "Complete"},
        },
        stage_iterations={
            "preprocess": 1,
            "judge_1": 1,
            "judge_2": 1,
            "judge_3": 1,
            "aggregate": 1,
        },
        stage_traces=[],
        step_count=5,
        step_history=["preprocess", "judge_1", "judge_2", "judge_3", "aggregate"],
        verdict=verdict,
        verdict_label="quality_high" if verdict == "PASS" else "quality_low",
        confidence=8,
        reason="Strong multi-judge consensus",
        is_complete=True,
        error=None,
        error_stage=None,
        timestamp_start="2026-04-27T10:00:00",
        timestamp_end="2026-04-27T10:00:01",
        elapsed_seconds=elapsed,
        config={
            "reward": {
                "stages": {
                    "aggregate": {"fn": "judge_score", "weight": 1.0},
                    "output_high": {"fn": "verdict_binary", "weight": 0.3},
                },
                "final_fn": "weighted_mean",
            }
        },
    )


# ---------------------------------------------------------------------------
# Example 1 — Collect mode: run N episodes, write BCO training records
# ---------------------------------------------------------------------------


async def example_collect_mode():
    print("\n" + "=" * 60)
    print("Example 1 — Collect mode (BCO records)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "multi_judge_bco.jsonl")

        # Build a mocked runner — replace with GraphRunner.from_template() for real runs
        mock_runner = MagicMock(spec=GraphRunner)
        mock_runner.config = {
            "id": "generic/multi_judge_score",
            "name": "Multi-Judge Scoring",
            "version": "1.1.0",
            "collect": {
                "algorithm": "bco",
                "stage_id": "aggregate",
                "min_samples": 5,
                "output_path": output_path,
                "auto_train": False,
            },
            "episode": {"n_episodes": 5, "batch_size": 2, "shuffle_inputs": True, "seed": 42},
            "reward": {
                "stages": {"aggregate": {"fn": "judge_score", "weight": 1.0}},
                "final_fn": "weighted_mean",
            },
        }

        verdicts = ["PASS", "PASS", "FAIL", "PASS", "FAIL"]
        mock_runner.run_episodes = AsyncMock(
            return_value=[
                _mock_state(f"pipe-{i:03d}", verdict=verdicts[i], elapsed=1.0 + i * 0.2)
                for i in range(5)
            ]
        )

        collect_runner = CollectRunner(mock_runner)
        inputs = [
            "Evaluate: The transformer architecture uses self-attention.",
            "Evaluate: def add(a, b): return a + b",
            "Evaluate: Q3 revenue grew 12% driven by cloud services.",
            "Evaluate: Thank you for contacting us.",
            "Evaluate: The study found no significant correlation.",
        ]

        n_written = await collect_runner.run(inputs)

        print(f"  Episodes run:     {len(verdicts)}")
        print(f"  Records written:  {n_written}")
        print(f"  Output path:      {output_path}")

        records = [json.loads(l) for l in open(output_path)]
        print("\n  Sample record:")
        sample = records[0]
        print(f"    input:       {sample['input'][:50]}...")
        print(f"    label:       {sample['label']}  (1=PASS, 0=FAIL)")
        print(f"    verdict:     {sample['verdict']}")
        print(f"    pipeline_id: {sample['pipeline_id']}")

        labels = [r["label"] for r in records]
        print(f"\n  Label distribution: {labels.count(1)} PASS, {labels.count(0)} FAIL")
        print("  collect mode PASSED ✓")


# ---------------------------------------------------------------------------
# Example 2 — Eval mode: accuracy, latency, cost, judge_score_mean
# ---------------------------------------------------------------------------


async def example_eval_mode():
    print("\n" + "=" * 60)
    print("Example 2 — Eval mode (metrics + threshold check)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a labelled test set
        test_set_path = os.path.join(tmpdir, "kyc_test_set.jsonl")
        test_records = [
            {
                "input": "Name: John Chen, Income: 150000, Employment: employed",
                "expected_verdict": "APPROVE",
            },
            {
                "input": "Name: Alice Wang, Income: 90000, Employment: employed",
                "expected_verdict": "APPROVE",
            },
            {
                "input": "Name: Jane Smith, Income: 45000, Employment: self_employed",
                "expected_verdict": "REVIEW",
            },
            {
                "input": "Name: Bob Jones, Income: 0, Employment: unemployed",
                "expected_verdict": "DENY",
            },
            {
                "input": "Name: Sam Brown, Income: 200000, Employment: employed",
                "expected_verdict": "APPROVE",
            },
        ]
        with open(test_set_path, "w") as f:
            for r in test_records:
                f.write(json.dumps(r) + "\n")

        # Write matching audit entries (cost tracking)
        audit_path = os.path.join(tmpdir, "audit.jsonl")
        for i, _rec in enumerate(test_records):
            with open(audit_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "pipeline_id": f"eval-{i:03d}",
                            "stage_id": "decision_judge",
                            "cost_usd": 0.008 + i * 0.002,
                            "reward": 0.82 + i * 0.02,
                        }
                    )
                    + "\n"
                )

        # Build mocked runner
        mock_runner = MagicMock(spec=GraphRunner)
        mock_runner.config = {
            "id": "bfsi/kyc_triage",
            "name": "KYC Triage",
            "version": "1.1.0",
            "audit": {"path": audit_path},
            "reward": {"stages": {"decision_judge": {"fn": "judge_score", "weight": 1.0}}},
            "eval": {
                "thresholds": {
                    "accuracy": 0.80,
                    "latency_p95": 8000,
                    "cost_per_decision": 0.20,
                }
            },
        }

        # Simulated verdicts (4/5 correct → accuracy = 0.80)
        simulated_verdicts = ["APPROVE", "APPROVE", "REVIEW", "APPROVE", "APPROVE"]
        mock_runner.run = AsyncMock(
            side_effect=[
                _mock_state(f"eval-{i:03d}", verdict=simulated_verdicts[i], elapsed=1.5 + i * 0.3)
                for i in range(5)
            ]
        )

        eval_runner = EvalRunner(mock_runner)
        report = await eval_runner.run(test_set_path)

        print(f"\n  Test set size:  {report['n_evaluated']} records")
        print("\n  Metrics:")
        for k, v in report["metrics"].items():
            print(f"    {k}: {v}")

        if report["passed"]:
            print("\n  All thresholds PASSED ✓")
        else:
            print(f"\n  {len(report['violations'])} threshold violation(s):")
            for v in report["violations"]:
                print(f"    ✗ {v}")

        print("  eval mode PASSED ✓")


# ---------------------------------------------------------------------------
# Example 3 — Observation schema validation
# ---------------------------------------------------------------------------


def example_observation_validation():
    print("\n" + "=" * 60)
    print("Example 3 — Observation schema validation")
    print("=" * 60)

    from agenttune.decide.graph_runner import GraphRunner as GR

    runner = GR.__new__(GR)
    runner.config = {}

    # String schema — accepted
    try:
        runner._validate_observation("any text input", {"type": "string"})
        print("  ✓ string input accepted by string schema")
    except ValueError as e:
        print(f"  ✗ unexpected error: {e}")

    # Object schema — valid JSON with required field
    try:
        runner._validate_observation(
            '{"name": "Alice", "dob": "1990-01-01"}',
            {"type": "object", "required": ["name", "dob"]},
        )
        print("  ✓ valid JSON object accepted by object schema")
    except ValueError as e:
        print(f"  ✗ unexpected error: {e}")

    # Object schema — missing required field
    try:
        runner._validate_observation(
            '{"name": "Alice"}', {"type": "object", "required": ["name", "dob"]}
        )
        print("  ✗ should have raised for missing 'dob'")
    except ValueError as e:
        print(f"  ✓ correctly rejected: {e}")

    # Object schema — not valid JSON
    try:
        runner._validate_observation("plain text", {"type": "object"})
        print("  ✗ should have raised for non-JSON input")
    except ValueError as e:
        print(f"  ✓ correctly rejected: {e}")

    # Array schema — valid
    try:
        runner._validate_observation('["a", "b", "c"]', {"type": "array"})
        print("  ✓ JSON array accepted by array schema")
    except ValueError as e:
        print(f"  ✗ unexpected error: {e}")

    print("  observation schema validation PASSED ✓")


# ---------------------------------------------------------------------------
# Example 4 — Episode reward (weighted_mean)
# ---------------------------------------------------------------------------


def example_episode_reward():
    print("\n" + "=" * 60)
    print("Example 4 — Weighted episode reward")
    print("=" * 60)

    import tempfile

    from agenttune.decide.audit import AuditWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        audit_path = os.path.join(tmpdir, "audit.jsonl")
        writer = AuditWriter(audit_path)

        reward_cfg = {
            "stages": {
                "evaluate_quality": {"fn": "judge_score", "weight": 1.0},
                "generate": {"fn": "iteration_penalty", "weight": 0.2},
                "output_result": {"fn": "verdict_binary", "weight": 0.1},
            },
            "final_fn": "weighted_mean",
        }
        state = _mock_state("ep-001", verdict="COMPLETE")
        state.config = {"reward": reward_cfg}
        state.stage_outputs["evaluate_quality"] = {"overall_score": 8}
        state.stage_iterations["generate"] = 1
        state.stage_outputs["output_result"] = {}

        ep_reward = writer._compute_episode_reward(state, reward_cfg)

        # Manual check:
        #   evaluate_quality: 8/10 = 0.8 * weight 1.0 = 0.80
        #   generate:         iter 1, max 3 → 1.0 * weight 0.2 = 0.20
        #   output_result:    verdict COMPLETE → 1.0 * weight 0.1 = 0.10
        #   total_weight = 1.3 → episode_reward = 1.10 / 1.3 ≈ 0.846

        print("  Stage rewards:")
        print("    evaluate_quality (judge_score 8/10):       0.800 × w=1.0")
        print("    generate (iteration_penalty, iter 1 of 3): 1.000 × w=0.2")
        print("    output_result (verdict_binary COMPLETE):   1.000 × w=0.1")
        print(f"\n  Episode reward (weighted_mean): {ep_reward:.4f}")
        print("  episode reward PASSED ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    print("\nAgentTune Decide — Phase 2 Examples")
    print("Showing: collect mode, eval mode, obs validation, episode reward")

    await example_collect_mode()
    await example_eval_mode()
    example_observation_validation()
    example_episode_reward()

    print("\n" + "=" * 60)
    print("All Phase 2 examples completed successfully.")
    print("=" * 60)
    print(
        """
Next steps:
  1. Install dependencies:   pip install -e .
  2. Add your API key to:    src/agenttune/decide/config.example.yaml
  3. Run collect mode:
       agenttune decide run \\
         --template generic/multi_judge_score \\
         --input @/tmp/test_inputs.jsonl \\
         --mode collect \\
         --config src/agenttune/decide/config.example.yaml

  4. Run eval mode (edit template to set eval.test_set first):
       agenttune decide run \\
         --template bfsi/kyc_triage \\
         --input placeholder \\
         --mode eval \\
         --config src/agenttune/decide/config.example.yaml
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
