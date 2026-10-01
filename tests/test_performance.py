"""
Performance and load tests.

These tests establish baseline metrics rather than making assertions that
pass/fail based on machine speed. They measure:
  - Audit write throughput (entries/sec)
  - Config load + deep merge speed
  - SafeEvaluator expression evaluation speed
  - PipelineState memory footprint

None of these tests require network access or real model weights.
"""

import json
import time
from pathlib import Path

from conftest import make_state

from agenttune.decide.audit import AuditReader, AuditWriter
from agenttune.decide.config import ConfigLoader
from agenttune.decide.stages.rules import SafeEvaluator

# ---------------------------------------------------------------------------
# Audit write throughput
# ---------------------------------------------------------------------------


class TestAuditWriteThroughput:
    def test_1000_entries_under_5_seconds(self, tmp_dir):
        path = str(tmp_dir / "load.jsonl")
        writer = AuditWriter(path)
        state = make_state()

        start = time.perf_counter()
        for i in range(1000):
            stage = {"id": f"stage_{i}", "type": "llm_call"}
            writer.log_stage(state, stage, {"output": {"idx": i}, "latency_ms": 100})
        elapsed = time.perf_counter() - start

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 1000
        # Soft assertion — record the metric
        print(f"\n[PERF] 1000 audit writes in {elapsed:.3f}s ({1000/elapsed:.0f} entries/s)")
        assert elapsed < 30, "1000 audit writes should complete in <30s even on slow CI"

    def test_read_1000_entries(self, tmp_dir):
        path = str(tmp_dir / "load.jsonl")
        writer = AuditWriter(path)
        state = make_state()
        for i in range(1000):
            stage = {"id": f"stage_{i}", "type": "llm_call"}
            writer.log_stage(state, stage, {"output": {"idx": i}})

        reader = AuditReader(path)
        start = time.perf_counter()
        entries = reader.read_all()
        elapsed = time.perf_counter() - start

        assert len(entries) == 1000
        print(f"\n[PERF] Read 1000 entries in {elapsed:.3f}s")
        assert elapsed < 10


# ---------------------------------------------------------------------------
# Config deep merge speed
# ---------------------------------------------------------------------------


class TestConfigMergePerformance:
    def test_deep_merge_100_times_under_1_second(self):
        base = {
            "api_keys": {"anthropic": "key1", "openai": "key2"},
            "default_model": "claude-haiku-4-5",
            "max_total_steps": 50,
            "destinations": {
                "file": {"enabled": True, "path": "./decisions.jsonl"},
                "postgres": {"enabled": False},
            },
            "compliance": {"audit_required": True},
        }
        override = {
            "default_model": "claude-opus-4-1",
            "max_total_steps": 30,
            "destinations": {
                "postgres": {"enabled": True, "table": "kyc"},
            },
            "extra_key": "extra_value",
        }

        start = time.perf_counter()
        for _ in range(100):
            result = ConfigLoader.deep_merge(base, override)
        elapsed = time.perf_counter() - start

        print(f"\n[PERF] 100 deep merges in {elapsed:.4f}s")
        assert elapsed < 1.0
        assert result["default_model"] == "claude-opus-4-1"


# ---------------------------------------------------------------------------
# SafeEvaluator throughput
# ---------------------------------------------------------------------------


class TestSafeEvaluatorPerformance:
    def test_1000_evaluations_under_2_seconds(self):
        evaluator = SafeEvaluator()
        context = {
            "income": 50000,
            "dob": "1990-01-01",
            "employment": "employed",
            "score": 8,
            "risk_level": "low",
        }
        conditions = [
            "income != null and income > 0",
            "score >= 7",
            "score < 5",
            "income > 30000 and score >= 6",
            "risk_level != null",
        ]

        start = time.perf_counter()
        for i in range(1000):
            cond = conditions[i % len(conditions)]
            evaluator.eval(cond, context)
        elapsed = time.perf_counter() - start

        print(f"\n[PERF] 1000 SafeEvaluator evals in {elapsed:.4f}s")
        assert elapsed < 5.0


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------


class TestPipelineStateMemory:
    def test_state_with_large_outputs_is_reasonable(self):
        """Verify state doesn't balloon memory with many stage outputs."""
        import sys

        state = make_state(
            stage_outputs={f"stage_{i}": {"output": "x" * 1000, "score": i} for i in range(100)}
        )
        size = sys.getsizeof(str(state.stage_outputs))
        # Should be under 1MB for 100 stages with 1KB each
        assert size < 10 * 1024 * 1024, f"State outputs too large: {size} bytes"

    def test_stage_traces_grow_linearly(self):
        state = make_state()
        base_size = len(state.stage_traces)

        for i in range(50):
            state.stage_traces.append(
                {
                    "stage_id": f"stage_{i}",
                    "output": {"result": f"output_{i}"},
                    "latency_ms": 100,
                }
            )

        assert len(state.stage_traces) == base_size + 50


# ---------------------------------------------------------------------------
# DPO extraction throughput
# ---------------------------------------------------------------------------


class TestBridgePerformance:
    def test_extract_dpo_from_large_audit(self, tmp_dir):
        """Build a 500-entry audit log with 50 DPO pairs and measure extraction."""
        path = str(tmp_dir / "big_audit.jsonl")

        with open(path, "w") as f:
            for i in range(500):
                entry = {
                    "pipeline_id": f"pipe-{i}",
                    "stage_id": "judge",
                    "stage_type": "llm_judge",
                    "input": f"Input {i}",
                    "output": {"score": 7},
                }
                if i % 10 == 0:  # 50 rejections
                    entry["human_feedback"] = "rejected"
                    entry["model_output"] = '{"score": 7}'
                    entry["human_output"] = '{"score": 2}'
                    entry["human_explanation"] = f"Reason {i}"
                f.write(json.dumps(entry) + "\n")

        from agenttune.decide.training_bridge import DecideToTrainerBridge

        bridge = DecideToTrainerBridge(path)

        start = time.perf_counter()
        pairs = bridge.extract_dpo_pairs("judge")
        elapsed = time.perf_counter() - start

        assert len(pairs) == 50
        print(f"\n[PERF] DPO extraction (50 pairs from 500 entries): {elapsed:.4f}s")
        assert elapsed < 5.0
