"""Tests for audit logging and reading."""

import json
import os
import tempfile
from datetime import datetime

from agenttune.decide.audit import AuditReader, AuditWriter
from agenttune.decide.state import PipelineState


class TestAuditWriter:
    """Test cases for audit writing."""

    def test_log_stage_entry(self):
        """Test logging a stage entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = os.path.join(tmpdir, "audit.jsonl")
            writer = AuditWriter(audit_path)

            # Create test state
            state = PipelineState(
                pipeline_id="test-1",
                template_id="bfsi/kyc",
                template_version="1.0.0",
                input_text="test input",
                input_hash="abc123",
                stage_outputs={},
                stage_iterations={},
            )

            # Create stage and result
            stage = {"id": "s1", "type": "llm_call", "prompt": "test prompt", "model": "gpt-4"}
            result = {"output": "test output", "latency_ms": 100, "cost_usd": 0.01}

            # Log entry
            writer.log_stage(state, stage, result)

            # Verify file was written
            assert os.path.exists(audit_path)

            # Verify content
            with open(audit_path) as f:
                line = f.read().strip()
                entry = json.loads(line)
                assert entry["pipeline_id"] == "test-1"
                assert entry["stage_id"] == "s1"
                assert entry["output"] == "test output"
                assert entry["latency_ms"] == 100

    def test_write_final_entry(self):
        """Test writing final completion entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = os.path.join(tmpdir, "audit.jsonl")
            writer = AuditWriter(audit_path)

            # Create final state
            state = PipelineState(
                pipeline_id="test-2",
                template_id="bfsi/kyc",
                template_version="1.0.0",
                input_text="test input",
                input_hash="xyz789",
                stage_outputs={},
                stage_iterations={},
                verdict="approve",
                verdict_label="approved",
                is_complete=True,
                step_count=5,
                elapsed_seconds=2.5,
                timestamp_end=datetime.utcnow().isoformat(),
            )

            # Write final entry
            writer.write(state)

            # Verify content
            with open(audit_path) as f:
                line = f.read().strip()
                entry = json.loads(line)
                assert entry["pipeline_id"] == "test-2"
                assert entry["verdict"] == "approve"
                assert entry["is_complete"] is True
                assert entry["step_count"] == 5


class TestAuditReader:
    """Test cases for audit reading."""

    def test_extract_dpo_pairs(self):
        """Test extracting DPO pairs from audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = os.path.join(tmpdir, "audit.jsonl")

            # Create audit file with DPO pairs
            entries = [
                {
                    "stage_id": "s1",
                    "human_feedback": "rejected",
                    "input": "test input",
                    "model_output": "rejected output",
                    "human_output": "correct output",
                    "human_explanation": "wrong reasoning",
                },
                {
                    "stage_id": "s1",
                    "human_feedback": "approved",
                    "input": "test input 2",
                    "model_output": "good output",
                },
                {
                    "stage_id": "s2",
                    "human_feedback": "rejected",
                    "input": "other input",
                    "model_output": "other rejected",
                    "human_output": "other correct",
                },
            ]

            with open(audit_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            # Extract DPO pairs for s1
            reader = AuditReader(audit_path)
            pairs = reader.extract_dpo_pairs("s1")

            # Verify extraction
            assert len(pairs) == 1
            assert pairs[0]["input"] == "test input"
            assert pairs[0]["rejected_output"] == "rejected output"
            assert pairs[0]["chosen_output"] == "correct output"
            assert pairs[0]["reason"] == "wrong reasoning"

    def test_extract_dpo_pairs_empty_file(self):
        """Test extracting DPO pairs from non-existent file."""
        reader = AuditReader("/nonexistent/path/audit.jsonl")
        pairs = reader.extract_dpo_pairs("s1")
        assert pairs == []
