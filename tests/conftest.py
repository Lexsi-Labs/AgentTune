"""
Shared pytest fixtures for the whole test suite (flat under tests/).

Fixtures here are available to every test in tests/ without import.
"""

import json
import os
import sys

import pytest

# Keep the default test run on the TRL backend only. Unsloth monkey-patches
# transformers' model classes (Qwen2Attention, Qwen2RotaryEmbedding, ...) at
# the class level the moment it's imported anywhere in the process -- once
# that happens, every later test that loads a plain (non-Unsloth) model in
# the SAME pytest process breaks with AttributeErrors like
# "'Qwen2RotaryEmbedding' object has no attribute 'extend_rope_embedding'".
# Unsloth backends are exercised separately, in their own isolated process.
# agenttune.backends._imports only checks PURE_TRL_MODE now; TRL_ONLY_MODE and
# DISABLE_UNSLOTH_FOR_TRL were older names for the same flag, collapsed into
# PURE_TRL_MODE. Set all three anyway so this keeps working against an older
# checkout of that module too.
for _var in ("TRL_ONLY_MODE", "DISABLE_UNSLOTH_FOR_TRL", "PURE_TRL_MODE"):
    os.environ.setdefault(_var, "1")
# hf_xet was observed to crawl (~30MB/13min) on this network; plain HTTPS is
# far faster here, so prefer it for every HF download during tests.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# test_masking_verification.py defaults to Qwen2.5-1.5B-Instruct; point it at
# the small model already cached for the rest of the suite instead.
os.environ.setdefault("RAG_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

sys.path.insert(0, os.path.dirname(__file__))

from agenttune.decide.state import PipelineState

# ---------------------------------------------------------------------------
# Directory / path helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory cleaned up after each test."""
    return tmp_path


# ---------------------------------------------------------------------------
# PipelineState factory
# ---------------------------------------------------------------------------


def make_state(**overrides) -> PipelineState:
    """Return a minimal PipelineState with sane defaults."""
    defaults = {
        "pipeline_id": "test-pipeline-001",
        "template_id": "test/template",
        "template_version": "1.0.0",
        "input_text": "Test customer document with income=50000 dob=1990-01-01",
        "input_hash": "deadbeef",
        "stage_outputs": {},
        "stage_iterations": {},
        "stage_traces": [],
        "step_count": 0,
        "step_history": [],
        "verdict": None,
        "verdict_label": None,
        "confidence": None,
        "reason": None,
        "is_complete": False,
        "error": None,
        "error_stage": None,
        "timestamp_start": "2026-04-27T10:00:00Z",
        "timestamp_end": None,
        "elapsed_seconds": 0.0,
        "config": {"default_model": "claude-haiku-4-5", "max_total_steps": 50},
    }
    defaults.update(overrides)
    return PipelineState(**defaults)


@pytest.fixture
def sample_state():
    return make_state()


# ---------------------------------------------------------------------------
# Audit log fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_audit_entries():
    return [
        {
            "timestamp": "2026-04-27T10:00:00Z",
            "pipeline_id": "pipe-001",
            "template_id": "bfsi/kyc_triage",
            "template_version": "1.0.0",
            "input_hash": "abc123",
            "stage_id": "extract",
            "stage_type": "llm_call",
            "iteration": 1,
            "input": "Extract from customer doc",
            "output": {"full_name": "Alice", "income": 50000, "dob": "1990-01-01"},
            "latency_ms": 350,
            "cost_usd": 0.001,
            "model": "claude-haiku-4-5",
            "is_retry": False,
            "error": None,
        },
        {
            "timestamp": "2026-04-27T10:00:01Z",
            "pipeline_id": "pipe-001",
            "template_id": "bfsi/kyc_triage",
            "template_version": "1.0.0",
            "input_hash": "abc123",
            "stage_id": "decision_judge",
            "stage_type": "llm_judge",
            "iteration": 1,
            "input": "Rate customer risk",
            "output": {"score": 8, "decision": "APPROVE", "explanation": "Good profile"},
            "latency_ms": 600,
            "cost_usd": 0.005,
            "model": "claude-opus-4-1",
            "is_retry": False,
            "error": None,
        },
        {
            "timestamp": "2026-04-27T10:00:02Z",
            "pipeline_id": "pipe-001",
            "template_id": "bfsi/kyc_triage",
            "stage_id": "decision_judge",
            "stage_type": "llm_judge",
            "human_feedback": "rejected",
            "model_output": '{"decision": "APPROVE", "score": 7}',
            "human_output": '{"decision": "DENY", "score": 3}',
            "human_explanation": "Customer has undisclosed fraud flag",
            "input": "Rate customer risk",
        },
        {
            "timestamp": "2026-04-27T10:00:03Z",
            "pipeline_id": "pipe-001",
            "template_id": "bfsi/kyc_triage",
            "stage_id": "approve",
            "stage_type": "output",
            "verdict": "APPROVE",
            "verdict_label": "kyc_approved",
            "input": "Final approval",
            "output": {"verdict": "APPROVE"},
        },
    ]


@pytest.fixture
def audit_file(tmp_dir, sample_audit_entries):
    path = tmp_dir / "audit.jsonl"
    with open(path, "w") as f:
        for entry in sample_audit_entries:
            f.write(json.dumps(entry) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Mock LLM response helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_response():
    """Return a fixed JSON string as if from an LLM."""
    return '{"summary": "A concise summary of the input.", "confidence": 0.9}'
