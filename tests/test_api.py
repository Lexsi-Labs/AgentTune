"""Tests for the public SDK facade: agenttune.api."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml


def _write_simple_template(tmpdir):
    """A minimal llm_call -> output template + config; returns (template_path, config_path)."""
    template = {
        "id": "test/simple",
        "name": "Simple Test",
        "version": "1.0.0",
        "stages": [
            {
                "id": "s1",
                "type": "llm_call",
                "model": "gpt-4",
                "prompt": "Test {input_text}",
                "max_iterations": 1,
            },
            {
                "id": "s2",
                "type": "output",
                "verdict_field": "s1.output.decision",
                "destinations": [],
            },
        ],
        "edges": [
            {"from": "s1", "to": "s2", "condition": None},
            {"from": "s2", "to": "__end__", "condition": None},
        ],
    }
    template_dir = os.path.join(tmpdir, "templates", "test")
    os.makedirs(template_dir, exist_ok=True)
    template_path = os.path.join(template_dir, "simple.yaml")
    with open(template_path, "w") as f:
        yaml.dump(template, f)

    config_path = os.path.join(tmpdir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({"api_keys": {"openai": "test-key"}}, f)
    return template_path, config_path


def test_api_exports():
    import agenttune.api as api

    for name in ("run_pipeline", "arun_pipeline", "train_agentic", "PipelineResult"):
        assert hasattr(api, name), f"agenttune.api is missing {name}"


def test_run_pipeline_inference():
    from agenttune.api import PipelineResult, run_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        template_path, config_path = _write_simple_template(tmpdir)
        with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
            mock_llm.return_value = {
                "choices": [{"message": {"content": '{"decision": "approve"}'}}]
            }
            result = run_pipeline(template_path, "test input", config=config_path)

    assert isinstance(result, PipelineResult)
    assert result.verdict == "approve"
    assert result.is_complete is True
    assert result.step_count > 0
    assert result.error is None


def test_pipeline_result_to_dict():
    from agenttune.api import PipelineResult

    r = PipelineResult(
        pipeline_id="p",
        template_id="t",
        verdict="approve",
        verdict_label="ok",
        confidence=0.9,
        reason="because",
        step_count=2,
        elapsed_seconds=0.1,
        is_complete=True,
        error=None,
    )
    d = r.to_dict()
    assert d["verdict"] == "approve"
    assert set(d) >= {"verdict", "template_id", "is_complete", "error"}


def test_train_agentic_rejects_unknown_algorithm():
    from agenttune.api import train_agentic

    with pytest.raises(ValueError) as exc:
        train_agentic("not_an_algo", model="x")
    # error should name the valid algorithms
    assert "grpo" in str(exc.value).lower()
