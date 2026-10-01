"""Tests for the public SDK facade: agenttune.api."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml
from _real_backends import real_generate

# Loads Qwen2.5-0.5B via _real_backends (3-5GB RSS + ~1GB download);
# not for the 7.8GB CPU CI runner. Runs under -m qwen_e2e.
pytestmark = pytest.mark.qwen_e2e

_SYSTEM_PROMPT = (
    "You are a JSON-only API for content moderation decisions. You ONLY ever output "
    'a single JSON object shaped exactly like {"decision": "approve"} or '
    '{"decision": "reject"}. Never output any other words, explanation, or punctuation.\n\n'
    'Example\nInput: "Hello, how is your day?"\nOutput: {"decision": "approve"}'
)


async def _real_acompletion(*args, **kwargs):
    """Drop-in for litellm.acompletion backed by Qwen2.5-0.5B-Instruct instead
    of a canned mock response — see tests/agentic_real/_real_backends.py.
    """
    messages = kwargs.get("messages") or (args[0] if args else [])
    prompt = messages[-1]["content"]
    text = real_generate(prompt, system=_SYSTEM_PROMPT, max_new_tokens=20)
    return {"choices": [{"message": {"content": text}}]}


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

    # REAL model call: Qwen/Qwen2.5-0.5B-Instruct generates the decision
    # instead of a canned mock dict (see tests/agentic_real/_real_backends.py).
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path, config_path = _write_simple_template(tmpdir)
        with patch(
            "agenttune.decide.stages.base.litellm.acompletion", side_effect=_real_acompletion
        ):
            result = run_pipeline(template_path, "test input", config=config_path)

    # The exact business decision (approve vs. reject) is the real model's own
    # greedy-decoded judgment call on a deliberately generic prompt ("test
    # input") — not something this test should pin down. What this test
    # actually verifies is that a real model call flows correctly through
    # run_pipeline()'s template -> stage -> output-extraction wiring end to
    # end, so assert the pipeline produced one of the two valid decisions
    # rather than a specific one.
    assert isinstance(result, PipelineResult)
    assert result.verdict in ("approve", "reject")
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
