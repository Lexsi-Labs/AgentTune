"""Tests for the top-level `agenttune` CLI (cli/unified.py)."""

import os
import tempfile
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from agenttune.cli.unified import app

runner = CliRunner()


def _write_simple_template(tmpdir):
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
    tdir = os.path.join(tmpdir, "templates", "test")
    os.makedirs(tdir, exist_ok=True)
    tpath = os.path.join(tdir, "simple.yaml")
    with open(tpath, "w") as f:
        yaml.dump(template, f)
    cpath = os.path.join(tmpdir, "config.yaml")
    with open(cpath, "w") as f:
        yaml.dump({"api_keys": {"openai": "test-key"}}, f)
    return tpath, cpath


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    import agenttune

    assert agenttune.__version__ in result.stdout


def test_pipeline_command_runs_and_writes_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        tpath, cpath = _write_simple_template(tmpdir)
        out = os.path.join(tmpdir, "decision.json")
        with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
            mock_llm.return_value = {
                "choices": [{"message": {"content": '{"decision": "approve"}'}}]
            }
            result = runner.invoke(
                app,
                [
                    "pipeline",
                    "--template",
                    tpath,
                    "--input",
                    "hello",
                    "--config",
                    cpath,
                    "--output",
                    out,
                ],
            )
        assert result.exit_code == 0, result.stdout
        assert "approve" in result.stdout
        assert os.path.exists(out)


def test_train_command_unknown_algorithm_errors_cleanly():
    result = runner.invoke(app, ["train", "--algorithm", "bogus", "--model", "x", "--dataset", "y"])
    assert result.exit_code != 0
    assert "Unknown algorithm" in result.stdout
