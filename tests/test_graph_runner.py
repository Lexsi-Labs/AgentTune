"""Tests for GraphRunner."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from agenttune.decide.graph_runner import GraphRunner


class TestGraphRunner:
    """Test cases for pipeline execution."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self):
        """Test full pipeline execution with simple template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple template
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

            # Save template
            template_dir = os.path.join(tmpdir, "templates", "test")
            os.makedirs(template_dir, exist_ok=True)
            template_path = os.path.join(template_dir, "simple.yaml")

            with open(template_path, "w") as f:
                yaml.dump(template, f)

            # Create config
            config = {"api_keys": {"openai": "test-key"}}
            config_path = os.path.join(tmpdir, "config.yaml")

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Create runner with full template path
            runner = GraphRunner.from_template(template_path, config_path)

            with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
                mock_llm.return_value = {
                    "choices": [{"message": {"content": '{"decision": "approve"}'}}]
                }

                # Run pipeline
                state = await runner.run("test input")

                # Verify execution
                assert state.is_complete is True
                assert state.verdict == "approve"
                assert state.step_count > 0

    @pytest.mark.asyncio
    async def test_loop_back_increments_iteration(self):
        """Test loop-back increments iteration counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create template with loop-back on max iterations
            template = {
                "id": "test/loopback",
                "name": "Loop Back Test",
                "version": "1.0.0",
                "stages": [
                    {
                        "id": "s1",
                        "type": "llm_call",
                        "model": "gpt-4",
                        "prompt": "Extract JSON from {input_text}",
                        "max_iterations": 3,
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
                    {"from": "s1", "to": "s1", "condition": "loopback"},
                ],
            }

            template_dir = os.path.join(tmpdir, "templates", "test")
            os.makedirs(template_dir, exist_ok=True)
            template_path = os.path.join(template_dir, "loopback.yaml")

            with open(template_path, "w") as f:
                yaml.dump(template, f)

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"api_keys": {"openai": "test-key"}}, f)

            runner = GraphRunner.from_template(template_path, config_path)

            with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
                # Return invalid JSON first 2 times, valid on 3rd
                mock_llm.side_effect = [
                    {"choices": [{"message": {"content": "not json"}}]},
                    {"choices": [{"message": {"content": "still not json"}}]},
                    {"choices": [{"message": {"content": '{"decision": "approve"}'}}]},
                ]

                state = await runner.run("test")

                # Verify iteration count
                assert state.stage_iterations.get("s1", 0) >= 1

    @pytest.mark.asyncio
    async def test_conditional_routing(self):
        """Test conditional routing works based on conditions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template = {
                "id": "test/routing",
                "name": "Routing Test",
                "version": "1.0.0",
                "stages": [
                    {
                        "id": "s1",
                        "type": "llm_call",
                        "model": "gpt-4",
                        "prompt": "Score {input_text}",
                        "max_iterations": 1,
                    },
                    {
                        "id": "s2",
                        "type": "router",
                        "on_result": [
                            {
                                "condition": "s1.output.score > 0.7",
                                "goto": "approve",
                            }
                        ],
                        "default": "reject",
                    },
                    {
                        "id": "approve",
                        "type": "output",
                        "verdict_field": '"approve"',
                        "destinations": [],
                    },
                    {
                        "id": "reject",
                        "type": "output",
                        "verdict_field": '"reject"',
                        "destinations": [],
                    },
                ],
            }

            template_dir = os.path.join(tmpdir, "templates", "test")
            os.makedirs(template_dir, exist_ok=True)
            template_path = os.path.join(template_dir, "routing.yaml")

            with open(template_path, "w") as f:
                yaml.dump(template, f)

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"api_keys": {"openai": "test-key"}}, f)

            runner = GraphRunner.from_template(template_path, config_path)

            with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
                mock_llm.return_value = {"choices": [{"message": {"content": '{"score": 0.8}'}}]}

                state = await runner.run("test")
                assert state.verdict == "approve"

    @pytest.mark.asyncio
    async def test_state_passing_and_interpolation(self):
        """Test state passing and interpolation between stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template = {
                "id": "test/interpolation",
                "name": "Interpolation Test",
                "version": "1.0.0",
                "stages": [
                    {
                        "id": "s1",
                        "type": "llm_call",
                        "model": "gpt-4",
                        "prompt": "Process {input_text}",
                        "max_iterations": 1,
                    },
                    {
                        "id": "s2",
                        "type": "llm_call",
                        "model": "gpt-4",
                        "prompt": "Refine {s1.output.value}",
                        "max_iterations": 1,
                    },
                ],
            }

            template_dir = os.path.join(tmpdir, "templates", "test")
            os.makedirs(template_dir, exist_ok=True)
            template_path = os.path.join(template_dir, "interpolation.yaml")

            with open(template_path, "w") as f:
                yaml.dump(template, f)

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"api_keys": {"openai": "test-key"}}, f)

            runner = GraphRunner.from_template(template_path, config_path)

            with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
                mock_llm.side_effect = [
                    {"choices": [{"message": {"content": '{"value": "processed"}'}}]},
                    {"choices": [{"message": {"content": '{"value": "refined"}'}}]},
                ]

                state = await runner.run("input data")

                # Verify state passed between stages
                assert "s1" in state.stage_outputs
                assert "s2" in state.stage_outputs
                assert state.stage_outputs["s1"]["value"] == "processed"
                assert state.stage_outputs["s2"]["value"] == "refined"

    @pytest.mark.asyncio
    async def test_max_total_steps_guard(self):
        """Test max_total_steps guard prevents infinite loops."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create template with self-loop
            template = {
                "id": "test/infinite",
                "name": "Infinite Loop Test",
                "version": "1.0.0",
                "stages": [
                    {
                        "id": "s1",
                        "type": "llm_call",
                        "model": "gpt-4",
                        "prompt": "Loop",
                        "max_iterations": 100,
                    }
                ],
                "edges": [
                    {"from": "s1", "to": "s1", "condition": None},
                ],
            }

            template_dir = os.path.join(tmpdir, "templates", "test")
            os.makedirs(template_dir, exist_ok=True)
            template_path = os.path.join(template_dir, "infinite.yaml")

            with open(template_path, "w") as f:
                yaml.dump(template, f)

            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"api_keys": {"openai": "test-key"}}, f)

            runner = GraphRunner.from_template(template_path, config_path)

            with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
                mock_llm.return_value = {"choices": [{"message": {"content": "{}"}}]}

                state = await runner.run("test")

                # Should hit max_total_steps guard
                max_steps = runner.config.get("max_total_steps", 50)
                assert state.step_count <= max_steps
                assert state.error is not None or state.step_count == max_steps
