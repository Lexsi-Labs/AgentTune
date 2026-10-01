"""Tests for stage handlers."""

from unittest.mock import patch

import pytest

from agenttune.decide.stages.llm_call import LLMCallStage
from agenttune.decide.stages.output import OutputStage
from agenttune.decide.stages.parallel import ParallelStage
from agenttune.decide.stages.router import RouterStage
from agenttune.decide.stages.rules import RulesStage
from agenttune.decide.state import PipelineState


class TestLLMCallStage:
    """Test cases for LLMCallStage."""

    @pytest.mark.asyncio
    async def test_llm_call_with_mocked_model(self):
        """Test LLM call with mocked model."""
        stage_config = {
            "id": "s1",
            "type": "llm_call",
            "model": "gpt-4",
            "prompt": "Decide if {input_text} is risky",
            "max_iterations": 1,
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="high risk user",
            input_hash="abc123",
            stage_outputs={},
            stage_iterations={},
        )

        handler = LLMCallStage(stage_config)

        with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
            mock_llm.return_value = {
                "choices": [{"message": {"content": '{"risk": "high", "score": 0.95}'}}]
            }

            result = await handler.execute(state)

            assert result["output"]["risk"] == "high"
            assert result["output"]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_prompt_interpolation(self):
        """Test prompt interpolation with context."""
        stage_config = {
            "id": "s1",
            "type": "llm_call",
            "model": "gpt-4",
            "prompt": "Evaluate: {input_text} with age {s0.output.age}",
            "max_iterations": 1,
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="John Doe",
            input_hash="abc123",
            stage_outputs={"s0": {"age": 25}},
            stage_iterations={},
        )

        handler = LLMCallStage(stage_config)

        with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
            mock_llm.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}

            await handler.execute(state)

            # Verify interpolation happened in the call
            called_prompt = mock_llm.call_args[1]["messages"][0]["content"]
            assert "John Doe" in called_prompt
            assert "25" in called_prompt

    @pytest.mark.asyncio
    async def test_json_schema_validation(self):
        """Test JSON schema validation."""
        stage_config = {
            "id": "s1",
            "type": "llm_call",
            "model": "gpt-4",
            "prompt": "Return JSON",
            "max_iterations": 2,
            "json_schema": {
                "type": "object",
                "properties": {"decision": {"type": "string"}},
                "required": ["decision"],
            },
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="test",
            input_hash="abc123",
            stage_outputs={},
            stage_iterations={},
        )

        handler = LLMCallStage(stage_config)

        # First call returns invalid JSON, second returns valid
        with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
            mock_llm.side_effect = [
                {"choices": [{"message": {"content": '```json\n{"bad": "data"}\n```'}}]},
                {"choices": [{"message": {"content": '```json\n{"decision": "approve"}\n```'}}]},
            ]

            result = await handler.execute(state)
            assert result["output"]["decision"] == "approve"


class TestRulesStage:
    """Test cases for RulesStage."""

    @pytest.mark.asyncio
    async def test_rules_with_known_context(self):
        """Test rules evaluation with known context."""
        stage_config = {
            "id": "s1",
            "type": "rules",
            "rules": [
                {
                    "condition": "s0.output.score > 0.5",
                    "on_failure": "reject",
                }
            ],
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="test",
            input_hash="abc123",
            stage_outputs={"s0": {"score": 0.7}},
            stage_iterations={},
        )

        handler = RulesStage(stage_config)
        result = await handler.execute(state)

        # Rule should pass (0.7 > 0.5)
        assert result["goto"] is None

    @pytest.mark.asyncio
    async def test_rules_failure_routing(self):
        """Test rules failure triggers routing."""
        stage_config = {
            "id": "s1",
            "type": "rules",
            "rules": [
                {
                    "condition": "s0.output.score > 0.9",
                    "on_failure": "reject_stage",
                }
            ],
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="test",
            input_hash="abc123",
            stage_outputs={"s0": {"score": 0.5}},
            stage_iterations={},
        )

        handler = RulesStage(stage_config)
        result = await handler.execute(state)

        # Rule should fail and route to reject_stage
        assert result["goto"] == "reject_stage"


class TestParallelStage:
    """Test cases for ParallelStage."""

    @pytest.mark.asyncio
    async def test_parallel_with_async_gather(self):
        """Test parallel execution with asyncio.gather."""
        stage_config = {
            "id": "s1",
            "type": "parallel",
            "branches": [
                {
                    "id": "branch_a",
                    "stages": [
                        {
                            "id": "s1a",
                            "type": "llm_call",
                            "model": "gpt-4",
                            "prompt": "Branch A",
                        }
                    ],
                },
                {
                    "id": "branch_b",
                    "stages": [
                        {
                            "id": "s1b",
                            "type": "llm_call",
                            "model": "gpt-4",
                            "prompt": "Branch B",
                        }
                    ],
                },
            ],
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="test",
            input_hash="abc123",
            stage_outputs={},
            stage_iterations={},
        )

        handler = ParallelStage(stage_config)

        with patch("agenttune.decide.stages.base.litellm.acompletion") as mock_llm:
            mock_llm.return_value = {"choices": [{"message": {"content": '{"result": true}'}}]}

            result = await handler.execute(state)

            # Both branches should have executed
            assert "output" in result
            assert mock_llm.call_count >= 2


class TestRouterStage:
    """Test cases for RouterStage."""

    @pytest.mark.asyncio
    async def test_router_conditional_logic(self):
        """Test router conditional logic."""
        stage_config = {
            "id": "s1",
            "type": "router",
            "on_result": [
                {
                    "condition": "s0.output.score > 0.8",
                    "goto": "approve",
                },
                {
                    "condition": "s0.output.score > 0.5",
                    "goto": "review",
                },
            ],
            "default": "reject",
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="test",
            input_hash="abc123",
            stage_outputs={"s0": {"score": 0.7}},
            stage_iterations={},
        )

        handler = RouterStage(stage_config)
        result = await handler.execute(state)

        # Should match second condition (0.7 > 0.5)
        assert result["goto"] == "review"

    @pytest.mark.asyncio
    async def test_router_default_routing(self):
        """Test router uses default when no conditions match."""
        stage_config = {
            "id": "s1",
            "type": "router",
            "on_result": [
                {
                    "condition": "s0.output.score > 0.9",
                    "goto": "approve",
                }
            ],
            "default": "reject",
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="test",
            input_hash="abc123",
            stage_outputs={"s0": {"score": 0.3}},
            stage_iterations={},
        )

        handler = RouterStage(stage_config)
        result = await handler.execute(state)

        # Should use default
        assert result["goto"] == "reject"


class TestOutputStage:
    """Test cases for OutputStage."""

    @pytest.mark.asyncio
    async def test_output_stage_verdict_setting(self):
        """Test output stage sets verdict correctly."""
        stage_config = {
            "id": "output",
            "type": "output",
            "verdict_field": "s2.output.decision",
            "confidence_field": "s2.output.confidence",
            "destinations": [],
        }

        state = PipelineState(
            pipeline_id="test-1",
            template_id="test",
            template_version="1.0.0",
            input_text="test",
            input_hash="abc123",
            stage_outputs={"s2": {"decision": "approve", "confidence": 0.95}},
            stage_iterations={},
        )

        handler = OutputStage(stage_config)
        await handler.execute(state)

        # Verify verdict was set in state
        assert state.verdict == "approve"
        assert state.confidence == 0.95
        assert state.is_complete is True
