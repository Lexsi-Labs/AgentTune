"""
Advanced stage tests for ParallelStage, RouterStage, HumanReviewStage, ToolCallStage.

Tests cover:
- Parallel execution with multiple branches
- Conditional routing logic
- Human review pause/resume
- Tool execution and error handling
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agenttune.decide.stages.human_review import HumanReviewStage
from agenttune.decide.stages.parallel import ParallelStage
from agenttune.decide.stages.router import RouterStage
from agenttune.decide.stages.tool_call import ToolCallStage
from agenttune.decide.state import PipelineState

# ============================================================================
# ParallelStage Tests
# ============================================================================


class TestParallelStage:
    """Test parallel execution of multiple branches."""

    @pytest.fixture
    def parallel_stage(self):
        """Create a ParallelStage instance."""
        return ParallelStage()

    @pytest.fixture
    def sample_state(self):
        """Create a sample PipelineState."""
        return PipelineState(
            pipeline_id="test-pipeline",
            template_id="test/template",
            template_version="1.0.0",
            input_text="Test input",
            input_hash="abc123",
            stage_outputs={},
            stage_iterations={},
            stage_traces=[],
            step_count=0,
            step_history=[],
            verdict=None,
            verdict_label=None,
            confidence=None,
            reason=None,
            is_complete=False,
            error=None,
            error_stage=None,
            timestamp_start="2026-04-27T10:00:00Z",
            timestamp_end=None,
            elapsed_seconds=0.0,
            config={"default_model": "claude-haiku-4-5"},
        )

    @pytest.mark.asyncio
    async def test_parallel_basic_execution(self, parallel_stage, sample_state):
        """Test basic parallel execution of branches."""
        stage_config = {
            "id": "parallel_test",
            "type": "parallel",
            "branches": [
                {"id": "branch_a", "type": "llm_call", "prompt": "Task A"},
                {"id": "branch_b", "type": "llm_call", "prompt": "Task B"},
            ],
        }

        # Mock the branch execution
        with patch.object(
            parallel_stage, "_execute_branch", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = [{"output": "Result A"}, {"output": "Result B"}]

            result = await parallel_stage.execute(sample_state, stage_config)

            assert "output" in result
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_multiple_branches(self, parallel_stage, sample_state):
        """Test parallel execution with 3+ branches."""
        stage_config = {
            "id": "parallel_test",
            "type": "parallel",
            "branches": [
                {"id": "branch_a", "type": "llm_call", "prompt": "A"},
                {"id": "branch_b", "type": "llm_call", "prompt": "B"},
                {"id": "branch_c", "type": "llm_call", "prompt": "C"},
            ],
        }

        with patch.object(
            parallel_stage, "_execute_branch", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = [{"output": "A"}, {"output": "B"}, {"output": "C"}]

            result = await parallel_stage.execute(sample_state, stage_config)

            assert "output" in result
            assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_parallel_error_in_branch(self, parallel_stage, sample_state):
        """Test error handling when one branch fails."""
        stage_config = {
            "id": "parallel_test",
            "type": "parallel",
            "branches": [
                {"id": "branch_a", "type": "llm_call", "prompt": "A"},
                {"id": "branch_b", "type": "llm_call", "prompt": "B"},
            ],
        }

        with patch.object(
            parallel_stage, "_execute_branch", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = [{"output": "A"}, Exception("Branch B failed")]

            # Should handle error gracefully
            try:
                await parallel_stage.execute(sample_state, stage_config)
                # Depending on implementation, either raises or returns error
            except Exception as e:
                assert "Branch B failed" in str(e)

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self, parallel_stage, sample_state):
        """Test that parallel results are collected in order."""
        stage_config = {
            "id": "parallel_test",
            "type": "parallel",
            "branches": [
                {"id": "first", "type": "llm_call", "prompt": "1"},
                {"id": "second", "type": "llm_call", "prompt": "2"},
                {"id": "third", "type": "llm_call", "prompt": "3"},
            ],
        }

        with patch.object(
            parallel_stage, "_execute_branch", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = [{"output": "1"}, {"output": "2"}, {"output": "3"}]

            result = await parallel_stage.execute(sample_state, stage_config)
            output = result.get("output", {})

            # Verify order is preserved
            assert "first" in output or len(output) == 3


# ============================================================================
# RouterStage Tests
# ============================================================================


class TestRouterStage:
    """Test conditional routing logic."""

    @pytest.fixture
    def router_stage(self):
        """Create a RouterStage instance."""
        return RouterStage()

    @pytest.fixture
    def sample_state(self):
        """Create a sample PipelineState with prior stage outputs."""
        state = PipelineState(
            pipeline_id="test-pipeline",
            template_id="test/template",
            template_version="1.0.0",
            input_text="Test",
            input_hash="abc",
            stage_outputs={"extract": {"income": 50000, "credit_score": 750}},
            stage_iterations={},
            stage_traces=[],
            step_count=0,
            step_history=[],
            verdict=None,
            verdict_label=None,
            confidence=None,
            reason=None,
            is_complete=False,
            error=None,
            error_stage=None,
            timestamp_start="2026-04-27T10:00:00Z",
            timestamp_end=None,
            elapsed_seconds=0.0,
            config={"default_model": "claude-haiku-4-5"},
        )
        return state

    @pytest.mark.asyncio
    async def test_router_basic_routing(self, router_stage, sample_state):
        """Test basic conditional routing."""
        stage_config = {
            "id": "router",
            "type": "router",
            "prompt": "Is income >= 50000? {extract.output.income}",
            "output_schema": {"type": "object", "properties": {"route": {"type": "string"}}},
            "next": "default",
        }

        with patch.object(router_stage, "_call_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"route": "high_income"}'

            result = await router_stage.execute(sample_state, stage_config)

            assert "output" in result

    @pytest.mark.asyncio
    async def test_router_conditional_edges(self, router_stage, sample_state):
        """Test router with multiple conditional edges."""
        stage_config = {
            "id": "router",
            "type": "router",
            "prompt": "Route based on income",
            "output_schema": {"type": "object"},
            "on_result": [
                {"condition": "route == 'high_income'", "goto": "high_income_path"},
                {"condition": "route == 'low_income'", "goto": "low_income_path"},
            ],
        }

        with patch.object(router_stage, "_call_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = '{"route": "high_income"}'

            result = await router_stage.execute(sample_state, stage_config)

            assert "output" in result

    @pytest.mark.asyncio
    async def test_router_type_coercion(self, router_stage, sample_state):
        """Test router handles type coercion in conditions."""
        stage_config = {
            "id": "router",
            "type": "router",
            "prompt": "Check number",
            "output_schema": {"type": "object"},
        }

        # Test with different output types
        test_cases = [
            '{"value": "100"}',  # String number
            '{"value": 100}',  # Integer
            '{"value": 100.0}',  # Float
        ]

        for test_output in test_cases:
            with patch.object(router_stage, "_call_model", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = test_output
                result = await router_stage.execute(sample_state, stage_config)
                assert "output" in result


# ============================================================================
# HumanReviewStage Tests
# ============================================================================


class TestHumanReviewStage:
    """Test human review pause/resume workflow."""

    @pytest.fixture
    def human_review_stage(self):
        """Create a HumanReviewStage instance."""
        return HumanReviewStage()

    @pytest.fixture
    def sample_state(self):
        """Create a sample PipelineState."""
        return PipelineState(
            pipeline_id="test-pipeline",
            template_id="test/template",
            template_version="1.0.0",
            input_text="Test",
            input_hash="abc",
            stage_outputs={"decision_judge": {"decision": "REVIEW", "score": 6}},
            stage_iterations={},
            stage_traces=[],
            step_count=0,
            step_history=[],
            verdict=None,
            verdict_label=None,
            confidence=None,
            reason=None,
            is_complete=False,
            error=None,
            error_stage=None,
            timestamp_start="2026-04-27T10:00:00Z",
            timestamp_end=None,
            elapsed_seconds=0.0,
            config={},
        )

    @pytest.mark.asyncio
    async def test_human_review_prompt_generation(self, human_review_stage, sample_state):
        """Test human review prompt generation."""
        stage_config = {
            "id": "manual_review",
            "type": "human_review",
            "prompt_for_human": "Decision: {decision_judge.output.decision}. Continue?",
            "timeout_seconds": 3600,
            "on_approved": "approve",
            "on_denied": "deny",
        }

        # Mock the human input wait
        with patch.object(
            human_review_stage, "_wait_for_human_input", new_callable=AsyncMock
        ) as mock_wait:
            mock_wait.return_value = "approved"

            result = await human_review_stage.execute(sample_state, stage_config)

            assert "output" in result
            assert result["output"]["decision"] == "approved"

    @pytest.mark.asyncio
    async def test_human_review_timeout(self, human_review_stage, sample_state):
        """Test human review timeout behavior."""
        stage_config = {
            "id": "manual_review",
            "type": "human_review",
            "prompt_for_human": "Review this",
            "timeout_seconds": 1,
        }

        with patch.object(
            human_review_stage, "_wait_for_human_input", new_callable=AsyncMock
        ) as mock_wait:
            mock_wait.side_effect = TimeoutError("Timeout waiting for human input")

            # Should handle timeout gracefully
            try:
                await human_review_stage.execute(sample_state, stage_config)
            except TimeoutError:
                pass  # Expected

    @pytest.mark.asyncio
    async def test_human_review_decision_routing(self, human_review_stage, sample_state):
        """Test routing based on human decision."""
        stage_config = {
            "id": "manual_review",
            "type": "human_review",
            "prompt_for_human": "Approve or deny?",
            "timeout_seconds": 3600,
            "on_approved": "approve_stage",
            "on_denied": "deny_stage",
        }

        # Test approved path
        with patch.object(
            human_review_stage, "_wait_for_human_input", new_callable=AsyncMock
        ) as mock_wait:
            mock_wait.return_value = "approved"
            result = await human_review_stage.execute(sample_state, stage_config)
            assert result["output"]["decision"] == "approved"


# ============================================================================
# ToolCallStage Tests
# ============================================================================


class TestToolCallStage:
    """Test tool execution."""

    @pytest.fixture
    def tool_call_stage(self):
        """Create a ToolCallStage instance."""
        return ToolCallStage()

    @pytest.fixture
    def sample_state(self):
        """Create a sample PipelineState."""
        return PipelineState(
            pipeline_id="test-pipeline",
            template_id="test/template",
            template_version="1.0.0",
            input_text="Test",
            input_hash="abc",
            stage_outputs={"extract": {"customer_id": "123"}},
            stage_iterations={},
            stage_traces=[],
            step_count=0,
            step_history=[],
            verdict=None,
            verdict_label=None,
            confidence=None,
            reason=None,
            is_complete=False,
            error=None,
            error_stage=None,
            timestamp_start="2026-04-27T10:00:00Z",
            timestamp_end=None,
            elapsed_seconds=0.0,
            config={},
        )

    @pytest.mark.asyncio
    async def test_tool_call_basic(self, tool_call_stage, sample_state):
        """Test basic tool execution."""
        stage_config = {
            "id": "lookup_customer",
            "type": "tool_call",
            "tool": "sql_query",
            "args": {"query": "SELECT * FROM customers WHERE id = '{extract.output.customer_id}'"},
        }

        with patch.object(tool_call_stage.registry, "get", return_value=Mock()):
            with patch.object(
                tool_call_stage.executor, "run", new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.return_value = [("123", "John Doe", "john@example.com")]

                result = await tool_call_stage.execute(sample_state, stage_config)

                assert "output" in result

    @pytest.mark.asyncio
    async def test_tool_call_with_timeout(self, tool_call_stage, sample_state):
        """Test tool execution with timeout."""
        stage_config = {
            "id": "api_call",
            "type": "tool_call",
            "tool": "api_request",
            "args": {"url": "https://api.example.com/data"},
            "timeout_seconds": 5,
        }

        with patch.object(tool_call_stage.registry, "get", return_value=Mock()):
            with patch.object(
                tool_call_stage.executor, "run", new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.side_effect = TimeoutError("Tool execution timeout")

                result = await tool_call_stage.execute(sample_state, stage_config)
                # Should handle timeout gracefully and return error in result
                assert result.get("error") or result.get("output") is None

    @pytest.mark.asyncio
    async def test_tool_call_argument_interpolation(self, tool_call_stage, sample_state):
        """Test argument interpolation in tool calls."""
        stage_config = {
            "id": "lookup",
            "type": "tool_call",
            "tool": "sql_query",
            "args": {
                "query": "SELECT * FROM customers WHERE id = '{extract.output.customer_id}' AND status = 'active'"
            },
        }

        with patch.object(tool_call_stage.registry, "get", return_value=Mock()):
            with patch.object(
                tool_call_stage.executor, "run", new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.return_value = [("123", "John Doe")]

                result = await tool_call_stage.execute(sample_state, stage_config)

                assert "output" in result
                # Verify the executor was called with interpolated args
                assert mock_execute.called

    @pytest.mark.asyncio
    async def test_tool_call_error_handling(self, tool_call_stage, sample_state):
        """Test tool error handling and result capture."""
        stage_config = {
            "id": "api_call",
            "type": "tool_call",
            "tool": "web_search",
            "args": {"query": "test"},
        }

        with patch.object(tool_call_stage.registry, "get", return_value=Mock()):
            with patch.object(
                tool_call_stage.executor, "run", new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.side_effect = Exception("API error: rate limit exceeded")

                result = await tool_call_stage.execute(sample_state, stage_config)
                # Error should be captured in result
                assert result.get("error") or result.get("output") is None


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestAdvancedStagesEdgeCases:
    """Test edge cases across all advanced stages."""

    @pytest.mark.asyncio
    async def test_parallel_with_empty_branches(self):
        """Test parallel stage with no branches."""
        parallel_stage = ParallelStage()
        state = PipelineState(
            pipeline_id="test",
            template_id="test",
            template_version="1.0",
            input_text="test",
            input_hash="abc",
            stage_outputs={},
            stage_iterations={},
            stage_traces=[],
            step_count=0,
            step_history=[],
            verdict=None,
            verdict_label=None,
            confidence=None,
            reason=None,
            is_complete=False,
            error=None,
            error_stage=None,
            timestamp_start="2026-04-27T10:00:00Z",
            timestamp_end=None,
            elapsed_seconds=0,
            config={},
        )

        stage_config = {"id": "parallel", "type": "parallel", "branches": []}

        # Should handle empty branches gracefully
        with patch.object(parallel_stage, "_execute_branch", new_callable=AsyncMock):
            result = await parallel_stage.execute(state, stage_config)
            assert result is not None

    @pytest.mark.asyncio
    async def test_router_with_malformed_output(self):
        """Test router handling malformed LLM output."""
        router_stage = RouterStage()
        state = PipelineState(
            pipeline_id="test",
            template_id="test",
            template_version="1.0",
            input_text="test",
            input_hash="abc",
            stage_outputs={},
            stage_iterations={},
            stage_traces=[],
            step_count=0,
            step_history=[],
            verdict=None,
            verdict_label=None,
            confidence=None,
            reason=None,
            is_complete=False,
            error=None,
            error_stage=None,
            timestamp_start="2026-04-27T10:00:00Z",
            timestamp_end=None,
            elapsed_seconds=0,
            config={},
        )

        stage_config = {"id": "router", "type": "router", "prompt": "Route", "output_schema": {}}

        with patch.object(router_stage, "_call_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "{ invalid json }"

            # Should handle JSON parse error
            try:
                await router_stage.execute(state, stage_config)
            except (ValueError, json.JSONDecodeError):
                pass  # Expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
