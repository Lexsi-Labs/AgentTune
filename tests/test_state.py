"""
Tests for PipelineState: state management and mutation.

Tests the dataclass that carries state through LangGraph execution.
"""

import uuid

from agenttune.decide.state import PipelineState


class TestPipelineStateCreation:
    """Test creating and initializing PipelineState."""

    def test_create_basic_state(self):
        """Test creating a minimal PipelineState."""
        state = PipelineState(
            pipeline_id=str(uuid.uuid4()),
            template_id="test/template",
            template_version="1.0.0",
            input_text="test input",
            input_hash="abc123",
        )
        assert state.pipeline_id is not None
        assert state.template_id == "test/template"
        assert state.template_version == "1.0.0"
        assert state.input_text == "test input"
        assert state.input_hash == "abc123"

    def test_create_state_with_defaults(self):
        """Test that defaults are set correctly."""
        state = PipelineState(
            pipeline_id="test_id",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        assert state.stage_outputs == {}
        assert state.stage_iterations == {}
        assert state.stage_traces == []
        assert state.step_count == 0
        assert state.step_history == []
        assert state.verdict is None
        assert state.is_complete is False
        assert state.error is None

    def test_state_timestamp_generated(self):
        """Test that timestamp_start is generated."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        # Should be a valid ISO format timestamp
        assert isinstance(state.timestamp_start, str)
        assert "T" in state.timestamp_start or ":" in state.timestamp_start


class TestStageOutputTracking:
    """Test tracking and accessing stage outputs."""

    def test_add_stage_output(self):
        """Test adding a stage output."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        output = {"field1": "value1", "field2": 42}
        state.add_stage_output("stage1", output)
        assert state.stage_outputs["stage1"] == output

    def test_add_multiple_stage_outputs(self):
        """Test adding multiple stage outputs."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        state.add_stage_output("stage1", {"output": "first"})
        state.add_stage_output("stage2", {"output": "second"})

        assert len(state.stage_outputs) == 2
        assert state.stage_outputs["stage1"]["output"] == "first"
        assert state.stage_outputs["stage2"]["output"] == "second"

    def test_overwrite_stage_output(self):
        """Test that adding a stage output twice overwrites."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        state.add_stage_output("stage1", {"output": "first"})
        state.add_stage_output("stage1", {"output": "second"})
        assert state.stage_outputs["stage1"]["output"] == "second"

    def test_nested_stage_output(self):
        """Test adding complex nested outputs."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        output = {
            "score": 8,
            "explanation": "test",
            "nested": {"field1": "value1", "field2": [1, 2, 3]},
        }
        state.add_stage_output("judge_stage", output)
        assert state.stage_outputs["judge_stage"]["nested"]["field2"] == [1, 2, 3]


class TestIterationTracking:
    """Test tracking stage iterations for loops."""

    def test_increment_iteration(self):
        """Test incrementing iteration count."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        count = state.increment_stage_iteration("stage1")
        assert count == 1
        assert state.stage_iterations["stage1"] == 1

    def test_increment_iteration_multiple_times(self):
        """Test incrementing iteration multiple times."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        count1 = state.increment_stage_iteration("stage1")
        count2 = state.increment_stage_iteration("stage1")
        count3 = state.increment_stage_iteration("stage1")

        assert count1 == 1
        assert count2 == 2
        assert count3 == 3
        assert state.stage_iterations["stage1"] == 3

    def test_increment_different_stages(self):
        """Test incrementing iterations for different stages."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        state.increment_stage_iteration("stage1")
        state.increment_stage_iteration("stage1")
        state.increment_stage_iteration("stage2")

        assert state.stage_iterations["stage1"] == 2
        assert state.stage_iterations["stage2"] == 1

    def test_get_iteration_default(self):
        """Test that getting iteration for unknown stage returns 0."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        count = state.stage_iterations.get("unknown_stage", 0)
        assert count == 0


class TestTraceTracking:
    """Test audit trail tracking."""

    def test_add_trace(self):
        """Test adding a trace entry."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        trace = {
            "stage_id": "stage1",
            "type": "llm_call",
            "input": "prompt",
            "output": {"result": "success"},
        }
        state.add_trace(trace)
        assert len(state.stage_traces) == 1
        assert state.stage_traces[0]["stage_id"] == "stage1"

    def test_add_multiple_traces(self):
        """Test adding multiple trace entries."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        for i in range(3):
            state.add_trace({"stage_id": f"stage{i}", "iteration": i})

        assert len(state.stage_traces) == 3
        assert state.stage_traces[0]["stage_id"] == "stage0"
        assert state.stage_traces[2]["stage_id"] == "stage2"

    def test_trace_with_timing_info(self):
        """Test trace with timing and cost info."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        trace = {
            "stage_id": "stage1",
            "latency_ms": 1234,
            "cost_usd": 0.015,
            "timestamp": "2026-04-14T10:30:45Z",
        }
        state.add_trace(trace)
        assert state.stage_traces[0]["latency_ms"] == 1234
        assert state.stage_traces[0]["cost_usd"] == 0.015


class TestVerdictTracking:
    """Test setting and tracking final verdict."""

    def test_set_verdict(self):
        """Test setting verdict."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        state.verdict = "APPROVE"
        state.verdict_label = "kyc_approved"
        state.confidence = 9
        state.reason = "All checks passed"

        assert state.verdict == "APPROVE"
        assert state.verdict_label == "kyc_approved"
        assert state.confidence == 9
        assert state.reason == "All checks passed"

    def test_verdict_initially_none(self):
        """Test that verdict is initially None."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        assert state.verdict is None
        assert state.verdict_label is None


class TestErrorTracking:
    """Test error tracking."""

    def test_set_error(self):
        """Test setting error."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        state.error = "Max iterations reached"
        state.error_stage = "stage1"
        state.is_complete = True

        assert state.error == "Max iterations reached"
        assert state.error_stage == "stage1"
        assert state.is_complete is True

    def test_error_initially_none(self):
        """Test that error is initially None."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        assert state.error is None
        assert state.error_stage is None


class TestStateCompletion:
    """Test marking pipeline as complete."""

    def test_mark_complete(self):
        """Test marking pipeline as complete."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        state.is_complete = True
        state.timestamp_end = "2026-04-14T10:31:00Z"
        state.elapsed_seconds = 15.5

        assert state.is_complete is True
        assert state.timestamp_end is not None
        assert state.elapsed_seconds == 15.5

    def test_step_counting(self):
        """Test tracking step counts."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        state.step_count = 0
        state.step_history = []

        state.step_count += 1
        state.step_history.append("stage1")

        state.step_count += 1
        state.step_history.append("stage2")

        assert state.step_count == 2
        assert state.step_history == ["stage1", "stage2"]


class TestConfigReference:
    """Test config reference in state."""

    def test_store_config(self):
        """Test storing config in state."""
        config = {
            "id": "test/template",
            "default_model": "claude-haiku-4-5",
            "api_keys": {"anthropic": "sk-test"},
        }
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
            config=config,
        )
        assert state.config["default_model"] == "claude-haiku-4-5"
        assert state.config["api_keys"]["anthropic"] == "sk-test"

    def test_default_empty_config(self):
        """Test that config defaults to empty dict."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        assert state.config == {}


class TestNextStageRouting:
    """Test next stage routing field."""

    def test_set_next_stage(self):
        """Test setting next stage."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        state.next_stage = "stage2"
        assert state.next_stage == "stage2"

    def test_next_stage_initially_none(self):
        """Test that next_stage is initially None."""
        state = PipelineState(
            pipeline_id="test",
            template_id="test/template",
            template_version="1.0.0",
            input_text="test",
            input_hash="hash",
        )
        assert state.next_stage is None
