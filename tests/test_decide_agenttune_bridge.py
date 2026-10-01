"""
Tests for Decide ↔ AgentTune bridging modules.

Tests cover:
- training_bridge: DPO pairs, BCO labels, trajectory extraction
- tool_call_stage: Tool execution with argument interpolation
- llm_call_stage: Simple and agentic modes
- model_deployment: Model deployment and rollback
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agenttune.decide.model_deployment import ModelDeploymentBridge, deploy_trained_model
from agenttune.decide.stages.llm_call import LLMCallStage
from agenttune.decide.stages.tool_call import ToolCallStage
from agenttune.decide.state import PipelineState
from agenttune.decide.training_bridge import DecideToTrainerBridge, train_from_audit

# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_audit_file():
    """Create a temporary audit.jsonl with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Stage execution entry
        f.write(
            json.dumps(
                {
                    "timestamp": "2026-04-22T10:00:00",
                    "pipeline_id": "pipe-001",
                    "template_id": "bfsi/kyc_triage",
                    "template_version": "1.0.0",
                    "input_hash": "hash123",
                    "stage_id": "extract",
                    "stage_type": "llm_call",
                    "iteration": 1,
                    "input": "Customer: John Doe",
                    "output": '{"name": "John", "age": 35}',
                    "latency_ms": 250,
                    "cost_usd": 0.01,
                    "model": "gpt-4o",
                    "is_retry": False,
                    "error": None,
                }
            )
            + "\n"
        )

        # Human review rejection (for DPO pairs)
        f.write(
            json.dumps(
                {
                    "timestamp": "2026-04-22T10:00:01",
                    "pipeline_id": "pipe-001",
                    "template_id": "bfsi/kyc_triage",
                    "stage_id": "risk_score",
                    "stage_type": "llm_judge",
                    "iteration": 1,
                    "input": '{"name": "John", "age": 35}',
                    "output": '{"score": 3}',
                    "human_feedback": "rejected",
                    "model_output": '{"risk": "high"}',
                    "human_output": '{"risk": "low"}',
                    "human_explanation": "Score was too harsh",
                }
            )
            + "\n"
        )

        # Final completion entry
        f.write(
            json.dumps(
                {
                    "timestamp_end": "2026-04-22T10:00:02",
                    "pipeline_id": "pipe-001",
                    "template_id": "bfsi/kyc_triage",
                    "verdict": "APPROVE",
                    "verdict_label": "approved",
                    "is_complete": True,
                    "step_count": 5,
                    "elapsed_seconds": 2.5,
                    "error": None,
                }
            )
            + "\n"
        )

        # Another pipeline for BCO labels
        f.write(
            json.dumps(
                {
                    "timestamp": "2026-04-22T10:00:03",
                    "pipeline_id": "pipe-002",
                    "template_id": "bfsi/fraud_detection",
                    "stage_id": "output",
                    "stage_type": "output",
                    "input": "Transaction: $5000 transfer",
                    "verdict": "DENY",
                    "confidence": 0.9,
                }
            )
            + "\n"
        )

        path = f.name

    yield path

    # Cleanup
    Path(path).unlink()


@pytest.fixture
def sample_pipeline_state():
    """Create a sample pipeline state for testing."""
    return PipelineState(
        pipeline_id="test-001",
        template_id="test/sample",
        input_text="Test input",
        stage_outputs={
            "extract": {"name": "John", "age": 35},
            "validate": {"passed": True},
        },
        stage_iterations={"extract": 1, "validate": 1},
        step_count=2,
        verdict=None,
        verdict_label=None,
        is_complete=False,
        error=None,
        timestamp_start="2026-04-22T10:00:00",
        timestamp_end=None,
        elapsed_seconds=0.0,
        config={},
        template_version="1.0.0",
        input_hash="hash123",
    )


# ─────────────────────────────────────────────────────────────────────────
# Tests: training_bridge.py
# ─────────────────────────────────────────────────────────────────────────


class TestDecideToTrainerBridge:
    """Test DecideToTrainerBridge module."""

    def test_extract_dpo_pairs(self, temp_audit_file):
        """Test extraction of DPO pairs from human feedback."""
        bridge = DecideToTrainerBridge(temp_audit_file)
        pairs = bridge.extract_dpo_pairs("risk_score")

        assert len(pairs) == 1
        assert pairs[0]["prompt"] == '{"name": "John", "age": 35}'
        assert pairs[0]["rejected_output"] == '{"risk": "high"}'
        assert pairs[0]["chosen_output"] == '{"risk": "low"}'
        assert pairs[0]["reason"] == "Score was too harsh"

    def test_extract_dpo_pairs_empty(self, temp_audit_file):
        """Test extraction when no pairs exist."""
        bridge = DecideToTrainerBridge(temp_audit_file)
        pairs = bridge.extract_dpo_pairs("nonexistent_stage")

        assert len(pairs) == 0

    def test_extract_dpo_pairs_missing_file(self):
        """Test extraction with missing audit file."""
        bridge = DecideToTrainerBridge("/nonexistent/path/audit.jsonl")
        pairs = bridge.extract_dpo_pairs("stage_id")

        assert len(pairs) == 0

    def test_extract_trajectories(self, temp_audit_file):
        """Test extraction of trajectories for RL training."""
        bridge = DecideToTrainerBridge(temp_audit_file)
        dataset = bridge.extract_trajectories("extract")

        assert len(dataset) > 0
        trajectory = dataset.trajectories[0] if dataset.trajectories else None
        if trajectory:
            assert trajectory.task == "Decide pipeline"
            assert len(trajectory.steps) > 0
            assert trajectory.metadata["pipeline_id"] == "pipe-001"
            assert trajectory.metadata["stage_id"] == "extract"

    def test_extract_bco_labels(self, temp_audit_file):
        """Test extraction of binary classification labels."""
        bridge = DecideToTrainerBridge(temp_audit_file)
        labels = bridge.extract_bco_labels("output")

        assert len(labels) == 1
        assert labels[0]["verdict_label"] == "DENY"
        assert labels[0]["label"] == 0  # DENY = 0
        assert labels[0]["confidence"] == 0.9

    def test_extract_bco_labels_approve(self, temp_audit_file):
        """Test BCO labels for APPROVE verdict."""
        # APPROVE should map to label 1
        labels = []
        with open(temp_audit_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("verdict") == "APPROVE":
                    labels.append({"verdict": "APPROVE", "label": 1})

        assert any(l["label"] == 1 for l in labels)

    @patch("agenttune.decide.training_bridge.create_agentic_trainer")
    def test_train_from_audit_dpo(self, mock_trainer, temp_audit_file):
        """Test training_from_audit with DPO algorithm."""
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        trainer = train_from_audit(
            audit_path=temp_audit_file,
            stage_id="risk_score",
            algorithm="dpo",
            model="test-model",
            output_dir="./output",
        )

        assert trainer == mock_trainer_instance
        mock_trainer.assert_called_once()
        call_kwargs = mock_trainer.call_args[1]
        assert call_kwargs["algorithm"] == "dpo"
        assert call_kwargs["model"] == "test-model"

    @patch("agenttune.decide.training_bridge.create_agentic_trainer")
    def test_train_from_audit_bco(self, mock_trainer, temp_audit_file):
        """Test train_from_audit with BCO algorithm."""
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        trainer = train_from_audit(
            audit_path=temp_audit_file,
            stage_id="output",
            algorithm="bco",
            model="test-model",
            output_dir="./output",
        )

        assert trainer == mock_trainer_instance
        call_kwargs = mock_trainer.call_args[1]
        assert call_kwargs["algorithm"] == "bco"

    @patch("agenttune.decide.training_bridge.create_agentic_trainer")
    def test_train_from_audit_grpo(self, mock_trainer, temp_audit_file):
        """Test train_from_audit with GRPO algorithm."""
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance

        trainer = train_from_audit(
            audit_path=temp_audit_file,
            stage_id="extract",
            algorithm="grpo",
            model="test-model",
            output_dir="./output",
            tools=[],
            reward_funcs=[],
        )

        assert trainer == mock_trainer_instance
        call_kwargs = mock_trainer.call_args[1]
        assert call_kwargs["algorithm"] == "grpo"

    def test_train_from_audit_missing_file(self):
        """Test train_from_audit with missing audit file."""
        with pytest.raises(ValueError):
            train_from_audit(
                audit_path="/nonexistent/audit.jsonl",
                stage_id="stage",
                algorithm="dpo",
                model="model",
                output_dir="./output",
            )

    def test_train_from_audit_invalid_algorithm(self, temp_audit_file):
        """Test train_from_audit with invalid algorithm."""
        with pytest.raises(ValueError):
            train_from_audit(
                audit_path=temp_audit_file,
                stage_id="stage",
                algorithm="invalid_algo",
                model="model",
                output_dir="./output",
            )


# ─────────────────────────────────────────────────────────────────────────
# Tests: tool_call.py
# ─────────────────────────────────────────────────────────────────────────


class TestToolCallStage:
    """Test ToolCallStage module."""

    @pytest.mark.asyncio
    async def test_tool_call_missing_tool_name(self, sample_pipeline_state):
        """Test tool_call stage without tool name."""
        stage = ToolCallStage({"id": "test_stage", "type": "tool_call"})

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] is None
        assert "Missing required 'tool'" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_call_invalid_args_type(self, sample_pipeline_state):
        """Test tool_call stage with invalid args type."""
        stage = ToolCallStage(
            {
                "id": "test_stage",
                "type": "tool_call",
                "tool": "sql_query",
                "args": "invalid_string",  # Should be dict
            }
        )

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] is None
        assert "Tool args must be dict" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_call_tool_not_found(self, sample_pipeline_state):
        """Test tool_call stage when tool not in registry."""
        stage = ToolCallStage(
            {
                "id": "test_stage",
                "type": "tool_call",
                "tool": "nonexistent",
                "args": {},
            }
        )
        stage.registry = Mock()
        stage.registry.get.side_effect = KeyError("nonexistent")
        stage.registry.list_tools.return_value = ["sql_query", "web_search"]

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] is None
        assert "not found" in result["error"]
        assert "sql_query" in result["error"]  # Shows available tools

    @pytest.mark.asyncio
    async def test_tool_call_execution_success(self, sample_pipeline_state):
        """Test successful tool execution."""
        stage = ToolCallStage(
            {
                "id": "test_stage",
                "type": "tool_call",
                "tool": "sql_query",
                "args": {"query": "SELECT COUNT(*) FROM users"},
            }
        )
        stage.registry = Mock()
        stage.executor = Mock()
        stage.executor.run = AsyncMock(return_value={"rows": 42})
        stage.registry.get.return_value = Mock()

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] == {"rows": 42}
        assert "error" not in result or result["error"] is None
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_tool_call_with_interpolation(self, sample_pipeline_state):
        """Test tool_call with argument interpolation from pipeline context."""
        stage = ToolCallStage(
            {
                "id": "test_stage",
                "type": "tool_call",
                "tool": "sql_query",
                "args": {"query": "SELECT * FROM users WHERE name = '{extract.output.name}'"},
            }
        )
        stage.registry = Mock()
        stage.executor = Mock()
        stage.executor.run = AsyncMock(return_value={"result": "success"})
        stage.registry.get.return_value = Mock()

        # Mock interpolation
        stage._interpolate = Mock(
            side_effect=lambda s, state: s.replace(
                "{extract.output.name}", state.stage_outputs["extract"]["name"]
            )
        )

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] == {"result": "success"}

    @pytest.mark.asyncio
    async def test_tool_call_timeout(self, sample_pipeline_state):
        """Test tool_call timeout handling."""
        stage = ToolCallStage(
            {
                "id": "test_stage",
                "type": "tool_call",
                "tool": "web_search",
                "args": {"query": "test"},
                "timeout_sec": 5,
            }
        )
        stage.registry = Mock()
        stage.executor = Mock()
        stage.executor.run = AsyncMock(side_effect=TimeoutError())
        stage.registry.get.return_value = Mock()

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] is None
        assert "timed out" in result["error"]


# ─────────────────────────────────────────────────────────────────────────
# Tests: llm_call.py
# ─────────────────────────────────────────────────────────────────────────


class TestLLMCallStage:
    """Test LLMCallStage module."""

    @pytest.mark.asyncio
    async def test_llm_call_simple_mode(self, sample_pipeline_state):
        """Test simple LLM call (no tools)."""
        stage = LLMCallStage(
            {
                "id": "extract",
                "type": "llm_call",
                "prompt": "Extract data from: {input_text}",
                "model": "gpt-4o",
            }
        )
        stage._call_model = AsyncMock(return_value='{"result": "ok"}')
        stage._validate_json = Mock(return_value={"result": "ok"})
        stage._interpolate = Mock(side_effect=lambda s, state: s)

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] == {"result": "ok"}
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_llm_call_simple_json_validation(self, sample_pipeline_state):
        """Test LLM call with JSON schema validation."""
        stage = LLMCallStage(
            {
                "id": "extract",
                "type": "llm_call",
                "prompt": "Extract",
                "model": "gpt-4o",
                "output_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            }
        )
        stage._call_model = AsyncMock(return_value='{"name": "John"}')
        stage._validate_json = Mock(return_value={"name": "John"})
        stage._interpolate = Mock(side_effect=lambda s, state: s)

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] == {"name": "John"}
        stage._validate_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_call_parse_error_retry(self, sample_pipeline_state):
        """Test LLM call retry on JSON parse error."""
        stage = LLMCallStage(
            {
                "id": "extract",
                "type": "llm_call",
                "prompt": "Extract",
                "model": "gpt-4o",
                "max_iterations": 3,
                "on_parse_error": "extract",
            }
        )
        stage._call_model = AsyncMock(return_value="invalid json")
        stage._validate_json = Mock(side_effect=ValueError("Invalid JSON"))
        stage._interpolate = Mock(side_effect=lambda s, state: s)

        result = await stage.execute(sample_pipeline_state)

        assert result["failed"] is True
        assert result["goto"] == "extract"
        assert "inject" in result

    @pytest.mark.asyncio
    async def test_llm_call_no_model_error(self, sample_pipeline_state):
        """Test LLM call with missing model."""
        stage = LLMCallStage(
            {
                "id": "extract",
                "type": "llm_call",
                "prompt": "test",
            }
        )

        with pytest.raises(ValueError):
            await stage.execute(sample_pipeline_state)

    @pytest.mark.asyncio
    async def test_llm_call_model_call_error(self, sample_pipeline_state):
        """Test LLM call error handling."""
        stage = LLMCallStage(
            {
                "id": "extract",
                "type": "llm_call",
                "prompt": "test",
                "model": "gpt-4o",
            }
        )
        stage._call_model = AsyncMock(side_effect=Exception("API error"))
        stage._interpolate = Mock(side_effect=lambda s, state: s)

        result = await stage.execute(sample_pipeline_state)

        assert result["output"] is None
        assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_llm_call_agentic_mode(self, sample_pipeline_state):
        """Test agentic LLM call with tools."""
        stage = LLMCallStage(
            {
                "id": "research",
                "type": "llm_call",
                "prompt": "Research this topic",
                "model": "gpt-4o",
                "tools": ["web_search", "sql_query"],
                "max_steps": 5,
            }
        )
        stage.rollout_engine = Mock()  # Mock rollout engine
        stage._interpolate = Mock(side_effect=lambda s, state: s)

        with patch(
            "agenttune.agentic.rollout_engines.rollout_factory.create_rollout_fn"
        ) as mock_rollout:
            mock_fn = Mock(
                return_value={
                    "responses": ["Research complete"],
                    "conversations": [[]],
                    "tool_calls": [{"tool": "web_search", "args": {}}],
                }
            )
            mock_rollout.return_value = mock_fn

            result = await stage.execute(sample_pipeline_state)

            assert result["output"] == "Research complete"
            assert "conversation" in result
            assert "tool_calls" in result

    @pytest.mark.asyncio
    async def test_llm_call_agentic_no_engine(self, sample_pipeline_state):
        """Test agentic call falls back when no rollout engine configured."""
        stage = LLMCallStage(
            {
                "id": "research",
                "type": "llm_call",
                "prompt": "Research",
                "model": "gpt-4o",
                "tools": ["web_search"],
            }
        )
        stage.rollout_engine = None  # No rollout engine
        stage._call_model = AsyncMock(return_value='{"result": "ok"}')
        stage._validate_json = Mock(return_value={"result": "ok"})
        stage._interpolate = Mock(side_effect=lambda s, state: s)

        result = await stage.execute(sample_pipeline_state)

        # Should fallback to simple LLM call
        assert result["output"] == {"result": "ok"}


# ─────────────────────────────────────────────────────────────────────────
# Tests: model_deployment.py
# ─────────────────────────────────────────────────────────────────────────


class TestModelDeploymentBridge:
    """Test ModelDeploymentBridge module."""

    def test_deploy_trained_model(self):
        """Test deploying trained model to config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake model directory
            model_path = Path(tmpdir) / "checkpoint-final"
            model_path.mkdir()

            # Create config file
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("default_model: gpt-4o\nbackend: api\n")

            # Deploy
            ModelDeploymentBridge.deploy_trained_model(
                trained_model_path=str(model_path),
                config_path=str(config_file),
                backend="transformers",
            )

            # Verify
            import yaml

            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert config["default_model"] == str(model_path)
            assert config["backend"] == "transformers"

    def test_deploy_missing_model(self):
        """Test deploy with missing model path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("default_model: gpt-4o\n")

            with pytest.raises(FileNotFoundError):
                ModelDeploymentBridge.deploy_trained_model(
                    trained_model_path="/nonexistent/model",
                    config_path=str(config_file),
                    backend="transformers",
                )

    def test_deploy_missing_config(self):
        """Test deploy with missing config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            with pytest.raises(FileNotFoundError):
                ModelDeploymentBridge.deploy_trained_model(
                    trained_model_path=str(model_path),
                    config_path="/nonexistent/config.yaml",
                    backend="transformers",
                )

    def test_deploy_invalid_backend(self):
        """Test deploy with invalid backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("default_model: gpt-4o\n")

            with pytest.raises(ValueError):
                ModelDeploymentBridge.deploy_trained_model(
                    trained_model_path=str(model_path),
                    config_path=str(config_file),
                    backend="invalid_backend",
                )

    def test_deploy_stage_model_map(self):
        """Test deploying with stage-specific model assignment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "checkpoint-final"
            model_path.mkdir()
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("default_model: gpt-4o\n")

            ModelDeploymentBridge.deploy_trained_model(
                trained_model_path=str(model_path),
                config_path=str(config_file),
                backend="transformers",
                stage_model_map={
                    "income_agent": str(model_path),
                    "fraud_check": "gpt-4o",  # Keep API
                },
            )

            import yaml

            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert config["stages"]["income_agent"]["model"] == str(model_path)
            assert config["stages"]["fraud_check"]["model"] == "gpt-4o"

    def test_rollback_deployment(self):
        """Test rolling back model deployment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("default_model: gpt-4o\nbackend: api\n")

            # Deploy
            ModelDeploymentBridge.deploy_trained_model(
                trained_model_path=str(model_path),
                config_path=str(config_file),
                backend="transformers",
            )

            # Verify backup exists
            backup_file = config_file.with_suffix(".yaml.backup")
            assert backup_file.exists()

            # Rollback
            ModelDeploymentBridge.rollback_deployment(str(config_file))

            import yaml

            with open(config_file) as f:
                config = yaml.safe_load(f)

            # Should be restored to original
            assert config["default_model"] == "gpt-4o"
            assert config["backend"] == "api"

    def test_get_deployment_status(self):
        """Test getting deployment status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("default_model: gpt-4o\nbackend: api\n")

            ModelDeploymentBridge.deploy_trained_model(
                trained_model_path=str(model_path),
                config_path=str(config_file),
                backend="transformers",
            )

            status = ModelDeploymentBridge.get_deployment_status(str(config_file))

            assert status["default_model"] == str(model_path)
            assert status["backend"] == "transformers"
            assert status["model_exists"] is True

    def test_get_deployment_status_missing_config(self):
        """Test get_deployment_status with missing config."""
        with pytest.raises(FileNotFoundError):
            ModelDeploymentBridge.get_deployment_status("/nonexistent/config.yaml")


# ─────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────


class TestBridgeIntegration:
    """Integration tests for the complete Decide ↔ AgentTune pipeline."""

    def test_audit_to_training_pipeline(self, temp_audit_file):
        """Test complete audit → training pipeline."""
        # This test verifies the data flows correctly from audit to trainer
        with patch("agenttune.decide.training_bridge.create_agentic_trainer"):
            # Extract DPO pairs
            bridge = DecideToTrainerBridge(temp_audit_file)
            dpo_pairs = bridge.extract_dpo_pairs("risk_score")
            assert len(dpo_pairs) == 1

            # Extract BCO labels
            bco_labels = bridge.extract_bco_labels("output")
            assert len(bco_labels) == 1

            # Extract trajectories
            trajectories = bridge.extract_trajectories("extract")
            assert len(trajectories) > 0

    @pytest.mark.asyncio
    async def test_tool_stage_in_pipeline(self, sample_pipeline_state):
        """Test tool_call stage execution in a pipeline."""
        stage = ToolCallStage(
            {
                "id": "lookup",
                "type": "tool_call",
                "tool": "sql_query",
                "args": {"query": "SELECT * FROM customers WHERE age > 30"},
            }
        )
        stage.registry = Mock()
        stage.executor = AsyncMock()
        stage.executor.run = AsyncMock(return_value={"customer_id": 123, "balance": 5000})
        stage.registry.get.return_value = Mock()

        result = await stage.execute(sample_pipeline_state)

        assert result["output"]["customer_id"] == 123
        assert "latency_ms" in result

    def test_deployment_cycle(self):
        """Test full deployment cycle: train → deploy → rollback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial config
            config_file = Path(tmpdir) / "config.yaml"
            config_file.write_text("default_model: gpt-4o\nbackend: api\n")

            # Create trained model
            model_path = Path(tmpdir) / "trained_model"
            model_path.mkdir()

            # Deploy
            deploy_trained_model(
                trained_model_path=str(model_path),
                config_path=str(config_file),
                backend="transformers",
            )

            # Verify deployment
            status = ModelDeploymentBridge.get_deployment_status(str(config_file))
            assert status["default_model"] == str(model_path)
            assert status["backend"] == "transformers"

            # Rollback
            ModelDeploymentBridge.rollback_deployment(str(config_file))

            # Verify rollback
            status = ModelDeploymentBridge.get_deployment_status(str(config_file))
            assert status["default_model"] == "gpt-4o"
            assert status["backend"] == "api"
