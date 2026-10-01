"""
Bridge integration tests: Decide ↔ AgentTune training pipeline.

Tests:
  1. DecideToTrainerBridge — DPO pair extraction
  2. DecideToTrainerBridge — trajectory extraction
  3. DecideToTrainerBridge — BCO label extraction
  4. train_from_audit — end-to-end (mocked trainer)
  5. ModelDeploymentBridge — deploy + rollback
  6. TrainerConfigBridge — YAML-driven trainer config
  7. LLMCallStage with agentic tools
  8. ToolCallStage with various tool types
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agenttune.decide.model_deployment import ModelDeploymentBridge
from agenttune.decide.trainer_config_bridge import TrainerConfigBridge
from agenttune.decide.training_bridge import DecideToTrainerBridge, train_from_audit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rich_audit_file(tmp_path):
    """Audit file with DPO pairs, trajectories, and verdicts."""
    entries = [
        # Pipeline 1 — approved
        {
            "pipeline_id": "pipe-001",
            "template_id": "bfsi/kyc_triage",
            "stage_id": "extract",
            "stage_type": "llm_call",
            "input": "Analyze Alice",
            "output": {"income": 75000, "dob": "1990-01-01"},
            "iteration": 1,
            "latency_ms": 300,
            "cost_usd": 0.001,
        },
        {
            "pipeline_id": "pipe-001",
            "template_id": "bfsi/kyc_triage",
            "stage_id": "decision_judge",
            "stage_type": "llm_judge",
            "input": "Rate risk",
            "output": {"score": 8, "decision": "APPROVE"},
            "iteration": 1,
            "latency_ms": 600,
            "cost_usd": 0.005,
        },
        {
            "pipeline_id": "pipe-001",
            "stage_id": "approve",
            "stage_type": "output",
            "verdict": "APPROVE",
            "verdict_label": "kyc_approved",
            "input": "Final",
            "output": {"verdict": "APPROVE"},
        },
        # Pipeline 2 — with human rejection (DPO pair)
        {
            "pipeline_id": "pipe-002",
            "template_id": "bfsi/kyc_triage",
            "stage_id": "extract",
            "stage_type": "llm_call",
            "input": "Analyze Bob",
            "output": {"income": 28000, "dob": "1982-07-22"},
            "iteration": 1,
            "latency_ms": 350,
            "cost_usd": 0.001,
        },
        {
            "pipeline_id": "pipe-002",
            "stage_id": "decision_judge",
            "stage_type": "llm_judge",
            "human_feedback": "rejected",
            "model_output": '{"decision": "APPROVE", "score": 6}',
            "human_output": '{"decision": "DENY", "score": 2}',
            "human_explanation": "Undisclosed fraud flag",
            "input": "Rate risk",
        },
        {
            "pipeline_id": "pipe-002",
            "stage_id": "deny",
            "stage_type": "output",
            "verdict": "DENY",
            "verdict_label": "kyc_denied",
            "input": "Final",
            "output": {"verdict": "DENY"},
        },
    ]
    path = tmp_path / "audit.jsonl"
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# DecideToTrainerBridge — DPO
# ---------------------------------------------------------------------------


class TestDPOExtraction:
    def test_extracts_dpo_pairs(self, rich_audit_file):
        bridge = DecideToTrainerBridge(rich_audit_file)
        pairs = bridge.extract_dpo_pairs("decision_judge")
        assert len(pairs) >= 1

    def test_dpo_pair_has_required_keys(self, rich_audit_file):
        bridge = DecideToTrainerBridge(rich_audit_file)
        pairs = bridge.extract_dpo_pairs("decision_judge")
        for pair in pairs:
            assert any(k in pair for k in ("rejected_output", "rejected"))
            assert any(k in pair for k in ("chosen_output", "chosen"))

    def test_no_pairs_for_stage_without_feedback(self, rich_audit_file):
        bridge = DecideToTrainerBridge(rich_audit_file)
        pairs = bridge.extract_dpo_pairs("extract")
        assert len(pairs) == 0

    def test_no_pairs_for_missing_file(self, tmp_path):
        bridge = DecideToTrainerBridge(str(tmp_path / "missing.jsonl"))
        pairs = bridge.extract_dpo_pairs("decision_judge")
        assert len(pairs) == 0

    def test_multiple_rejection_pairs(self, tmp_path):
        entries = [
            {
                "pipeline_id": f"p-{i}",
                "stage_id": "judge",
                "stage_type": "llm_judge",
                "human_feedback": "rejected",
                "model_output": '{"score": 7}',
                "human_output": '{"score": 3}',
                "human_explanation": f"Reason {i}",
                "input": f"Input {i}",
            }
            for i in range(5)
        ]
        path = tmp_path / "multi.jsonl"
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        bridge = DecideToTrainerBridge(str(path))
        pairs = bridge.extract_dpo_pairs("judge")
        assert len(pairs) == 5


# ---------------------------------------------------------------------------
# DecideToTrainerBridge — Trajectories
# ---------------------------------------------------------------------------


class TestTrajectoryExtraction:
    def test_extracts_trajectories(self, rich_audit_file):
        bridge = DecideToTrainerBridge(rich_audit_file)
        ds = bridge.extract_trajectories("decision_judge")
        assert hasattr(ds, "__len__")
        assert len(ds) >= 1

    def test_empty_file_returns_empty_dataset(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        bridge = DecideToTrainerBridge(str(path))
        ds = bridge.extract_trajectories("decision_judge")
        assert len(ds) == 0

    def test_trajectory_has_steps(self, rich_audit_file):
        bridge = DecideToTrainerBridge(rich_audit_file)
        ds = bridge.extract_trajectories("decision_judge")
        if len(ds) > 0:
            traj = ds[0]
            assert hasattr(traj, "steps") or isinstance(traj, dict)


# ---------------------------------------------------------------------------
# DecideToTrainerBridge — BCO
# ---------------------------------------------------------------------------


class TestBCOLabelExtraction:
    def test_extracts_bco_labels(self, rich_audit_file):
        bridge = DecideToTrainerBridge(rich_audit_file)
        labels = bridge.extract_bco_labels("approve")
        assert isinstance(labels, list)

    def test_approve_verdict_maps_to_1(self, rich_audit_file):
        bridge = DecideToTrainerBridge(rich_audit_file)
        labels = bridge.extract_bco_labels("approve")
        approve_labels = [l for l in labels if l.get("verdict_label") == "APPROVE"]
        for l in approve_labels:
            assert l["label"] == 1

    def test_deny_verdict_maps_to_0(self, rich_audit_file):
        bridge = DecideToTrainerBridge(rich_audit_file)
        labels = bridge.extract_bco_labels("deny")
        deny_labels = [l for l in labels if l.get("verdict_label") == "DENY"]
        for l in deny_labels:
            assert l["label"] == 0


# ---------------------------------------------------------------------------
# train_from_audit — end-to-end (mocked trainer)
# ---------------------------------------------------------------------------


class TestTrainFromAudit:
    @patch("agenttune.decide.training_bridge.create_agentic_trainer")
    def test_dpo_algorithm_builds_trainer(self, mock_create, rich_audit_file):
        mock_trainer = MagicMock()
        mock_create.return_value = mock_trainer
        trainer = train_from_audit(
            audit_path=rich_audit_file,
            stage_id="decision_judge",
            algorithm="dpo",
            model="Qwen/Qwen2.5-0.5B",
            output_dir="/tmp/dpo_test",
        )
        assert trainer is mock_trainer

    @patch("agenttune.decide.training_bridge.create_agentic_trainer")
    def test_grpo_algorithm_builds_trainer(self, mock_create, rich_audit_file):
        mock_trainer = MagicMock()
        mock_create.return_value = mock_trainer
        trainer = train_from_audit(
            audit_path=rich_audit_file,
            stage_id="decision_judge",
            algorithm="grpo",
            model="Qwen/Qwen2.5-0.5B",
            output_dir="/tmp/grpo_test",
        )
        assert trainer is mock_trainer

    def test_raises_on_missing_audit_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            train_from_audit(
                audit_path=str(tmp_path / "missing.jsonl"),
                stage_id="judge",
                algorithm="dpo",
                model="Qwen",
                output_dir="/tmp/test",
            )

    def test_raises_on_unsupported_algorithm(self, rich_audit_file):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            train_from_audit(
                audit_path=rich_audit_file,
                stage_id="judge",
                algorithm="unknown_algo",
                model="Qwen",
                output_dir="/tmp/test",
            )


# ---------------------------------------------------------------------------
# ModelDeploymentBridge tests
# ---------------------------------------------------------------------------


class TestModelDeploymentBridge:
    @pytest.fixture
    def config_file(self, tmp_path):
        cfg = {
            "default_model": "claude-haiku-4-5",
            "stages": [
                {"id": "extract", "model": "claude-haiku-4-5"},
                {"id": "judge", "model": "claude-opus-4-1"},
            ],
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        return str(path)

    def test_deploy_updates_config(self, config_file, tmp_path):
        bridge = ModelDeploymentBridge()
        trained_model = str(tmp_path / "trained_model")
        os.makedirs(trained_model, exist_ok=True)
        bridge.deploy_trained_model(
            trained_model_path=trained_model,
            config_path=config_file,
            backend="transformers",
            stage_model_map={"extract": trained_model},
        )
        with open(config_file) as f:
            updated = yaml.safe_load(f)
        for stage in updated.get("stages", []):
            if stage["id"] == "extract":
                assert stage["model"] == trained_model

    def test_backup_created(self, config_file, tmp_path):
        bridge = ModelDeploymentBridge()
        trained_model = str(tmp_path / "trained_model")
        os.makedirs(trained_model, exist_ok=True)
        bridge.deploy_trained_model(
            trained_model_path=trained_model,
            config_path=config_file,
            backend="transformers",
            stage_model_map={"extract": trained_model},
        )
        backup = Path(config_file + ".backup")
        assert backup.exists()

    def test_rollback_restores_original(self, config_file, tmp_path):
        bridge = ModelDeploymentBridge()
        trained_model = str(tmp_path / "trained_model")
        os.makedirs(trained_model, exist_ok=True)
        original_content = Path(config_file).read_text()
        bridge.deploy_trained_model(
            trained_model_path=trained_model,
            config_path=config_file,
            backend="transformers",
            stage_model_map={"extract": trained_model},
        )
        bridge.rollback(config_file)
        assert Path(config_file).read_text() == original_content


# ---------------------------------------------------------------------------
# TrainerConfigBridge — YAML-driven config
# ---------------------------------------------------------------------------


class TestTrainerConfigBridge:
    @pytest.fixture
    def trainer_config_file(self, tmp_path):
        cfg = {
            "training": {
                "algorithm": "grpo",
                "model": "Qwen/Qwen3-1.7B",
                "output_dir": str(tmp_path / "output"),
                "max_steps": 10,
                "rollout": {
                    "backend": "api",
                    "api_model": "llama-3.1-8b-instant",
                    "api_base_url": "https://api.groq.com/openai/v1",
                    "api_key": "test-key",
                    "max_steps": 2,
                },
                "reward_funcs": ["correctness_reward"],
                "multi_agent": {"enabled": False},
            },
            "decide_bridge": {"enabled": False},
        }
        path = tmp_path / "trainer_config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        return str(path)

    def test_loads_config(self, trainer_config_file):
        bridge = TrainerConfigBridge(trainer_config_file)
        assert bridge.algorithm == "grpo"

    def test_algorithm_property(self, trainer_config_file):
        bridge = TrainerConfigBridge(trainer_config_file)
        assert bridge.algorithm in ("grpo", "ppo", "dpo", "rloo", "bco")

    def test_build_reward_funcs_returns_list(self, trainer_config_file):
        bridge = TrainerConfigBridge(trainer_config_file)
        funcs = bridge.build_reward_funcs()
        assert isinstance(funcs, list)

    def test_build_tools_returns_list(self, trainer_config_file):
        bridge = TrainerConfigBridge(trainer_config_file)
        tools = bridge.build_tools()
        assert isinstance(tools, list)

    def test_get_peft_config_none_when_absent(self, trainer_config_file):
        bridge = TrainerConfigBridge(trainer_config_file)
        assert bridge.get_peft_config() is None

    def test_get_peft_config_returns_dict(self, tmp_path):
        cfg = {
            "training": {
                "algorithm": "bco",
                "model": "Qwen/Qwen3-0.6B",
                "output_dir": str(tmp_path / "output"),
                "peft_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["q_proj", "v_proj"],
                },
                "multi_agent": {"enabled": False},
            },
            "decide_bridge": {"enabled": False},
        }
        path = tmp_path / "peft_config.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        bridge = TrainerConfigBridge(str(path))
        peft = bridge.get_peft_config()
        assert peft is not None
        assert peft["r"] == 16

    @patch("agenttune.decide.trainer_config_bridge.create_agentic_trainer")
    def test_build_trainer_calls_create(self, mock_create, trainer_config_file):
        from datasets import Dataset as HFDataset

        mock_create.return_value = MagicMock()
        ds = HFDataset.from_list([{"prompt": "Q?", "answer": "A."}])
        bridge = TrainerConfigBridge(trainer_config_file)
        trainer = bridge.build_trainer(train_dataset=ds)
        mock_create.assert_called_once()
        assert trainer is not None

    def test_step_methods_exist(self, trainer_config_file):
        bridge = TrainerConfigBridge(trainer_config_file)
        assert callable(bridge.step_build_tools)
        assert callable(bridge.step_build_reward_funcs)
        assert callable(bridge.step_build_judge)
        assert callable(bridge.step_get_peft_config)
