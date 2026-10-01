"""
Model deployment bridge between AgentTune trainers and Decide inference.

After training with AgentTune, deploy the fine-tuned model back to Decide
by updating config.yaml with the trained model path and backend type.

Usage:
    from agenttune.decide.model_deployment import deploy_trained_model

    # Deploy fine-tuned model to Decide
    deploy_trained_model(
        trained_model_path="./runs/dpo/checkpoint-final",
        config_path="./config.yaml",
        backend="transformers",  # transformers, vllm, or api
        stage_model_map={"income_agent": "transformers"}  # stage-specific
    )
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ModelDeploymentBridge:
    """Deploy trained AgentTune models to Decide inference pipelines."""

    @staticmethod
    def deploy_trained_model(
        trained_model_path: str,
        config_path: str,
        backend: str = "transformers",
        stage_model_map: dict[str, str] | None = None,
        backup: bool = True,
    ) -> None:
        """
        Deploy trained model to Decide config.

        Updates config.yaml with the trained model path and backend type.
        Supports stage-specific model assignment for A/B testing.

        Args:
            trained_model_path: Path to trained model checkpoint
                (e.g., "./runs/dpo/checkpoint-final")
            config_path: Path to config.yaml
            backend: Backend type: "transformers" (local), "vllm" (fast local),
                or "api" (OpenAI/Groq compatible)
            stage_model_map: (Optional) Dict mapping stage_id → model_path
                for stage-specific deployment. If provided, overrides global model.
            backup: Create backup of config.yaml before updating (default True)

        Raises:
            FileNotFoundError: If paths don't exist
            ValueError: If backend not supported or model path invalid

        Example:
            # Deploy globally to all stages
            deploy_trained_model(
                trained_model_path="./runs/dpo/checkpoint-final",
                config_path="./config.yaml",
                backend="transformers"
            )

            # Deploy to specific stages (A/B testing)
            deploy_trained_model(
                trained_model_path="./runs/dpo/checkpoint-final",
                config_path="./config.yaml",
                stage_model_map={
                    "income_agent": "./runs/dpo/checkpoint-final",
                    "fraud_check": "gpt-4o"  # Keep API for this stage
                }
            )
        """
        trained_path = Path(trained_model_path)
        config_file = Path(config_path)

        # Validate paths
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if not trained_path.exists():
            raise FileNotFoundError(f"Trained model not found: {trained_model_path}")

        # Validate backend
        if backend not in ("transformers", "vllm", "api"):
            raise ValueError(
                f"Unsupported backend '{backend}'. " "Choose from: transformers, vllm, api"
            )

        # Read config
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        # Create backup
        if backup:
            backup_file = config_file.with_suffix(".yaml.backup")
            with open(backup_file, "w") as f:
                yaml.dump(config, f)
            logger.info(f"✓ Backup created: {backup_file}")

        # Update global model
        config["default_model"] = str(trained_path)
        config["backend"] = backend

        # Update stage-specific models if provided
        if stage_model_map:
            stages = config.get("stages")

            # Handle both list (YAML array) and dict formats
            if isinstance(stages, list) and stages:
                # Non-empty list: find and update stages by id
                for stage in stages:
                    if isinstance(stage, dict) and stage.get("id") in stage_model_map:
                        stage["model"] = stage_model_map[stage["id"]]
            else:
                # Dict format (or empty/missing stages): use dict keyed by stage_id
                if not isinstance(stages, dict):
                    config["stages"] = {}
                    stages = config["stages"]
                for stage_id, model_path in stage_model_map.items():
                    if stage_id not in stages:
                        stages[stage_id] = {}
                    stages[stage_id]["model"] = model_path

        # Write updated config
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        logger.info(f"✓ Model deployed: {trained_path}")
        logger.info(f"✓ Backend: {backend}")
        logger.info(f"✓ Config updated: {config_path}")

    @staticmethod
    def rollback_deployment(config_path: str) -> None:
        """Restore previous model deployment from backup."""
        config_file = Path(config_path)
        backup_file = config_file.with_suffix(".yaml.backup")

        if not backup_file.exists():
            raise FileNotFoundError(f"No backup found: {backup_file}")

        import shutil

        shutil.copy(backup_file, config_file)
        logger.info(f"✓ Rollback complete: {config_path}")

    @staticmethod
    def rollback(config_path: str) -> None:
        """Alias for rollback_deployment for backward compatibility."""
        ModelDeploymentBridge.rollback_deployment(config_path)

    @staticmethod
    def get_deployment_status(config_path: str) -> dict[str, Any]:
        """Get current deployment status (model path, backend, version)."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_file) as f:
            config = yaml.safe_load(f) or {}

        return {
            "default_model": config.get("default_model"),
            "backend": config.get("backend"),
            "stage_models": config.get("stages", {}),
            "model_exists": Path(config.get("default_model", "")).exists(),
        }


def deploy_trained_model(
    trained_model_path: str,
    config_path: str,
    backend: str = "transformers",
    stage_model_map: dict[str, str] | None = None,
    backup: bool = True,
) -> None:
    """
    Deploy trained AgentTune model to Decide inference.

    Convenience function wrapping ModelDeploymentBridge.deploy_trained_model().

    Args:
        trained_model_path: Path to trained model
        config_path: Path to decide config.yaml
        backend: "transformers", "vllm", or "api"
        stage_model_map: Optional stage-specific model assignments
        backup: Create backup before updating

    Example:
        from agenttune.decide.model_deployment import deploy_trained_model

        deploy_trained_model(
            trained_model_path="./runs/dpo/checkpoint-final",
            config_path="./config.yaml",
            backend="transformers"
        )
    """
    ModelDeploymentBridge.deploy_trained_model(
        trained_model_path, config_path, backend, stage_model_map, backup
    )
