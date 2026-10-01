"""
Bridge between Decide audit trails and AgentTune trainers.

Converts Decide's audit logs into training data for all 5 agentic algorithms:
- DPO/BCO: Human preference pairs from rejected feedback
- GRPO/PPO/RLOO: Trajectories from full pipeline executions

Usage:
    from agenttune.decide.training_bridge import train_from_audit

    # Train DPO from human corrections
    trainer = train_from_audit(
        audit_path="./audit.jsonl",
        stage_id="income_agent",
        algorithm="dpo",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="./runs/dpo",
    )
    results = trainer.train()
"""

import json
import os
from typing import Any

from agenttune.agentic.trajectory.dataset import Step, Trajectory, TrajectoryDataset
from agenttune.core.backend_factory import create_agentic_trainer
from agenttune.decide.audit import AuditReader


class DecideToTrainerBridge:
    """Convert Decide audit logs to AgentTune training data."""

    def __init__(self, audit_path: str):
        """
        Initialize bridge.

        Args:
            audit_path: Path to decide audit.jsonl file
        """
        self.audit_path = audit_path
        self.reader = AuditReader(audit_path)

    def extract_dpo_pairs(self, stage_id: str) -> list[dict[str, Any]]:
        """
        Extract DPO training pairs from human feedback.

        Returns list of dicts with keys:
        - prompt: The input/prompt
        - rejected: Model's original output (rejected by human)
        - chosen: Human's corrected output
        - reason: Why human rejected the output

        Args:
            stage_id: Stage to extract pairs from

        Returns:
            List of DPO pair dicts
        """
        return self.reader.extract_dpo_pairs(stage_id)

    def extract_trajectories(self, stage_id: str) -> TrajectoryDataset:
        """
        Convert stage outputs into Trajectory objects for RL training.

        Reads all executions of a stage, converts each to a trajectory
        where steps represent stage progression through a pipeline.

        Returns TrajectoryDataset compatible with TRL trainers.

        Args:
            stage_id: Stage to extract trajectories from

        Returns:
            TrajectoryDataset with Trajectory objects
        """
        trajectories = []

        if not os.path.exists(self.audit_path):
            return TrajectoryDataset([])

        with open(self.audit_path) as f:
            current_pipeline = None
            pipeline_steps = []
            pipeline_reward = 0.0
            pipeline_metadata = {}

            for line in f:
                try:
                    entry = json.loads(line)

                    # Skip non-stage entries (final completion entries)
                    if "stage_type" not in entry:
                        continue

                    # New pipeline started
                    if entry.get("pipeline_id") != current_pipeline:
                        # Save previous pipeline's trajectory
                        if current_pipeline and pipeline_steps:
                            trajectory = Trajectory(
                                task=pipeline_metadata.get("task", "Decide pipeline"),
                                steps=pipeline_steps,
                                reward=pipeline_reward,
                                metadata={
                                    "pipeline_id": current_pipeline,
                                    "stage_id": stage_id,
                                    "template_id": pipeline_metadata.get("template_id"),
                                },
                            )
                            trajectories.append(trajectory)

                        current_pipeline = entry.get("pipeline_id")
                        pipeline_steps = []
                        pipeline_metadata = {
                            "template_id": entry.get("template_id"),
                        }

                    # Skip stages not matching target
                    if entry.get("stage_id") != stage_id:
                        continue

                    # Convert stage to Step
                    step = Step(
                        step_number=entry.get("iteration", 1),
                        state=f"Stage: {entry.get('stage_id')}",
                        action={
                            "name": "llm_call",
                            "arguments": {"prompt": entry.get("input")},
                        },
                        observation=entry.get("output", ""),
                        thought=f"Iteration {entry.get('iteration', 1)}",
                        reward=None,  # Reward from judge or final verdict
                    )
                    pipeline_steps.append(step)

                    # Track reward if this is a judge stage (score 0-1 normalized)
                    if entry.get("stage_type") == "llm_judge":
                        score = entry.get("output", {})
                        if isinstance(score, dict):
                            pipeline_reward = float(score.get("score", 0.5)) / 10.0
                        else:
                            pipeline_reward = float(score) / 10.0 if score else 0.5

                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

            # Don't forget last pipeline
            if current_pipeline and pipeline_steps:
                trajectory = Trajectory(
                    task=pipeline_metadata.get("task", "Decide pipeline"),
                    steps=pipeline_steps,
                    reward=pipeline_reward,
                    metadata={
                        "pipeline_id": current_pipeline,
                        "stage_id": stage_id,
                        "template_id": pipeline_metadata.get("template_id"),
                    },
                )
                trajectories.append(trajectory)

        return TrajectoryDataset(trajectories)

    def extract_bco_labels(self, output_stage_id: str) -> list[dict[str, Any]]:
        """
        Extract binary classification labels from output stage verdicts.

        BCO (Binary Classification Optimization) trains on APPROVE/DENY verdicts.
        Returns list of dicts with keys:
        - input: The original input text
        - label: 1 for APPROVE, 0 for DENY
        - verdict_label: Original verdict string

        Args:
            output_stage_id: ID of the output stage (usually "output")

        Returns:
            List of BCO training examples
        """
        labels = []

        if not os.path.exists(self.audit_path):
            return labels

        verdict_to_label = {"APPROVE": 1, "DENY": 0, "REVIEW": 0}

        with open(self.audit_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)

                    # Look for output stage entries with verdicts
                    if entry.get("stage_id") == output_stage_id and "verdict" in entry:
                        verdict = entry.get("verdict", "").upper()
                        label = verdict_to_label.get(verdict)

                        if label is not None:
                            example = {
                                "input": entry.get("input", ""),
                                "label": label,
                                "verdict_label": verdict,
                                "confidence": entry.get(
                                    "confidence", 0.5
                                ),  # Use decide's confidence
                                "pipeline_id": entry.get("pipeline_id"),
                            }
                            labels.append(example)

                except (json.JSONDecodeError, KeyError):
                    continue

        return labels


def train_from_audit(
    audit_path: str,
    stage_id: str,
    algorithm: str,
    model: str,
    output_dir: str,
    tools: list[Any] | None = None,
    reward_funcs: list[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Train an agentic model from Decide audit logs.

    Handles all 5 algorithms:
    - DPO: Human preference pairs (rejected vs chosen)
    - BCO: Binary verdict labels (approve vs deny)
    - GRPO: Full trajectories with RL reward signal
    - PPO: Full trajectories with RL reward signal
    - RLOO: Full trajectories with leave-one-out baseline

    Args:
        audit_path: Path to decide audit.jsonl
        stage_id: Stage to extract training data from
        algorithm: One of "dpo", "bco", "grpo", "ppo", "rloo"
        model: Model name or path
        output_dir: Directory to save trained model
        tools: List of tools available during training (for GRPO/PPO/RLOO)
        reward_funcs: List of reward functions (for GRPO/PPO/RLOO)
        **kwargs: Additional args passed to create_agentic_trainer()

    Returns:
        Trainer instance (call .train() to start training)

    Raises:
        ValueError: If algorithm not supported or audit file not found
    """
    if not os.path.exists(audit_path):
        raise ValueError(f"Audit file not found: {audit_path}")

    bridge = DecideToTrainerBridge(audit_path)
    algorithm = algorithm.lower()

    if algorithm in ("dpo", "bco"):
        # Preference learning from human feedback
        if algorithm == "dpo":
            from datasets import Dataset as HFDataset

            pairs = bridge.extract_dpo_pairs(stage_id)
            if not pairs:
                raise ValueError(
                    f"No DPO pairs found in stage '{stage_id}'. "
                    "Ensure stage has human_review steps with rejections."
                )
            # extract_dpo_pairs() returns {input, prompt, chosen_output, rejected_output,
            # reason} dicts (see DecideAuditReader.extract_dpo_pairs); the DPO trainer
            # needs both a real Dataset (not a list) and the standard TRL column names.
            dataset = HFDataset.from_list(
                [
                    {
                        "prompt": pair.get("prompt") or pair.get("input"),
                        "chosen": pair.get("chosen_output"),
                        "rejected": pair.get("rejected_output"),
                    }
                    for pair in pairs
                ]
            )
        else:  # bco
            from datasets import Dataset as HFDataset

            labels = bridge.extract_bco_labels(stage_id)
            if not labels:
                raise ValueError(
                    f"No BCO labels found in stage '{stage_id}'. "
                    "Ensure output stage has APPROVE/DENY verdicts."
                )
            dataset = HFDataset.from_list(labels)

        trainer = create_agentic_trainer(
            algorithm=algorithm,
            model=model,
            train_dataset=dataset,
            output_dir=output_dir,
            **kwargs,
        )

    elif algorithm in ("grpo", "ppo", "rloo"):
        # RL training from trajectories
        dataset = bridge.extract_trajectories(stage_id)
        if len(dataset) == 0:
            raise ValueError(
                f"No trajectories found for stage '{stage_id}'. "
                "Ensure pipeline runs are recorded in audit log."
            )

        trainer = create_agentic_trainer(
            algorithm=algorithm,
            model=model,
            train_dataset=dataset,
            tools=tools or [],
            reward_funcs=reward_funcs or [],
            output_dir=output_dir,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. " "Choose from: dpo, bco, grpo, ppo, rloo"
        )

    return trainer
