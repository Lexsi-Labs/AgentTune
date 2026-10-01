"""Collect-mode runner: executes N episodes and writes training records."""

import json
import os
from typing import Any

from agenttune.decide.graph_runner import GraphRunner
from agenttune.decide.state import PASS_VERDICTS, PipelineState
from agenttune.decide.trainer_config_bridge import TrainerConfigBridge


class CollectRunner:
    """
    Runs N pipeline episodes and serialises training data to JSONL.

    Supported algorithms and their output format:
      bco   → {input, label (0/1), verdict, pipeline_id}
      dpo   → {input, chosen_output, rejected_output, pipeline_id}
              (requires human_feedback pairs; falls back to verdict-based labelling)
      grpo / ppo / rloo → {input, output, reward, pipeline_id}
    """

    def __init__(self, runner: GraphRunner) -> None:
        self.runner = runner
        self.config = runner.config

    async def run(self, inputs: list[str]) -> int:
        """
        Execute collect loop and write records.

        Args:
            inputs: Pool of input texts (cycled across episodes)

        Returns:
            Number of records written
        """
        collect_cfg = self.config.get("collect", {})
        episode_cfg = self.config.get("episode", {})

        algorithm = collect_cfg.get("algorithm", "bco")
        stage_id = collect_cfg.get("stage_id")
        min_samples = int(collect_cfg.get("min_samples", 50))
        output_path = collect_cfg.get("output_path", "./collected_data.jsonl")
        n_episodes = int(episode_cfg.get("n_episodes", min_samples))
        batch_size = int(episode_cfg.get("batch_size", 1))
        shuffle = bool(episode_cfg.get("shuffle_inputs", False))
        seed = episode_cfg.get("seed")

        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)

        states = await self.runner.run_episodes(
            inputs,
            n_episodes=n_episodes,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

        records = []
        for state in states:
            record = self._extract_record(state, stage_id, algorithm)
            if record is not None:
                records.append(record)
            if len(records) >= min_samples:
                break

        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        n_written = len(records)

        # auto_train: trigger TrainerConfigBridge when collect.auto_train: true
        trainer_config_path = self.config.get("training", {}).get("trainer_config")
        if collect_cfg.get("auto_train") and trainer_config_path:
            bridge = TrainerConfigBridge(trainer_config_path)
            trainer = bridge.build_trainer()
            trainer.train()

        return n_written

    def _extract_record(
        self,
        state: PipelineState,
        stage_id: str | None,
        algorithm: str,
    ) -> dict[str, Any] | None:
        """Build a single training record from a completed episode state."""
        output = state.stage_outputs.get(stage_id, {}) if stage_id else {}
        verdict = str(state.verdict or "").upper()
        is_pass = verdict in PASS_VERDICTS

        if algorithm == "bco":
            return {
                "input": state.input_text,
                "label": 1 if is_pass else 0,
                "verdict": state.verdict,
                "pipeline_id": state.pipeline_id,
            }

        elif algorithm == "dpo":
            # Prefer human-feedback pairs from audit; fall back to verdict split
            if is_pass:
                chosen, rejected = output, {}
            else:
                chosen, rejected = {}, output
            return {
                "input": state.input_text,
                "chosen_output": chosen,
                "rejected_output": rejected,
                "verdict": state.verdict,
                "pipeline_id": state.pipeline_id,
            }

        elif algorithm in {"grpo", "ppo", "rloo"}:
            # Compute reward: judge score if available, else binary
            reward = self._stage_reward(state, stage_id)
            return {
                "input": state.input_text,
                "output": output,
                "reward": reward,
                "verdict": state.verdict,
                "pipeline_id": state.pipeline_id,
            }

        return None

    def _stage_reward(self, state: PipelineState, stage_id: str | None) -> float:
        """Return the RL reward for the target stage, or binary fallback."""
        if not stage_id:
            verdict = str(state.verdict or "").upper()
            return 1.0 if verdict in PASS_VERDICTS else 0.0

        reward_cfg = state.config.get("reward", {})
        stage_cfg = reward_cfg.get("stages", {}).get(stage_id, {})
        fn = stage_cfg.get("fn", "verdict_binary")
        output = state.stage_outputs.get(stage_id, {})
        if not isinstance(output, dict):
            output = {}

        if fn == "judge_score":
            raw = (
                output.get("overall_score") or output.get("consensus_score") or output.get("score")
            )
            if raw is not None:
                return max(0.0, min(1.0, float(raw) / 10.0))

        verdict = str(state.verdict or "").upper()
        return 1.0 if verdict in PASS_VERDICTS else 0.0
