"""Audit logging and reading for decision trails."""

import json
import os
from datetime import datetime
from typing import Any

from agenttune.decide.state import PASS_VERDICTS, PipelineState


class AuditWriter:
    """Write execution traces to JSONL audit log."""

    def __init__(self, path: str = "./audit.jsonl") -> None:
        """
        Initialize audit writer.

        Args:
            path: Path to audit.jsonl file
        """
        self.path = path

    def _compute_reward(
        self,
        state: PipelineState,
        stage: dict[str, Any],
        result: dict[str, Any],
    ) -> float | None:
        """
        Compute the RL reward for a stage execution.

        Reads reward.stages.<stage_id>.fn from the pipeline config and
        returns a normalized [0.0, 1.0] scalar or None if the stage has
        no reward config.
        """
        reward_cfg = state.config.get("reward", {})
        stage_rewards = reward_cfg.get("stages", {})
        stage_id = stage["id"]

        if stage_id not in stage_rewards:
            return None

        fn = stage_rewards[stage_id].get("fn")
        output = result.get("output") or {}
        if not isinstance(output, dict):
            output = {}

        if fn == "verdict_binary":
            verdict = result.get("verdict") or output.get("verdict") or state.verdict
            return 1.0 if str(verdict or "").upper() in PASS_VERDICTS else 0.0

        elif fn == "judge_score":
            raw = (
                output.get("overall_score") or output.get("consensus_score") or output.get("score")
            )
            if raw is not None:
                return max(0.0, min(1.0, float(raw) / 10.0))
            return None

        elif fn == "rule_pass_rate":
            passed = output.get("rules_passed", 0)
            total = output.get("rules_total", 1)
            return float(passed) / max(int(total), 1)

        elif fn == "iteration_penalty":
            iteration = state.stage_iterations.get(stage_id, 1)
            max_iter = max(stage.get("max_iterations", 3), 1)
            return max(0.0, 1.0 - (iteration - 1) / max_iter)

        return None

    def log_stage(
        self, state: PipelineState, stage: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """
        Log a stage execution to audit trail.

        Args:
            state: Pipeline state
            stage: Stage configuration
            result: Stage execution result
        """
        reward = self._compute_reward(state, stage, result)
        error = result.get("error")

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_id": state.pipeline_id,
            "template_id": state.template_id,
            "template_version": state.template_version,
            "input_hash": state.input_hash,
            # Stage details
            "stage_id": stage["id"],
            "stage_type": stage["type"],
            "iteration": state.stage_iterations.get(stage["id"], 1),
            # I/O
            "input": stage.get("prompt"),
            "output": result.get("output"),
            "latency_ms": result.get("latency_ms"),
            "cost_usd": result.get("cost_usd"),
            # RL signal (None when stage has no reward config)
            "reward": reward,
            # Metadata
            "model": stage.get("model"),
            "is_retry": state.stage_iterations.get(stage["id"], 1) > 1,
            "error": error,
            # ── Fields required by closed_loop.failure_detector.FailureDetector's
            # scan_audit_log() strict schema check — aliases of the fields above, plus
            # a state snapshot, so the self-healing closed loop can actually detect
            # failures against a real DECIDE-generated log (see Known Issues: "Self-
            # healing doesn't trigger on real audit logs"). Purely additive — nothing
            # above is renamed or removed, so any existing reader of this file keeps
            # working unchanged.
            "trajectory_id": state.pipeline_id,
            "stage_name": stage["id"],
            "status": "error" if error else "ok",
            "error_details": error,
            "state_snapshot": {
                "input_text": state.input_text,
                "output": result.get("output"),
                "stage_outputs": dict(state.stage_outputs),
            },
            # low_judge_score detection reads result.score — reuse the same
            # already-normalized (0.0-1.0) reward this entry logs, when this stage
            # has a reward.stages.<id> config (typically the case for llm_judge
            # stages, whose fn is usually "judge_score").
            "result": {"score": reward} if reward is not None else None,
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _compute_episode_reward(
        self,
        state: PipelineState,
        reward_cfg: dict[str, Any],
    ) -> float | None:
        """
        Compute a weighted-mean episode reward from all stage outputs.

        Uses reward.stages.<id>.weight for aggregation when
        reward.final_fn: weighted_mean is set.
        """
        stages = reward_cfg.get("stages", {})
        weighted_sum = 0.0
        total_weight = 0.0

        for stage_id, stage_cfg in stages.items():
            weight = float(stage_cfg.get("weight", 1.0))
            fn = stage_cfg.get("fn")
            output = state.stage_outputs.get(stage_id, {})
            if not isinstance(output, dict):
                output = {}

            reward: float | None = None

            if fn == "judge_score":
                raw = (
                    output.get("overall_score")
                    or output.get("consensus_score")
                    or output.get("score")
                )
                if raw is not None:
                    reward = max(0.0, min(1.0, float(raw) / 10.0))

            elif fn == "verdict_binary":
                verdict = state.verdict
                reward = 1.0 if str(verdict or "").upper() in PASS_VERDICTS else 0.0

            elif fn == "iteration_penalty":
                iteration = state.stage_iterations.get(stage_id, 1)
                max_iter = max(stages.get(stage_id, {}).get("max_iterations", 3), 1)
                reward = max(0.0, 1.0 - (iteration - 1) / max_iter)

            elif fn == "rule_pass_rate":
                passed = output.get("rules_passed", 0)
                total = output.get("rules_total", 1)
                reward = float(passed) / max(int(total), 1)

            if reward is not None:
                weighted_sum += reward * weight
                total_weight += weight

        if total_weight == 0.0:
            return None
        return round(weighted_sum / total_weight, 4)

    def write(self, state: PipelineState) -> None:
        """
        Write final completion entry to audit log.

        Args:
            state: Final pipeline state
        """
        reward_cfg = state.config.get("reward", {})
        episode_reward: float | None = None
        if reward_cfg.get("final_fn") == "weighted_mean":
            episode_reward = self._compute_episode_reward(state, reward_cfg)

        entry = {
            "timestamp_end": state.timestamp_end,
            "pipeline_id": state.pipeline_id,
            "template_id": state.template_id,
            "verdict": state.verdict,
            "verdict_label": state.verdict_label,
            "is_complete": state.is_complete,
            "step_count": state.step_count,
            "elapsed_seconds": state.elapsed_seconds,
            "episode_reward": episode_reward,
            "error": state.error,
            # Same schema-compatibility fields as log_stage() — see the comment there.
            # stage_type deliberately isn't "tool_call"/"llm_judge" so this completion
            # line never itself trips loop_collapse/tool_crash/low_judge_score; it's
            # additive context, not a stage execution.
            "trajectory_id": state.pipeline_id,
            "stage_name": "__pipeline_complete__",
            "stage_type": "output",
            "status": "error" if state.error else "ok",
            "error_details": state.error,
            "state_snapshot": {
                "verdict": state.verdict,
                "verdict_label": state.verdict_label,
                "confidence": state.confidence,
                "reason": state.reason,
                "stage_outputs": dict(state.stage_outputs),
            },
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class AuditReader:
    """Read and extract data from JSONL audit logs."""

    def __init__(self, path: str) -> None:
        """
        Initialize audit reader.

        Args:
            path: Path to audit.jsonl file
        """
        self.path = path

    def read_all(self) -> list[dict[str, Any]]:
        """
        Read all entries from audit log.

        Returns:
            List of audit entry dictionaries
        """
        entries = []
        if not os.path.exists(self.path):
            return entries
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

    def filter_by_stage(self, stage_id: str) -> list[dict[str, Any]]:
        """
        Filter audit entries by stage_id.

        Args:
            stage_id: Stage identifier to filter by

        Returns:
            List of matching audit entries
        """
        entries = self.read_all()
        return [e for e in entries if e.get("stage_id") == stage_id]

    def filter_by_pipeline_id(self, pipeline_id: str) -> list[dict[str, Any]]:
        """
        Filter audit entries by pipeline_id.

        Args:
            pipeline_id: Pipeline identifier to filter by

        Returns:
            List of matching audit entries
        """
        entries = self.read_all()
        return [e for e in entries if e.get("pipeline_id") == pipeline_id]

    def extract_dpo_pairs(self, stage_id: str) -> list[dict[str, Any]]:
        """
        Extract DPO training pairs from human feedback.

        Args:
            stage_id: Stage identifier to extract pairs from

        Returns:
            List of DPO pair dictionaries
        """
        pairs = []

        if not os.path.exists(self.path):
            return pairs

        with open(self.path) as f:
            for line in f:
                try:
                    entry = json.loads(line)

                    # Look for human review rejections
                    if (
                        entry.get("stage_id") == stage_id
                        and entry.get("human_feedback") == "rejected"
                    ):
                        pair = {
                            "input": entry.get("input"),
                            "prompt": entry.get("input"),
                            "rejected_output": entry.get("model_output"),
                            "chosen_output": entry.get("human_output"),
                            "reason": entry.get("human_explanation"),
                        }
                        pairs.append(pair)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

        return pairs
