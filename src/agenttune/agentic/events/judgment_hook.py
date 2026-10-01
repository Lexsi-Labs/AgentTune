import json
import logging
import os
from typing import Any

from agenttune.decide.closed_loop.contracts import AgenticEvalResult

logger = logging.getLogger(__name__)


class JudgmentHook:
    """
    Subscribes to the rollout engine or evaluation pipeline to intercept
    judgments made by the Teacher LLM Judge. It logs these directly to a
    persistent JSONL file for distilling the small Reward Model later.
    """

    def __init__(self, output_file: str = "data/judgments.jsonl"):
        self.output_file = output_file
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def log_judgment(self, trajectory: dict[str, Any], eval_result: AgenticEvalResult):
        """
        Called after TrajectoryEvaluator produces a result.
        """
        try:
            # Reconstruct the prompt the judge saw
            from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

            # We instantiate a dummy evaluator just to rebuild the prompt for logging.
            # In a real system, the evaluator might emit the exact prompt it used.
            evaluator = TrajectoryEvaluator()
            messages, _ = evaluator._build_judge_prompt(trajectory)

            # The 'rationale' (if we had reasoning tokens) and verdict
            verdict = {
                "overall_score": eval_result.overall_judge_score,
                "goal_completion": eval_result.goal_completion_score,
                "tool_sequence_validity": eval_result.tool_sequence_validity,
                "error_recovery": eval_result.error_recovery_score,
                "intent_action_alignment": eval_result.iasa_score,
                "evidence_grounding": eval_result.egs_score,
            }

            record = {
                "trajectory_id": eval_result.trajectory_id,
                "prompt": messages,
                "verdict": verdict,
                "rotation_id": eval_result.rotation_id,
            }

            with open(self.output_file, "a") as f:
                f.write(json.dumps(record) + "\n")

        except Exception as e:
            logger.error(f"Failed to log judgment for {eval_result.trajectory_id}: {e}")
