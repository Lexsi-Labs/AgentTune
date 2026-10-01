"""Custom reward functions for fraud detection pipeline."""

import logging
from typing import Any

from agenttune.agentic.rewards.llm_judge import LLMJudge
from agenttune.utils.score_logger import log_score

logger = logging.getLogger(__name__)


# Global judge instance (lazy-loaded)
_judge_instance = None


def get_judge(model_path: str = "Qwen/Qwen2.5-1.5B-Instruct") -> LLMJudge:
    """Get or create a local judge using transformers backend."""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = LLMJudge(
            backend="transformers",  # Use local model, no API
            model_path=model_path,
            system_prompt=(
                "You are a strict evaluator of fraud detection decisions. "
                "Score from 0.0 to 1.0 based on correctness and reasoning quality. "
                'Respond ONLY with JSON: {"score": <float 0-1>, "explanation": "<one sentence>"}'
            ),
        )
    return _judge_instance


def judge_reward_fn(trajectory: Any) -> float:
    """Score trajectory using local LLM judge (no API calls)."""
    judge = get_judge()
    result = judge.evaluate_trajectory(
        task="Evaluate this fraud detection decision: is it correct and well-reasoned?",
        trajectory=trajectory,
    )
    log_score(
        "judge_reward_fn",
        result.score,
        reasons=[f"LLM judge returned score={result.score}"]
        + ([f"explanation: {result.explanation}"] if getattr(result, "explanation", None) else []),
    )
    return result.score


def fraud_detection_reward(trajectory: Any) -> float:
    """
    Score a fraud detection trajectory based on decision quality.

    Returns float in [0, 1]:
    - 1.0: High confidence decision with clear reasoning
    - 0.5: Medium confidence or partial reasoning
    - 0.0: Low confidence or missing decision
    """
    reasons: list[str] = []
    try:
        # Check if we have a final decision
        if not hasattr(trajectory, "state") or not trajectory.state:
            reasons.append("trajectory has no 'state' -> 0.0")
            log_score("fraud_detection_reward", 0.0, reasons=reasons)
            return 0.0

        state = trajectory.state

        # Check for verdict (APPROVE, BLOCK, CHALLENGE)
        verdict = getattr(state, "verdict", None)
        if not verdict:
            reasons.append("state has no verdict -> 0.0")
            log_score("fraud_detection_reward", 0.0, reasons=reasons)
            return 0.0

        score = 0.5  # Base score for having a verdict
        reasons.append(f"verdict={verdict!r} present -> base score 0.5")

        # Boost score if we have high confidence
        if hasattr(state, "confidence"):
            conf = state.confidence
            if conf == "high":
                score += 0.3
                reasons.append("confidence=high -> +0.3")
            elif conf == "medium":
                score += 0.15
                reasons.append("confidence=medium -> +0.15")
            else:
                reasons.append(f"confidence={conf!r} (not high/medium) -> +0.0")
        else:
            reasons.append("state has no 'confidence' attribute -> +0.0")

        # Boost score if we have reasoning
        if hasattr(state, "reasoning") and state.reasoning:
            score += 0.15
            reasons.append("non-empty reasoning present -> +0.15")
        else:
            reasons.append("no reasoning present -> +0.0")

        # Ensure we stay in [0, 1]
        final_score = min(1.0, max(0.0, score))
        if final_score != score:
            reasons.append(f"clamped {score} to [0, 1] -> {final_score}")
        log_score(
            "fraud_detection_reward",
            final_score,
            reasons=reasons,
            components={"verdict_base": 0.5, "raw_before_clamp": score},
        )
        return final_score

    except Exception as e:
        logger.error(f"Error evaluating trajectory: {e}")
        reasons.append(f"exception during scoring: {e} -> 0.0")
        log_score("fraud_detection_reward", 0.0, reasons=reasons)
        return 0.0


def verdict_binary_reward(trajectory: Any) -> float:
    """Simple binary reward: 1.0 if approved/complete, 0.0 otherwise."""
    try:
        if not hasattr(trajectory, "state") or not trajectory.state:
            log_score("verdict_binary_reward", 0.0, reasons=["trajectory has no 'state' -> 0.0"])
            return 0.0

        verdict = getattr(trajectory.state, "verdict", None)
        if verdict in ["APPROVE", "COMPLETE"]:
            log_score(
                "verdict_binary_reward",
                1.0,
                reasons=[f"verdict={verdict!r} in APPROVE/COMPLETE -> 1.0"],
            )
            return 1.0
        elif verdict in ["BLOCK", "DENY"]:
            log_score(
                "verdict_binary_reward", 0.0, reasons=[f"verdict={verdict!r} in BLOCK/DENY -> 0.0"]
            )
            return 0.0
        else:
            log_score(
                "verdict_binary_reward",
                0.5,
                reasons=[
                    f"verdict={verdict!r} is neither approve/complete nor block/deny -> neutral 0.5"
                ],
            )
            return 0.5  # Neutral for CHALLENGE or unknown

    except Exception as e:
        log_score("verdict_binary_reward", 0.0, reasons=[f"exception during scoring: {e} -> 0.0"])
        return 0.0


def risk_score_reward(trajectory: Any) -> float:
    """Normalize the risk score from the trajectory as a reward."""
    try:
        if not hasattr(trajectory, "state") or not trajectory.state:
            log_score(
                "risk_score_reward",
                0.5,
                reasons=["trajectory has no 'state' -> neutral 0.5"],
            )
            return 0.5

        # If overall_score exists (0-10), normalize to [0, 1]
        if hasattr(trajectory.state, "overall_score"):
            # Low score (low risk) = high reward
            score = trajectory.state.overall_score
            reward = 1.0 - (score / 10.0)
            log_score(
                "risk_score_reward",
                reward,
                reasons=[f"overall_score={score} (0-10 risk) -> reward = 1 - score/10 = {reward}"],
                components={"overall_score": score},
            )
            return reward

        log_score(
            "risk_score_reward",
            0.5,
            reasons=["state has no 'overall_score' -> neutral 0.5"],
        )
        return 0.5

    except Exception as e:
        log_score(
            "risk_score_reward", 0.5, reasons=[f"exception during scoring: {e} -> neutral 0.5"]
        )
        return 0.5
