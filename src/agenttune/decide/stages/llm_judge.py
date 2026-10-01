"""LLM judge stage wrapping agenttune.agentic.rewards.llm_judge."""

import asyncio
import logging
import os
import time
from typing import Any

from agenttune.decide.stages.base import StageHandler
from agenttune.decide.state import PipelineState

logger = logging.getLogger(__name__)

try:
    from agenttune.agentic.rewards.llm_judge import LLMJudge

    LLMJUDGE_AVAILABLE = True
except ImportError:
    LLMJUDGE_AVAILABLE = False


class LLMJudgeStage(StageHandler):
    """
    Stage for evaluating outputs using LLMJudge.

    Wraps agenttune.agentic.rewards.llm_judge.LLMJudge for scoring and ranking.
    """

    def __init__(self, stage_config: dict[str, Any] | None = None) -> None:
        """
        Initialize LLMJudgeStage.

        Args:
            stage_config: Stage configuration dictionary
        """
        super().__init__(stage_config)

        # Set environment variables from api_keys if available
        api_keys = self.stage_config.get("api_keys", {})
        if api_keys.get("anthropic"):
            os.environ["ANTHROPIC_API_KEY"] = api_keys["anthropic"]
        if api_keys.get("openai"):
            os.environ["OPENAI_API_KEY"] = api_keys["openai"]
        if api_keys.get("groq"):
            os.environ["GROQ_API_KEY"] = api_keys["groq"]

        # Initialize LLMJudge if available
        self.judge = None
        if LLMJUDGE_AVAILABLE:
            try:
                # Use default LLMJudge or custom judge_params from config
                judge_params = self.stage_config.get("judge_params", {})
                self.judge = LLMJudge(**judge_params)
            except Exception as e:
                # Log but don't fail - fall back to LLMCall
                logger.warning(f"Warning: Failed to initialize LLMJudge: {str(e)}")

    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute judge evaluation.

        Args:
            state: Pipeline state
            stage_config: Optional stage configuration override

        Returns:
            Dictionary with score and explanation
        """
        start_time = time.time()

        # Use provided config or instance config
        config = stage_config or self.stage_config

        # Interpolate prompt
        prompt = config.get("prompt", "")
        interpolated_prompt = self._interpolate(prompt, state)

        try:
            # Only use LLMJudge's generic trajectory rubric when the stage has no
            # output_schema of its own — otherwise every llm_judge stage with a custom
            # prompt/schema (e.g. a DECIDE decision stage) would get LLMJudge's fixed
            # overall_score/recommendation/confidence shape instead of its own declared
            # schema, and LLMJudge's default model ("gpt-4o-mini") instead of the
            # stage's configured `model:`. The branch below already does the right thing
            # (respects `model`/`prompt`/`output_schema`) — just wasn't being reached.
            use_generic_judge = (
                self.judge and LLMJUDGE_AVAILABLE and not config.get("output_schema")
            )
            if use_generic_judge:
                # Use LLMJudge.evaluate_trajectory (synchronous method)
                # Run in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                judge_result = await loop.run_in_executor(
                    None,
                    self.judge.evaluate_trajectory,
                    interpolated_prompt,  # task argument
                    "",  # trajectory argument (empty string)
                )

                # judge_result is a JudgeScore object with .score and .explanation
                score_value = judge_result.score if hasattr(judge_result, "score") else judge_result
                explanation = (
                    judge_result.explanation if hasattr(judge_result, "explanation") else ""
                )

                # Map score to recommendation and confidence
                score_0_10 = (
                    score_value * 10
                    if isinstance(score_value, float) and score_value <= 1.0
                    else float(score_value)
                )

                if score_0_10 >= 7:
                    recommendation = "APPROVE"
                    confidence = "high"
                elif score_0_10 >= 4:
                    recommendation = "CHALLENGE"
                    confidence = "medium"
                else:
                    recommendation = "BLOCK"
                    confidence = "low"

                output = {
                    "overall_score": int(score_0_10),
                    "recommendation": recommendation,
                    "confidence": confidence,
                    "reasoning": explanation or f"Fraud detection score: {score_0_10:.1f}/10",
                    "key_factors": [],
                }
            else:
                # Fall back to regular LLM call with judge-like parsing
                model = config.get("model") or config.get("judge_model")
                if not model and hasattr(self, "global_config"):
                    model = self.global_config.get("default_model")
                if not model:
                    raise ValueError("No model specified and no default_model in config")
                response = await self._call_model(model, interpolated_prompt)

                # Validate against schema
                schema = config.get("output_schema", {})
                output = self._validate_json(response, schema)

                # Ensure output has expected fields
                if "score" not in output:
                    output["score"] = None
                if "verdict" not in output:
                    output["verdict"] = "FAIL"
                output["judge_model"] = model

            return {
                "output": output,
                "latency_ms": int((time.time() - start_time) * 1000),
                "cost_usd": 0.0,
            }
        except Exception as e:
            return {
                "output": None,
                "error": str(e),
                "latency_ms": int((time.time() - start_time) * 1000),
            }
