import logging

from agenttune.utils.score_logger import log_score

logger = logging.getLogger(__name__)


class DistilledJudge:
    """
    A local Reward Model distilled from the LLM Judge.
    Fulfills the reward_funcs interface for GRPO/PPO.
    """

    def __init__(self, model_path: str = "custom_reward_model"):
        self.model_path = model_path
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline

            # Text classification pipeline for regression head
            self.pipeline = pipeline(
                "text-classification", model=self.model_path, device_map="auto"
            )
        except ImportError:
            logger.warning("transformers not installed. DistilledJudge running in dummy mode.")
        except Exception as e:
            logger.error(f"Failed to load DistilledJudge from {self.model_path}: {e}")

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        if not self.pipeline:
            # Fallback if model isn't loaded
            log_score(
                "distilled_judge_reward",
                0.0,
                reasons=[
                    "pipeline not loaded (transformers missing or load failed) -> 0.0 for every completion"
                ],
                meta={"n": len(prompts)},
            )
            return [0.0] * len(prompts)

        scores = []
        for i, (prompt, comp) in enumerate(zip(prompts, completions, strict=False)):
            try:
                # The prompt structure should match the SFT training phase
                input_text = prompt + comp
                # Max length truncation should ideally be handled by pipeline config
                result = self.pipeline(input_text, truncation=True, max_length=2048)
                # Pipeline returns [{'label': 'LABEL_0', 'score': 0.85}] for single-label regression
                # Or just the raw float if configured purely for regression.
                if isinstance(result, list) and len(result) > 0:
                    score = float(result[0].get("score", 0.0))
                    reason = f"distilled reward model output score={score} (label={result[0].get('label')})"
                else:
                    score = 0.0
                    reason = "pipeline returned an empty result -> 0.0"
                log_score("distilled_judge_reward", score, reasons=[reason], meta={"index": i})
                scores.append(score)
            except Exception as e:
                logger.error(f"Inference error in DistilledJudge: {e}")
                log_score(
                    "distilled_judge_reward",
                    0.0,
                    reasons=[f"inference error: {e} -> 0.0"],
                    meta={"index": i},
                )
                scores.append(0.0)

        return scores


# Registry wrapper
def distilled_judge_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    model_path = kwargs.get("distilled_judge_path", "custom_reward_model")
    # In production, we'd cache the instance to avoid reloading.
    # For now, instantiate once per batch (acceptable if pipeline handles caching under the hood,
    # but practically we should cache the class instance globally).
    if not hasattr(distilled_judge_reward, "_judge"):
        distilled_judge_reward._judge = DistilledJudge(model_path)

    return distilled_judge_reward._judge(prompts, completions, **kwargs)
