"""
Reinforcement Learning specific metrics.
"""

from typing import Any

import numpy as np

from .base import Metric


class KLDivergenceMetric(Metric):
    """
    Computes KL Divergence aggregation.
    """

    def __init__(self):
        super().__init__("kl_divergence")

    @property
    def requires_generation(self) -> bool:
        return False

    def compute(self, predictions: list[Any], references: list[Any], **kwargs) -> dict[str, float]:
        if predictions and isinstance(predictions[0], float | int):
            return {"kl_divergence": float(np.mean(predictions))}
        return {"kl_divergence": 0.0}


class RewardAccuracyMetric(Metric):
    """
    Computes accuracy of the reward model on chosen vs rejected pairs.
    """

    def __init__(self):
        super().__init__("reward_accuracy")

    @property
    def requires_generation(self) -> bool:
        return False

    def compute(self, predictions: list[Any], references: list[Any], **kwargs) -> dict[str, float]:
        if not predictions:
            return {"reward_accuracy": 0.0}

        if isinstance(predictions[0], tuple | list) and len(predictions[0]) == 2:
            correct = sum(1 for c, r in predictions if c > r)
            return {"reward_accuracy": float(correct / len(predictions))}

        return {"reward_accuracy": 0.0}


class PolicyEntropyMetric(Metric):
    """Computes the entropy of the policy."""

    def __init__(self):
        super().__init__("policy_entropy")

    @property
    def requires_generation(self) -> bool:
        return False

    def compute(self, predictions: list[Any], references: list[Any], **kwargs) -> dict[str, float]:
        if predictions and isinstance(predictions[0], float | int):
            return {"policy_entropy": float(np.mean(predictions))}
        return {"policy_entropy": 0.0}
