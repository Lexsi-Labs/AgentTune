import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class BehavioralDiversityMonitor:
    """
    Monitors agent trajectories post-retraining to detect behavioral collapse,
    such as becoming overly predictable or repeating the same exact tool sequences.
    """

    def __init__(self, history_size: int = 100, diversity_threshold: float = 0.3):
        # We store hashes or summaries of the recent tool sequences
        self.history_size = history_size
        self.trajectory_history: deque[str] = deque(maxlen=history_size)
        self.diversity_threshold = diversity_threshold
        self.collapse_alert_active = False

    def _extract_behavior_signature(self, trajectory: dict[str, Any]) -> str:
        """
        Creates a signature for a trajectory based on the sequence of tools called.
        """
        tools_called = trajectory.get("tool_calls", [])
        return "->".join(tools_called)

    def observe_trajectory(self, trajectory: dict[str, Any]):
        """
        Record a new trajectory and check for diversity collapse.
        """
        signature = self._extract_behavior_signature(trajectory)
        self.trajectory_history.append(signature)

    def check_diversity(self) -> bool:
        """
        Checks if the diversity has dropped below the threshold.
        Returns False if diversity is acceptable, True if an alert should be raised (collapse).
        """
        if len(self.trajectory_history) < min(10, self.history_size):
            # Not enough data to assess diversity yet
            return False

        # Calculate uniqueness ratio
        unique_signatures = len(set(self.trajectory_history))
        diversity_score = unique_signatures / len(self.trajectory_history)

        if diversity_score < self.diversity_threshold:
            if not self.collapse_alert_active:
                logger.warning(
                    f"BEHAVIORAL COLLAPSE ALERT: Diversity score dropped to {diversity_score:.2f} "
                    f"(Threshold: {self.diversity_threshold}). The agent is showing highly repetitive behavior."
                )
                self.collapse_alert_active = True
            return True
        else:
            if self.collapse_alert_active:
                logger.info(f"Behavioral diversity recovered. Current score: {diversity_score:.2f}")
                self.collapse_alert_active = False
            return False
