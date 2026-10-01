"""Human review stage for pause-and-resume workflow."""

import asyncio
import json
import os
import uuid
from typing import Any

from agenttune.decide.stages.base import StageHandler
from agenttune.decide.state import PipelineState


class HumanReviewStage(StageHandler):
    """
    Stage for pausing pipeline and waiting for human input.

    Generates review prompt, saves task, waits for human decision.
    """

    def __init__(self, stage_config: dict[str, Any] = None) -> None:
        """Initialize HumanReviewStage."""
        super().__init__(stage_config)

    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute human review stage.

        Args:
            state: Pipeline state
            stage_config: Stage configuration (optional, uses self.stage_config if not provided)

        Returns:
            Dictionary with human decision and routing
        """
        # Generate review prompt
        config = stage_config or self.stage_config
        prompt_for_human = config.get("prompt_for_human", "")
        interpolated_prompt = self._interpolate(prompt_for_human, state)

        # Generate unique review ID
        review_id = str(uuid.uuid4())

        # Save pending review task
        await self._save_pending_review(review_id, interpolated_prompt, state)

        # Wait for human input (with timeout)
        timeout_seconds = config.get("timeout_seconds", 3600)
        try:
            decision = await self._wait_for_human_input(review_id, timeout_seconds)
        except TimeoutError:
            decision = "timeout"

        # Route based on decision
        if decision == "approved":
            goto = config.get("on_approved")
        elif decision == "denied":
            goto = config.get("on_denied")
        else:
            goto = config.get("on_failed", config.get("on_denied"))

        return {
            "output": {"decision": decision, "review_id": review_id},
            "goto": goto,
        }

    async def _save_pending_review(self, review_id: str, prompt: str, state: PipelineState) -> None:
        """
        Save a pending review task.

        Args:
            review_id: Unique review identifier
            prompt: Prompt for human reviewer
            state: Pipeline state
        """
        # Save to a simple JSON file (can be replaced with DB in production)
        review_dir = "./pending_reviews"
        os.makedirs(review_dir, exist_ok=True)

        review_data = {
            "review_id": review_id,
            "pipeline_id": state.pipeline_id,
            "template_id": state.template_id,
            "prompt": prompt,
            "timestamp": state.timestamp_start,
            "stage_outputs": state.stage_outputs,
        }

        review_path = os.path.join(review_dir, f"{review_id}.json")
        with open(review_path, "w") as f:
            json.dump(review_data, f, indent=2)

    async def _wait_for_human_input(self, review_id: str, timeout_seconds: int) -> str:
        """
        Wait for human input with timeout.

        Args:
            review_id: Unique review identifier
            timeout_seconds: Timeout in seconds

        Returns:
            Human decision (e.g., "approved", "denied")
        """
        # For now, implement a simple file-based mechanism
        # In production, this could poll a database or webhook
        decisions_dir = "./review_decisions"
        decision_file = os.path.join(decisions_dir, f"{review_id}.txt")

        # Poll for decision file
        elapsed = 0
        poll_interval = 1  # Check every 1 second

        while elapsed < timeout_seconds:
            if os.path.exists(decision_file):
                with open(decision_file) as f:
                    decision = f.read().strip().lower()
                # Clean up
                os.remove(decision_file)
                return decision

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Human review timeout after {timeout_seconds}s")
