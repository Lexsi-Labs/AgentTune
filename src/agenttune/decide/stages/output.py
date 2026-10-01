"""Output stage for writing final decisions."""

import logging
from typing import Any

from agenttune.decide.stages.base import StageHandler
from agenttune.decide.state import PipelineState

logger = logging.getLogger(__name__)


class OutputStage(StageHandler):
    """
    Terminal stage for writing decision to destinations.

    Sets verdict and routes to configured destinations.
    """

    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute output stage.

        Args:
            state: Pipeline state
            stage_config: Optional stage configuration override

        Returns:
            Dictionary with verdict
        """
        # Use provided config or instance config
        config = stage_config or self.stage_config

        # Extract verdict from field reference if specified
        verdict_field = config.get("verdict_field")
        if verdict_field:
            state.verdict = self._extract_field(verdict_field, state)
        else:
            state.verdict = config.get("verdict")

        # Extract confidence from field reference if specified
        confidence_field = config.get("confidence_field")
        if confidence_field:
            state.confidence = self._extract_field(confidence_field, state)
        else:
            state.confidence = config.get("confidence")

        # Extract verdict_label if specified
        verdict_label_field = config.get("verdict_label_field")
        if verdict_label_field:
            state.verdict_label = self._extract_field(verdict_label_field, state)
        else:
            state.verdict_label = config.get("verdict_label")

        # Extract next_stage from field reference if specified
        next_stage_field = config.get("next_stage_field")
        if next_stage_field:
            state.next_stage = self._extract_field(next_stage_field, state)
        else:
            state.next_stage = config.get("next_stage")

        # Extract reason from field reference if specified
        reason_field = config.get("reason_field")
        if reason_field:
            state.reason = self._extract_field(reason_field, state)
        else:
            state.reason = config.get("reason")

        state.is_complete = True

        # Route to destinations if specified
        destinations = config.get("destinations", ["file"])
        for dest in destinations:
            try:
                await self._route_to_destination(dest, state)
            except Exception as e:
                logger.warning(f"Warning: Failed to route to {dest}: {str(e)}")

        # Build output with reason if available
        output_dict = {
            "verdict": state.verdict,
            "verdict_label": state.verdict_label,
            "confidence": state.confidence,
        }
        if state.reason:
            output_dict["reason"] = state.reason

        return {"output": output_dict}

    def _extract_field(self, field_ref: str, state: PipelineState) -> Any:
        """
        Extract value from field reference like 's2.output.decision' or literal like '"approve"'.

        Args:
            field_ref: Field reference string or literal value
            state: Pipeline state

        Returns:
            Extracted value or None
        """
        # Check if this is a literal value (wrapped in quotes)
        field_ref = field_ref.strip()
        if (field_ref.startswith('"') and field_ref.endswith('"')) or (
            field_ref.startswith("'") and field_ref.endswith("'")
        ):
            # It's a literal string value
            return field_ref[1:-1]

        # Try to parse as JSON literal (for numbers, booleans, null)
        try:
            import json

            return json.loads(field_ref)
        except (json.JSONDecodeError, ValueError):
            pass

        # Otherwise treat as field reference
        parts = field_ref.split(".")
        if len(parts) < 1:
            return None

        stage_id = parts[0]
        remaining = ".".join(parts[1:]) if len(parts) > 1 else ""

        # Get stage output
        value = state.stage_outputs.get(stage_id)
        if value is None:
            return None

        # Handle 'output.' prefix (skip it)
        if remaining.startswith("output."):
            remaining = remaining[7:]  # Remove "output."

        # Navigate through remaining path (e.g., decision)
        if remaining:
            for part in remaining.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None

        return value

    async def _route_to_destination(self, destination: str, state: PipelineState) -> None:
        """
        Route state to a specific destination.

        Args:
            destination: Destination name (e.g., "file", "postgres")
            state: Pipeline state
        """
        # Destinations are handled by DestinationRouter after pipeline completes
        # This is a placeholder for now
        pass
