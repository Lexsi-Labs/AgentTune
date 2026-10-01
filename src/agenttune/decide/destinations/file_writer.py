"""File destination writer for JSON/JSONL output."""

import json
import logging
from datetime import datetime
from typing import Any

from agenttune.decide.destinations.base import DestinationWriter
from agenttune.decide.state import PipelineState

logger = logging.getLogger(__name__)


class FileWriter(DestinationWriter):
    """Write pipeline decisions to local JSON/JSONL file."""

    def __init__(self, config: dict[str, Any] = None) -> None:
        """Initialize FileWriter with optional config."""
        self.config = config or {}

    def write(self, state: PipelineState, config: dict[str, Any]) -> None:
        """
        Write decision to file.

        Args:
            state: Final pipeline state
            config: File destination configuration
        """
        # Get file config - try passed config first, fall back to self.config
        file_config = config.get("destinations", {}).get("file", {})
        if not file_config:
            # Try self.config if passed config doesn't have destinations
            file_config = self.config

        # Check if enabled
        if not file_config.get("enabled", True):
            return

        filepath = file_config.get("path", "./decisions.jsonl")
        file_format = file_config.get("format", "jsonl")

        # Prepare decision output
        decision_data = {
            "pipeline_id": state.pipeline_id,
            "template_id": state.template_id,
            "template_version": state.template_version,
            "verdict": state.verdict,
            "verdict_label": state.verdict_label,
            "confidence": state.confidence,
            "reason": state.reason,
            "step_count": state.step_count,
            "elapsed_seconds": state.elapsed_seconds,
            "timestamp": datetime.utcnow().isoformat(),
            "is_complete": state.is_complete,
            "error": state.error,
        }

        # Write to file
        try:
            # Ensure parent directory exists
            import os

            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            if file_format == "jsonl":
                # Append as single line
                with open(filepath, "a") as f:
                    f.write(json.dumps(decision_data) + "\n")
            else:
                # Write as JSON (overwrites or creates)
                with open(filepath, "w") as f:
                    json.dump(decision_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Warning: Failed to write to {filepath}: {str(e)}")
