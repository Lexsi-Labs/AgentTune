"""Webhook destination sender for HTTP delivery."""

import logging
from typing import Any

import requests

from agenttune.decide.destinations.base import DestinationWriter
from agenttune.decide.state import PipelineState

logger = logging.getLogger(__name__)


class WebhookSender(DestinationWriter):
    """Send pipeline decisions to HTTP webhook endpoint."""

    def __init__(self, config: dict[str, Any] = None) -> None:
        """Initialize WebhookSender with optional config."""
        self.config = config or {}

    def write(self, state: PipelineState, config: dict[str, Any]) -> None:
        """
        Send decision to webhook.

        Args:
            state: Final pipeline state
            config: Webhook destination configuration
        """
        # Get webhook config - try passed config first, fall back to self.config
        webhook_config = config.get("destinations", {}).get("webhook", {})
        if not webhook_config:
            # Try self.config if passed config doesn't have destinations
            webhook_config = self.config

        # Check if enabled
        if not webhook_config.get("enabled", True):
            return

        webhook_url = webhook_config.get("url")
        method = webhook_config.get("method", "POST").upper()
        headers = webhook_config.get("headers", {})

        if not webhook_url:
            logger.warning("Warning: No webhook URL configured")
            return

        # Prepare payload
        payload = {
            "pipeline_id": state.pipeline_id,
            "template_id": state.template_id,
            "verdict": state.verdict,
            "verdict_label": state.verdict_label,
            "confidence": state.confidence,
            "reason": state.reason,
            "step_count": state.step_count,
            "elapsed_seconds": state.elapsed_seconds,
            "is_complete": state.is_complete,
            "error": state.error,
            "stage_outputs": state.stage_outputs,
        }

        # Ensure JSON content-type
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        try:
            response = requests.request(
                method,
                webhook_url,
                json=payload,
                headers=headers,
                timeout=30,
            )
            if response.status_code >= 300:
                logger.warning(f"Warning: Webhook returned status {response.status_code}")
            else:
                logger.info(f"Webhook sent successfully: {webhook_url}")
        except Exception as e:
            logger.warning(f"Warning: Failed to send webhook: {str(e)}")
