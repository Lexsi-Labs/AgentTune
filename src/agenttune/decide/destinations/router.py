"""Destination router for dispatching to multiple writers."""

import logging
from typing import Any

from agenttune.decide.destinations.file_writer import FileWriter
from agenttune.decide.destinations.postgres_writer import PostgresWriter
from agenttune.decide.destinations.webhook_sender import WebhookSender
from agenttune.decide.state import PipelineState

logger = logging.getLogger(__name__)


class DestinationRouter:
    """Route pipeline decisions to configured destinations."""

    # Registry of destination writers
    _writers = {
        "file": FileWriter(),
        "postgres": PostgresWriter(),
        "webhook": WebhookSender(),
    }

    @staticmethod
    def route(state: PipelineState, config: dict[str, Any]) -> None:
        """
        Route decision to all enabled destinations.

        Args:
            state: Final pipeline state
            config: Global configuration with destinations
        """
        destinations_config = config.get("destinations", {})

        # Route to each destination
        for dest_name, writer in DestinationRouter._writers.items():
            dest_config = destinations_config.get(dest_name, {})

            # Check if destination is enabled
            if not dest_config.get("enabled", False):
                continue

            try:
                writer.write(state, config)
            except Exception as e:
                # Log but don't fail the pipeline
                logger.warning(f"Warning: Destination {dest_name} failed: {str(e)}")
