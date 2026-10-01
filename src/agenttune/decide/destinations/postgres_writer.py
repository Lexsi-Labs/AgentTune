"""Postgres destination writer for database storage."""

import logging
from typing import Any

from agenttune.decide.destinations.base import DestinationWriter
from agenttune.decide.state import PipelineState

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import Json

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class PostgresWriter(DestinationWriter):
    """Write pipeline decisions to Postgres database."""

    def __init__(self, config: dict[str, Any] = None) -> None:
        """Initialize PostgresWriter with optional config."""
        self.config = config or {}

    def _get_connection(self, connection_string: str):
        """
        Get a database connection (for testing/mocking).

        Args:
            connection_string: Postgres connection string

        Returns:
            Database connection
        """
        return psycopg2.connect(connection_string)

    def write(self, state: PipelineState, config: dict[str, Any]) -> None:
        """
        Write decision to Postgres.

        Args:
            state: Final pipeline state
            config: Postgres destination configuration
        """
        if not PSYCOPG2_AVAILABLE:
            logger.warning("Warning: psycopg2 not installed, skipping Postgres write")
            return

        # Get postgres config - try passed config first, fall back to self.config
        pg_config = config.get("destinations", {}).get("postgres", {})
        if not pg_config:
            # Try self.config if passed config doesn't have destinations
            pg_config = self.config
        connection_string = pg_config.get("connection_string")
        table_name = pg_config.get("table", "decisions")

        if not connection_string:
            logger.warning("Warning: No postgres connection_string configured")
            return

        try:
            # Connect to database
            conn = self._get_connection(connection_string)
            cursor = conn.cursor()

            # Prepare insert statement
            insert_sql = f"""
            INSERT INTO {table_name}
            (pipeline_id, template_id, verdict, verdict_label, step_count, elapsed_seconds, is_complete, error, metadata, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """

            # Prepare metadata JSON
            metadata = {
                "stage_outputs": state.stage_outputs,
                "stage_iterations": state.stage_iterations,
                "step_history": state.step_history,
            }

            # Execute insert
            cursor.execute(
                insert_sql,
                (
                    state.pipeline_id,
                    state.template_id,
                    state.verdict,
                    state.verdict_label,
                    state.step_count,
                    state.elapsed_seconds,
                    state.is_complete,
                    state.error,
                    Json(metadata),
                ),
            )

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.warning(f"Warning: Failed to write to Postgres: {str(e)}")
