"""Destination writers for decision routing."""

from agenttune.decide.destinations.base import DestinationWriter
from agenttune.decide.destinations.file_writer import FileWriter
from agenttune.decide.destinations.postgres_writer import PostgresWriter
from agenttune.decide.destinations.router import DestinationRouter
from agenttune.decide.destinations.webhook_sender import WebhookSender

__all__ = [
    "DestinationWriter",
    "FileWriter",
    "PostgresWriter",
    "WebhookSender",
    "DestinationRouter",
]
