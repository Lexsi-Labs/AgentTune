"""Base destination writer protocol."""

from abc import ABC, abstractmethod
from typing import Any

from agenttune.decide.state import PipelineState


class DestinationWriter(ABC):
    """
    Abstract base class for destination writers.

    Defines contract for routing decisions to various destinations.
    """

    @abstractmethod
    def write(self, state: PipelineState, config: dict[str, Any]) -> None:
        """
        Write pipeline state to destination.

        Args:
            state: Final pipeline state
            config: Destination configuration
        """
        pass
