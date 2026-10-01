"""Stage handler dispatcher with registry pattern."""

from typing import Any


class StageExecutor:
    """
    Registry and dispatcher for stage handlers.

    Manages stage type to handler class mapping.
    """

    _handlers: dict[str, type[Any]] = {}

    @classmethod
    def register(cls, stage_type: str, handler_class: type[Any]) -> None:
        """
        Register a stage handler class.

        Args:
            stage_type: Stage type identifier (e.g., "llm_call")
            handler_class: Stage handler class
        """
        cls._handlers[stage_type] = handler_class

    @classmethod
    def get(cls, stage_type: str) -> type[Any]:
        """
        Get a registered stage handler class.

        Args:
            stage_type: Stage type identifier

        Returns:
            Stage handler class

        Raises:
            ValueError: If stage type is not registered
        """
        if stage_type not in cls._handlers:
            raise ValueError(
                f"Unknown stage type: {stage_type}. "
                f"Registered types: {list(cls._handlers.keys())}"
            )
        return cls._handlers[stage_type]
