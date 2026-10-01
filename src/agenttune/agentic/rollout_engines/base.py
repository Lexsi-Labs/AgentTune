from abc import ABC, abstractmethod
from typing import Any


class RolloutEngine(ABC):
    """
    Base class for all rollout engines.
    Subclass this to add a new inference backend.
    """

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        tools: list[Any],
        gen_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Returns:
            {
                'completions': List[str],
                'logprobs':    List[Any],   # None if backend doesn't support
                'metadata':    Dict
            }
        """
        pass

    def is_available(self) -> bool:
        """Override to add availability check."""
        return True
