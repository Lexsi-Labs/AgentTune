import abc


class InferenceEngine(abc.ABC):
    """
    Abstract base class for all inference engines (API-based, vLLM offline, etc.)
    used for LLM judging and agentic rollouts.
    """

    @abc.abstractmethod
    async def generate_single(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a single completion for a conversation."""
        pass

    @abc.abstractmethod
    async def generate_batch(
        self, batch_messages: list[list[dict[str, str]]], **kwargs
    ) -> list[str]:
        """Generate completions for a batch of conversations."""
        pass
