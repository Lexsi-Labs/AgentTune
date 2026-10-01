import asyncio
import logging

import litellm

from .base_engine import InferenceEngine

logger = logging.getLogger(__name__)


class APIEngine(InferenceEngine):
    """
    Inference Engine wrapper around litellm for hitting external or local APIs.
    """

    def __init__(self, model_name: str, api_base: str | None = None):
        self.model_name = model_name
        self.api_base = api_base

    async def generate_single(self, messages: list[dict[str, str]], **kwargs) -> str:
        call_kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.api_base:
            call_kwargs["api_base"] = self.api_base

        call_kwargs.update(kwargs)

        try:
            response = await litellm.acompletion(**call_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"APIEngine single generation failed: {e}")
            raise e

    async def generate_batch(
        self, batch_messages: list[list[dict[str, str]]], **kwargs
    ) -> list[str]:
        # Simple concurrent API calls
        tasks = [self.generate_single(msgs, **kwargs) for msgs in batch_messages]
        # Return results (or empty strings on error if we want to handle it, but litellm errors bubble up)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"APIEngine batch item failed: {r}")
                outputs.append("")
            else:
                outputs.append(r)

        return outputs
