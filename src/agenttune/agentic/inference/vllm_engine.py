import asyncio
import logging

from .base_engine import InferenceEngine

logger = logging.getLogger(__name__)


class OfflineVLLMEngine(InferenceEngine):
    """
    Inference Engine using vLLM's offline batch API for blazing fast local execution.
    WARNING: Running this engine allocates a large GPU KV cache. You must destroy the
    engine and empty the CUDA cache before running TRL trainers in the same process.
    """

    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        self.model_name = model_name
        self.llm = None

        from agenttune.utils.optional import require_vllm

        require_vllm()
        try:
            from vllm import LLM

            # Initialize the heavy vLLM engine
            logger.info(f"Initializing OfflineVLLMEngine with model {model_name}...")
            self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
        except ImportError:
            logger.error("vllm is not installed. OfflineVLLMEngine cannot be initialized.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise e

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """
        Converts the list of message dicts into the model's expected string format.
        Ideally uses the tokenizer's chat template.
        """
        if not self.llm:
            return ""

        tokenizer = self.llm.get_tokenizer()
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Fallback simplistic template if tokenizer lacks one
        prompt = ""
        for m in messages:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    async def generate_single(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        Since vLLM is highly optimized for batching, generating a single sequence
        offline is just a batch of size 1. Note: This blocks the event loop unless
        run in an executor, but we assume offline batching is the primary use case.
        """
        return (await self.generate_batch([messages], **kwargs))[0]

    async def generate_batch(
        self, batch_messages: list[list[dict[str, str]]], **kwargs
    ) -> list[str]:
        if not self.llm:
            raise RuntimeError("vLLM engine is not initialized.")

        from vllm import SamplingParams

        prompts = [self._apply_chat_template(msgs) for msgs in batch_messages]

        # Extract vLLM specific kwargs
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        # In a purely sync offline script, this blocks.
        # If this is called in an async loop, we should ideally run it in a threadpool
        # or use AsyncLLMEngine. For offline data collection, blocking is often acceptable
        # if the whole batch is processed at once.
        loop = asyncio.get_running_loop()

        def _run_vllm():
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            # llm.generate processes the entire list of prompts efficiently
            outputs = self.llm.generate(prompts, sampling_params)
            return [out.outputs[0].text for out in outputs]

        # Run in executor to not freeze the asyncio loop
        results = await loop.run_in_executor(None, _run_vllm)
        return results

    def cleanup(self):
        """
        Frees the massive KV cache. MUST be called before starting SFT/DPO in the same script.
        """
        if self.llm:
            del self.llm
            self.llm = None
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            logger.info("vLLM engine destroyed and CUDA cache cleared.")
