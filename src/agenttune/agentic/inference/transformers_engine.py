import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base_engine import InferenceEngine

logger = logging.getLogger(__name__)


class TransformersEngine(InferenceEngine):
    """
    Inference Engine wrapper around HuggingFace transformers pipeline.
    This is designed to fallback to MPS/CPU natively on Macs since vLLM is not supported.
    """

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        logger.info(f"Loading TransformersEngine with model {model_name} on device {device}...")

        # Check if MPS is available for Mac
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info(f"TransformersEngine selected device: {device}")

        # Load directly rather than through pipeline auto to ensure tokenizer and config sync
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=torch.device(device) if device != "auto" else "auto",
        )

    async def generate_single(self, messages: list[dict[str, str]], **kwargs) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        max_new_tokens = kwargs.get("max_tokens", 512)

        results = self.generator(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
        return results[0]["generated_text"]

    async def generate_batch(
        self, batch_messages: list[list[dict[str, str]]], **kwargs
    ) -> list[str]:
        # Transformers pipeline isn't natively async, so we'll just process sequentially for this fallback
        outputs = []
        for msgs in batch_messages:
            try:
                res = await self.generate_single(msgs, **kwargs)
                outputs.append(res)
            except Exception as e:
                logger.error(f"TransformersEngine batch item failed: {e}")
                outputs.append("")
        return outputs

    def cleanup(self):
        del self.generator
        del self.model
        del self.tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
