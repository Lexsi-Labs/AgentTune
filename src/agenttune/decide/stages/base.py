"""Base stage handler protocol and utilities."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import litellm

from agenttune.decide.state import PipelineState


class StageHandler(ABC):
    """
    Abstract base class for stage handlers.

    Defines the contract for all stage types.
    """

    _engine_cache: dict[str, Any] = {}

    def __init__(self, stage_config: dict[str, Any] | None = None) -> None:
        """
        Initialize stage handler.

        Args:
            stage_config: Stage configuration dictionary with id, type, etc.
        """
        self.stage_config = stage_config or {}
        self.config = self.stage_config  # For backward compatibility
        self.current_inject: str | None = None
        self._shared_engine: Any | None = None
        self.global_config: dict[str, Any] = {}

    @abstractmethod
    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute stage logic.

        Args:
            state: Current pipeline state
            stage_config: Optional stage configuration override

        Returns:
            Dictionary with execution result
        """
        pass

    def _interpolate(self, prompt: str, state: PipelineState) -> str:
        """
        Interpolate context variables in prompt.

        Supports syntax:
        - {input_text}: Original input
        - {stage_id.output.field}: Output from specific stage
        - {stage_id.iteration}: Iteration count for stage
        - {pipeline.step_count}: Total steps executed
        - {pipeline.elapsed_seconds}: Elapsed time
        - {feedback}: Injected feedback context

        Args:
            prompt: Prompt with placeholders
            state: Pipeline state

        Returns:
            Interpolated prompt string
        """

        def replacer(match):
            var = match.group(1)

            if var == "input_text":
                return str(state.input_text)

            elif var == "feedback":
                # Special: injected feedback from loop
                return str(self.current_inject or "")

            elif var.startswith("pipeline."):
                key = var.split(".", 1)[1]
                if key == "step_count":
                    return str(state.step_count)
                elif key == "elapsed_seconds":
                    # Calculate elapsed time
                    from datetime import datetime

                    if state.timestamp_start:
                        start = datetime.fromisoformat(state.timestamp_start)
                        now = datetime.utcnow()
                        elapsed = (now - start).total_seconds()
                        return str(elapsed)
                    return "0"

            elif "." in var:
                # {stage_id.output.field} or {stage_id.iteration}
                parts = var.split(".", 1)
                stage_id = parts[0]
                rest = parts[1] if len(parts) > 1 else ""

                if rest == "iteration":
                    return str(state.stage_iterations.get(stage_id, 0))
                elif rest.startswith("output."):
                    # Navigate nested output: {stage_id.output.field.subfield}
                    output = state.stage_outputs.get(stage_id, {})
                    remaining = rest.split(".", 1)[1]  # Remove "output."

                    for part in remaining.split("."):
                        if isinstance(output, dict):
                            output = output.get(part)
                        else:
                            output = None
                            break

                    return str(output) if output is not None else ""
                else:
                    # Try {stage_id.field} directly
                    output = state.stage_outputs.get(stage_id, {})
                    remaining_parts = rest.split(".")
                    for part in remaining_parts:
                        if isinstance(output, dict):
                            output = output.get(part)
                        else:
                            output = None
                            break
                    return str(output) if output is not None else ""

            # No match, return as-is
            return f"{{{var}}}"

        return re.sub(r"\{([^}]+)\}", replacer, prompt)

    async def _call_model(self, model: str, prompt: str) -> str:
        """
        Call an LLM with the given prompt using the rollout engine.
        Supports transformers (local), vLLM (server), and API-based models.

        Detects model type:
        - Transformers models: Any HuggingFace model ID (contains "/" or matches known prefixes)
        - vLLM models: Can be configured via backend detection
        - API models: Claude, GPT, Groq, etc. (routed to litellm)

        Args:
            model: Model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct", "gpt-4", "meta-llama/Llama-2-7b")
            prompt: Prompt text

        Returns:
            Model response
        """
        try:
            # Detect model type based on identifier format
            is_api_model = self._is_api_model(model)
            is_vllm_model = self._is_vllm_model(model)

            if not is_api_model:
                # Use rollout engine for local/server models (transformers, vLLM, etc.)
                try:
                    from agenttune.agentic.rollout_engines.rollout_factory import (
                        create_rollout_engine,
                        create_rollout_fn,
                    )

                    # Use injected shared engine if available
                    if self._shared_engine is not None:
                        engine = self._shared_engine
                    else:
                        # Detect backend automatically
                        backend = "vllm" if is_vllm_model else "transformers"
                        cache_key = f"{backend}:{model}"

                        if cache_key not in StageHandler._engine_cache:
                            StageHandler._engine_cache[cache_key] = create_rollout_engine(
                                backend=backend,
                                model_path=model,
                                torch_dtype="bfloat16",
                                device_map="auto",
                            )
                        engine = StageHandler._engine_cache[cache_key]

                    rollout_fn = create_rollout_fn(rollout_engine=engine, max_steps=1, tools=None)

                    result = rollout_fn([prompt])
                    if result and "responses" in result and result["responses"]:
                        return result["responses"][0]
                    else:
                        raise RuntimeError(f"No response from rollout engine for {model}")

                except ImportError:
                    # Fall back to litellm if rollout engine not available
                    return await self._call_model_litellm(model, prompt)
            else:
                # Use litellm for API-based closed-source models
                return await self._call_model_litellm(model, prompt)

        except Exception as e:
            raise RuntimeError(f"LLM call failed for model {model}: {str(e)}")  # noqa: B904

    def _is_api_model(self, model: str) -> bool:
        """
        Detect if model is an API-based model that should use litellm.

        Returns True for Claude, GPT, Groq, and other closed-source API models.
        Returns False for HuggingFace models and local model paths.

        Args:
            model: Model identifier string

        Returns:
            True if API-based model, False if local/open-source
        """
        # Known API model identifiers
        api_prefixes = (
            "gpt-",  # OpenAI
            "claude-",  # Anthropic
            "groq-",  # Groq (alternative)
            "grok-",  # xAI Grok
            "gemini-",  # Google Gemini
            "command",  # Cohere
            "j2-",  # AI21 Labs
            "palm-",  # Google PaLM (legacy)
            "text-davinci",  # OpenAI legacy
            "text-curie",  # OpenAI legacy
            "together",  # Together AI
            "replicate",  # Replicate
            "openrouter",  # OpenRouter
        )

        # Groq and OpenAI via groq prefix
        if model.lower().startswith("groq/"):
            return True

        # Check if it's a known API model prefix
        if any(model.lower().startswith(prefix) for prefix in api_prefixes):
            return True

        # Treat anything with "/" as HuggingFace (local model)
        if "/" in model:
            return False

        # Known open-source model names (no "/" in the ID)
        openai_prefixes = (
            "meta-llama",  # LLaMA
            "llama-",  # LLaMA (alternative naming)
            "llama2",  # LLaMA 2
            "mistral",  # Mistral
            "mistralai",  # Mistral (alternate)
            "qwen",  # Qwen (can be "qwen" or "Qwen/...")
            "baichuan",  # Baichuan
            "bigcode",  # BigCode
            "mosaicml",  # MosaicML MPT
            "tiiuae",  # Falcon
            "allenai",  # AllenAI
            "databricks",  # Databricks
            "nvidia",  # NVIDIA
            "xlnet",  # XLNet
            "t5-",  # T5
            "t5_",  # T5 (underscore)
            "flan-t5",  # Flan-T5
            "flan_t5",  # Flan-T5 (underscore)
            "gpt2",  # GPT2 (open-source, from OpenAI)
            "roberta",  # RoBERTa
            "bert-",  # BERT variants
            "distilbert",  # DistilBERT
            "electra",  # ELECTRA
            "bloom",  # BigScience BLOOM
            "opt-",  # Meta OPT
            "mpt-",  # MPT
            "mpt_",  # MPT (underscore)
            "dolly",  # Databricks Dolly
            "stablelm",  # Stability StableLM
            "open-llama",  # OpenLLaMA
            "openllama",  # OpenLLaMA (no dash)
            "airbytehq",  # Airbyte
            "cerebras",  # Cerebras
            "palmyra",  # Airbyte Palmyra
            "openchat",  # OpenChat
            "neural-chat",  # Intel Neural Chat
            "codellama",  # CodeLlama
            "code-llama",  # CodeLlama (with dash)
            "phind",  # Phind
            "orca",  # Orca
            "solar",  # Solar
            "zephyr",  # Zephyr
            "starling",  # Starling
            "neural",  # Neural
        )

        if any(model.lower().startswith(prefix) for prefix in openai_prefixes):
            return False

        # Default: treat unknown models as API models (safer fallback)
        return True

    def _is_vllm_model(self, model: str) -> bool:
        """
        Detect if model should use vLLM backend instead of transformers.

        Args:
            model: Model identifier string

        Returns:
            True if vLLM should be used, False for transformers
        """
        # Check for vllm-specific indicators (custom vLLM endpoint, etc.)
        if "vllm" in model.lower() or "localhost:" in model or "127.0.0.1:" in model:
            return True

        # Check environment variable override
        import os

        if os.getenv("AGENTTUNE_USE_VLLM", "").lower() == "true":
            return True

        # Default to transformers for HuggingFace models
        return False

    async def _call_model_litellm(self, model: str, prompt: str) -> str:
        """
        Fallback to litellm for API-based models.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-opus-4-1")
            prompt: Prompt text

        Returns:
            Model response
        """
        try:
            response = await litellm.acompletion(
                model=model, messages=[{"role": "user", "content": prompt}]
            )

            if isinstance(response, dict):
                return response["choices"][0]["message"]["content"]
            else:
                return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM call failed for model {model}: {str(e)}")  # noqa: B904

    def _validate_json(self, response: str, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and parse JSON response against schema.

        Args:
            response: JSON response string
            schema: JSON schema for validation

        Returns:
            Parsed and validated JSON object

        Raises:
            ValueError: If JSON is invalid or doesn't match schema
        """
        import re

        # Extract JSON from response (handle markdown code blocks)
        json_str = response.strip()

        # Try to extract JSON from markdown code blocks
        # Pattern: ```(json)?\n...\n``` or similar variations
        match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            # Fallback: remove opening and closing ``` if present
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            elif json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

        try:
            parsed = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")  # noqa: B904

        # Basic schema validation: check required fields if specified
        if schema and "required" in schema:
            required_fields = schema["required"]
            if isinstance(parsed, dict):
                missing = [f for f in required_fields if f not in parsed]
                if missing:
                    raise ValueError(f"Missing required fields: {missing}")

        return parsed


def flatten_stage_outputs(stage_outputs: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested stage outputs for evaluator.

    Converts {"s0": {"score": 0.7}} to {"s0.output.score": 0.7}.
    Used by _eval_condition, RulesStage, and RouterStage.

    Args:
        stage_outputs: Nested stage outputs dictionary

    Returns:
        Flattened context dict
    """
    flattened = {}
    # First pass: stage-prefixed keys
    for stage_id, output in stage_outputs.items():
        if isinstance(output, dict):
            for key, value in output.items():
                flattened[f"{stage_id}.output.{key}"] = value
                flattened[f"{stage_id}.{key}"] = value
        else:
            flattened[f"{stage_id}.output"] = output
            flattened[stage_id] = output
    # Second pass: add top-level short names (last stage with a given key wins)
    for stage_id, output in stage_outputs.items():
        if isinstance(output, dict):
            for key, value in output.items():
                flattened[key] = value
    return flattened
