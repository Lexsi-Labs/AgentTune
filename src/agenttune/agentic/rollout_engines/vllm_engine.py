"""
VLLMRolloutEngine — wraps TRL's VLLMGeneration exactly the way GRPOTrainer does.
"""

import logging
from typing import Any

from .base import RolloutEngine
from .rollout_factory import (
    _keep_tool_results,
    coerce_tool_call_arguments_to_dict,
    fold_tool_messages_into_user,
)

logger = logging.getLogger(__name__)


class VLLMRolloutEngine(RolloutEngine):
    """
    Fast rollout engine wrapping TRL's VLLMGeneration backend.

    Args:
        model_path: HuggingFace model ID or local path.
        gpu_memory_utilization: Fraction of GPU VRAM for KV cache (default 0.9).
        tensor_parallel_size: Number of GPUs for tensor parallelism (default 1).
        max_model_len: Override model's max context length (default None = auto).
        max_num_seqs: Max concurrent sequences (default 32).
        max_completion_length: Max new tokens per generation (default 512).
        temperature: Sampling temperature (default 0.9).
        top_p: Top-p nucleus sampling (default 1.0).
        top_k: Top-k sampling, -1 = disabled (default -1).
        min_p: Min-p sampling, 0.0 = disabled (default 0.0).
        enable_sleep_mode: Sleep between generate() calls to free VRAM (default False).
        dtype: Model dtype string e.g. "bfloat16" (default None = auto).
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int = 32,
        max_completion_length: int = 512,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        enable_sleep_mode: bool = False,
        dtype: str | None = None,
        **model_init_kwargs,
    ):
        try:
            from trl.generation.vllm_generation import VLLMGeneration
        except ImportError as e:
            raise ImportError(  # noqa: B904
                "TRL is required for VLLMRolloutEngine. Install with: pip install trl\n"
                f"Original error: {e}"
            )

        # Compatibility shim: vllm's bundled Pixtral model code (used by any
        # architecture with a Pixtral-style vision tower, e.g. Mistral3's)
        # does `from transformers...modeling_pixtral import
        # PixtralRotaryEmbedding`, a class newer transformers versions
        # renamed to PixtralVisionRotaryEmbedding (same constructor
        # signature -- confirmed by reading both, not assumed). Without
        # this, constructing the vLLM engine for ANY Pixtral-vision model
        # fails with ImportError before the engine even starts. Applied
        # here (not by installing an older transformers) since this repo
        # needs the newer transformers for other model support (e.g.
        # Ministral3Config).
        try:
            import transformers.models.pixtral.modeling_pixtral as _pixtral_mod

            if not hasattr(_pixtral_mod, "PixtralRotaryEmbedding") and hasattr(
                _pixtral_mod, "PixtralVisionRotaryEmbedding"
            ):
                _pixtral_mod.PixtralRotaryEmbedding = _pixtral_mod.PixtralVisionRotaryEmbedding
        except ImportError:
            pass

        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(f"transformers is required: {e}")  # noqa: B904

        from .rollout_factory import load_causal_or_multimodal_model

        if dtype is not None:
            model_init_kwargs["torch_dtype"] = getattr(torch, dtype)

        logger.info(f"[VLLMRolloutEngine] Loading model: {model_path}")
        # Tries AutoModelForCausalLM, then the auto classes real
        # tool-calling checkpoints that ship as vision+text wrappers
        # actually use (e.g. Ministral-3's Mistral3ForConditionalGeneration)
        # -- see load_causal_or_multimodal_model's docstring.
        model = load_causal_or_multimodal_model(model_path, **model_init_kwargs)

        try:
            processing_class = AutoProcessor.from_pretrained(
                model_path,
                truncation_side="left",
                padding_side="left",
            )
        except OSError:
            # Some text-only checkpoints (e.g. google/gemma-3-1b-it) still
            # declare a processor class that expects an image processor in
            # their config, but don't ship one -- AutoProcessor fails with
            # "Can't load image processor" even though the model itself is
            # plain text. A tokenizer is all a text-only model needs.
            from transformers import AutoTokenizer

            processing_class = AutoTokenizer.from_pretrained(
                model_path,
                truncation_side="left",
                padding_side="left",
            )

        from transformers import ProcessorMixin

        tokenizer = (
            processing_class.tokenizer
            if isinstance(processing_class, ProcessorMixin)
            else processing_class
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            from accelerate import Accelerator
        except ImportError as e:
            raise ImportError(f"accelerate is required: {e}")  # noqa: B904

        accelerator = Accelerator()

        # ── One-time TRL setup (mirrors GRPOTrainer.__init__) ─────────────────
        self._chat_template = None
        self._can_parse = False

        native_tmpl = getattr(tokenizer, "chat_template", None)
        try:
            import transformers
            from packaging.version import Version
            from trl.chat_template_utils import add_response_schema

            if Version(transformers.__version__) >= Version("5.0.0"):
                if not getattr(tokenizer, "response_schema", None):
                    processing_class = add_response_schema(processing_class)
                    tokenizer = (
                        processing_class.tokenizer
                        if isinstance(processing_class, ProcessorMixin)
                        else processing_class
                    )
                self._can_parse = getattr(tokenizer, "response_schema", None) is not None
                if native_tmpl is not None:
                    tokenizer.chat_template = native_tmpl
                    if hasattr(processing_class, "chat_template"):
                        processing_class.chat_template = native_tmpl
                    inner = getattr(processing_class, "tokenizer", None)
                    if inner is not None and native_tmpl is not None:
                        inner.chat_template = native_tmpl
        except (ImportError, Exception):
            pass
        # ──────────────────────────────────────────────────────────────────────

        logger.info("[VLLMRolloutEngine] Initialising VLLMGeneration (colocate mode)...")
        # trl[vllm]==1.7.1 (the version this repo pins) has no
        # "is_fsdp_enabled" parameter on VLLMGeneration.__init__ -- passing
        # it raised TypeError unconditionally, so this engine could never
        # be constructed at all. Confirmed against the installed package's
        # actual signature, not assumed.
        self.vllm_generation = VLLMGeneration(
            model=model,
            accelerator=accelerator,
            processing_class=processing_class,
            mode="colocate",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_length=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_sleep_mode=enable_sleep_mode,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_completion_length=max_completion_length,
            repetition_penalty=1.0,
            structured_outputs_regex=None,
            # chat_template=None,
            # chat_template_kwargs={},
            # tools=[],
            # rollout_func=None,
        )

        logger.info("[VLLMRolloutEngine] Ready.")
        del model

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_tokenizer(self):
        from transformers import ProcessorMixin

        pc = self.vllm_generation.processing_class
        return pc.tokenizer if isinstance(pc, ProcessorMixin) else pc

    def _apply_template(
        self,
        conv: list[dict],
        tools: list[dict] | None,
        enable_thinking: bool,
    ) -> str:
        """Apply chat template with graceful fallback."""
        import jinja2

        tokenizer = self._get_tokenizer()
        conv = _keep_tool_results(tokenizer, conv, self._chat_template)

        def _render(c):
            common = {
                "conversation": c,
                "tools": tools,
                "chat_template": self._chat_template,
                "add_generation_prompt": True,
                "tokenize": False,
            }
            # Always pass enable_thinking explicitly, even when False: some
            # templates (e.g. Qwen3's) treat an OMITTED enable_thinking as "use
            # the model's default" (thinking on), not as False -- only an
            # explicit False pre-fills the empty "<think></think>" that
            # actually suppresses it. Omitting the kwarg here (the previous
            # behaviour whenever enable_thinking was False) silently left
            # thinking mode on regardless of what the caller asked for.
            #
            # The nested TypeError retries below only fire when that's
            # actually what the error is about (checked via the message) --
            # not when a same-named TypeError comes from inside the jinja
            # template itself (e.g. Qwen3.5's below), which retrying the
            # same way would just reproduce identically.
            try:
                return tokenizer.apply_chat_template(**common, enable_thinking=enable_thinking)
            except TypeError as e:
                if "enable_thinking" not in str(e):
                    raise
            try:
                return tokenizer.apply_chat_template(**common)
            except TypeError as e:
                if "chat_template" not in str(e):
                    raise
                common.pop("chat_template", None)
                return tokenizer.apply_chat_template(**common)

        # Some templates reject the conversation as-is and need one of
        # these repairs (or both together): Gemma's has no "tool" role at
        # all and hard-fails on one ("Conversation roles must
        # alternate..."), a jinja2.exceptions.TemplateError; Qwen3.5's
        # requires tool_calls[].function.arguments to be a dict, not the
        # JSON string every other convention here uses, and raises a
        # plain TypeError ("Can only get item pairs from a mapping")
        # from inside its own template when it isn't.
        attempts = [
            conv,
            fold_tool_messages_into_user(conv),
            coerce_tool_call_arguments_to_dict(conv),
            coerce_tool_call_arguments_to_dict(fold_tool_messages_into_user(conv)),
        ]
        last_err: Exception | None = None
        for c in attempts:
            try:
                return _render(c)
            except (jinja2.exceptions.TemplateError, TypeError) as e:
                last_err = e
                continue
        raise last_err

    def _parse_completion(self, token_ids: list[int], text: str) -> dict:
        """
        Parse completion into structured dict with optional tool_calls key.
        Mirrors GRPOTrainer._generate's parse_response path.
        """
        if self._can_parse:
            try:
                from trl.chat_template_utils import parse_response

                return parse_response(self._get_tokenizer(), token_ids)
            except Exception:
                pass
        return {"role": "assistant", "content": text}

    # ─────────────────────────────────────────────────────────────────────────
    # RolloutEngine interface
    # ─────────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import vllm  # noqa

            return True
        except ImportError:
            return False

    def generate(
        self,
        prompts: Any,
        tools: list[Any],
        gen_cfg: dict[str, Any],
    ) -> dict[str, Any]:

        num_generations = gen_cfg.get("num_generations", 1)
        enable_thinking = gen_cfg.get("enable_thinking", False)
        tools_for_template = tools if tools else None

        if not prompts:
            return {"completions": [], "completions_structured": [], "logprobs": [], "metadata": {}}

        # ── Normalise prompts → flat list of strings ──────────────────────────
        first = prompts[0] if isinstance(prompts, list) else prompts

        if isinstance(prompts, str):
            flat_prompts = [prompts]

        elif isinstance(first, str):
            flat_prompts = prompts

        elif isinstance(first, dict) and "role" in first:
            flat_prompts = [self._apply_template(prompts, tools_for_template, enable_thinking)]

        elif isinstance(first, list) and isinstance(first[0], dict) and "role" in first[0]:
            flat_prompts = [
                self._apply_template(conv, tools_for_template, enable_thinking) for conv in prompts
            ]

        else:
            flat_prompts = prompts

        # ── Tokenize strings → list[list[int]] ───────────────────────────────
        tokenizer = self._get_tokenizer()
        tokenized = tokenizer(
            flat_prompts,
            return_tensors=None,
            truncation=True,
            padding=False,
        )[
            "input_ids"
        ]  # list[list[int]]

        # ── Generate ──────────────────────────────────────────────────────────
        prompt_ids, completion_ids, logprobs, logprob_token_ids = self.vllm_generation.generate(
            prompts=tokenized,
            images=None,  # ← required positional arg
            num_generations=num_generations,
        )

        # ── Decode ────────────────────────────────────────────────────────────
        processing_class = self.vllm_generation.processing_class
        completions_text = processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # Some tool-call formats are delimited by tokens the tokenizer
        # treats as "special" (e.g. Mistral's mistral_common backend
        # renders a call's "[TOOL_CALLS]"/"[ARGS]" as special tokens) --
        # decoding with skip_special_tokens (completions_text above) erases
        # exactly the delimiters the parser needs. Also decode without
        # stripping them, same as the transformers engine does, so callers
        # can prefer this when it's present.
        try:
            completions_raw = processing_class.batch_decode(
                completion_ids, skip_special_tokens=False
            )
        except Exception:
            completions_raw = completions_text

        # ── Structured parse ──────────────────────────────────────────────────
        completions_structured = [
            self._parse_completion(ids, text)
            for ids, text in zip(completion_ids, completions_text, strict=False)
        ]

        logprobs_out = [None] * len(completions_text) if logprobs is None else logprobs

        return {
            "completions": completions_text,
            "completions_raw": completions_raw,
            "completions_structured": completions_structured,
            "logprobs": logprobs_out,
            "metadata": {
                "backend": "trl_vllm_generation",
                "enable_thinking": enable_thinking,
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "extra_fields": logprob_token_ids,
                "can_parse": self._can_parse,
            },
        }
