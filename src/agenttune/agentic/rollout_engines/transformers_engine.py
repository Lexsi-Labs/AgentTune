import warnings
from typing import Any

import torch
import torch.nn.functional as F

from .base import RolloutEngine
from .rollout_factory import (
    _keep_tool_results,
    coerce_tool_call_arguments_to_dict,
    fold_tool_messages_into_user,
)


class TransformersRolloutEngine(RolloutEngine):
    """
    Rollout engine wrapping a HuggingFace transformers model.

    Accepts raw Python functions as tools (no BaseTool wrapper needed).
    Uses TRL's parse_response + prefix-preserving chat template when available,
    falling back gracefully so it works with transformers < 5.0 too.
    """

    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

        # ── Robust device resolution ──────────────────────────────────────────
        # Handles single-device, multi-GPU (device_map="auto"), and CPU-only.
        # We resolve to a single "primary" device used for moving input tensors
        # when the model is NOT device-mapped across multiple GPUs.
        self._device = self._resolve_device(model)
        # True when the model is spread across multiple devices (device_map="auto"
        # or a custom map). In that case we must NOT call .to(device) on the
        # encoded inputs — HF's generate() handles placement internally.
        self._is_multi_device = self._check_multi_device(model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generate needs left padding.
        self.tokenizer.padding_side = "left"
        gen_cfg_obj = getattr(self.model, "generation_config", None)
        if gen_cfg_obj is not None and self.tokenizer.pad_token_id is not None:
            gen_cfg_obj.pad_token_id = self.tokenizer.pad_token_id

        # Keep the checkpoint's own chat template. TRL's training template is a
        # different dialect and will silently train/eval a model against the
        # wrong tool-call syntax.
        native_tmpl = getattr(self.tokenizer, "chat_template", None)
        self._chat_template = None
        self._can_parse = False

        try:
            import transformers
            from packaging.version import Version
            from trl.chat_template_utils import add_response_schema

            if Version(transformers.__version__) >= Version("5.0.0"):
                if not getattr(self.tokenizer, "response_schema", None):
                    self.tokenizer = add_response_schema(self.tokenizer)
                self._can_parse = getattr(self.tokenizer, "response_schema", None) is not None
                if native_tmpl is not None:
                    self.tokenizer.chat_template = native_tmpl

        except (ImportError, Exception):
            pass

    # ── Device helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(model) -> torch.device:
        """
        Return the primary device for the model.

        Priority:
          1. model.device  — set by .to() / .cuda() / from_pretrained(device_map=...)
          2. First parameter's device — reliable for single-device models.
          3. CPU fallback.
        """
        # Attribute set by transformers when device_map places everything on one device
        if hasattr(model, "device") and model.device.type != "meta":
            return model.device

        # Walk parameters; skip meta tensors (uninitialised shards)
        for param in model.parameters():
            if param.device.type != "meta":
                return param.device

        return torch.device("cpu")

    @staticmethod
    def _check_multi_device(model) -> bool:
        """
        Return True when model parameters live on more than one real device
        (i.e. device_map spread the model across multiple GPUs or CPU+GPU).
        Meta tensors are ignored — they are not yet placed.
        """
        devices = {param.device for param in model.parameters() if param.device.type != "meta"}
        return len(devices) > 1

    def _move_inputs(self, encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Move tokeniser outputs to the correct device(s).

        - Multi-device model  → move to CPU so HF's dispatch hook re-routes
          each layer's inputs automatically. (Placing on any single GPU would
          break layers on other GPUs.)
        - Single-device model → move everything to that device.
        """
        target = self._device if self._device.type != "meta" else torch.device("cpu")
        return {k: v.to(target, non_blocking=target.type == "cuda") for k, v in encoded.items()}

    # ── Helpers ───────────────────────────────────────────────────────────────

    # def _apply_chat_template(
    #     self,
    #     conversations: List[List[dict]],
    #     tools: Optional[List[dict]],
    #     enable_thinking: bool,
    # ) -> dict:
    #     """
    #     Tokenise conversations with graceful fallback for older transformers.
    #     Mirrors GRPOTrainer._generate_single_turn's apply_chat_template call.
    #     """
    #     common = dict(
    #         conversation=conversations,
    #         tools=tools if tools else None,
    #         chat_template=self._chat_template,
    #         add_generation_prompt=True,
    #         tokenize=True,
    #         padding=True,
    #         padding_side="left",
    #         return_tensors="pt",
    #         return_dict=True,
    #     )

    #     # Try with enable_thinking first (Qwen3 / thinking models)
    #     if enable_thinking:
    #         try:
    #             return self.tokenizer.apply_chat_template(
    #                 **common, enable_thinking=True
    #             )
    #         except TypeError:
    #             pass

    #     # Try without enable_thinking
    #     try:
    #         return self.tokenizer.apply_chat_template(**common)
    #     except TypeError:
    #         # Oldest fallback: strip chat_template kwarg
    #         common.pop("chat_template", None)
    #         return self.tokenizer.apply_chat_template(**common)

    def _tool_content_survives_render(
        self, convs: list[list[dict]], tools: list[dict] | None
    ) -> bool:
        """Detect a chat template that renders a "tool"-role message as an
        empty turn instead of raising -- e.g. CohereLabs/tiny-aya-fire's
        default template has no `elif role == 'tool'` branch at all, so the
        turn's `<|START_OF_TURN_TOKEN|>...<|END_OF_TURN_TOKEN|>` wrapper comes
        out empty and the tool result is silently gone from what the model
        sees next. That renders successfully (no exception), so the
        exception-driven `attempts` ladder below never retries it -- this
        check is what makes a *silent* drop trigger the same fallback path a
        loud one (Gemma's hard "must alternate" crash) already gets.
        Best-effort: any error here just means "can't confirm it survived".
        """
        has_tool_msg = any(m.get("role") == "tool" for c in convs for m in c)
        if not has_tool_msg:
            return True
        try:
            texts = self.tokenizer.apply_chat_template(
                conversation=convs,
                tools=tools if tools else None,
                chat_template=self._chat_template,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            return False
        if isinstance(texts, str):
            texts = [texts]
        for conv, text in zip(convs, texts, strict=False):
            for msg in conv:
                if msg.get("role") != "tool":
                    continue
                content = str(msg.get("content", "")).strip()
                if content and content[:200] not in text:
                    return False
        return True

    def _apply_chat_template(
        self,
        conversations: list[list[dict]],
        tools: list[dict] | None,
        enable_thinking: bool,
    ) -> dict:
        import jinja2

        def _render(convs):
            common = {
                "conversation": convs,
                "tools": tools if tools else None,
                "chat_template": self._chat_template,
                "add_generation_prompt": True,
                "tokenize": True,
                "padding": True,
                "padding_side": "left",
                "return_tensors": "pt",
                "return_dict": True,
            }
            try:
                return self.tokenizer.apply_chat_template(
                    **common, enable_thinking=enable_thinking  # ← always explicit
                )
            except TypeError as e:
                # Only retry without enable_thinking if THAT's actually
                # the problem (an unexpected-keyword-argument TypeError
                # from apply_chat_template's own call signature) -- not a
                # same-named TypeError raised from inside the jinja
                # template itself (e.g. Qwen3.5's below), which retrying
                # the same way would just reproduce identically.
                if "enable_thinking" not in str(e):
                    raise
                common.pop("chat_template", None)
                return self.tokenizer.apply_chat_template(**common)

        # Some templates reject the conversation as-is and need one of
        # these repairs (or both together): Gemma's has no "tool" role at
        # all and hard-fails on one ("Conversation roles must
        # alternate..."), a jinja2.exceptions.TemplateError; Qwen3.5's
        # requires tool_calls[].function.arguments to be a dict, not the
        # JSON string every other convention here uses, and raises a
        # plain TypeError ("Can only get item pairs from a mapping")
        # from inside its own template when it isn't.
        conversations = [
            _keep_tool_results(self.tokenizer, c, self._chat_template) for c in conversations
        ]
        attempts = [
            conversations,
            [fold_tool_messages_into_user(c) for c in conversations],
            [coerce_tool_call_arguments_to_dict(c) for c in conversations],
            [
                coerce_tool_call_arguments_to_dict(fold_tool_messages_into_user(c))
                for c in conversations
            ],
        ]
        last_err: Exception | None = None
        silent_drop_fallback: dict | None = None
        for convs in attempts:
            survives = self._tool_content_survives_render(convs, tools)
            try:
                rendered = _render(convs)
            except (jinja2.exceptions.TemplateError, TypeError) as e:
                last_err = e
                continue
            if survives:
                return rendered
            # Rendered without raising, but the template silently dropped a
            # tool-result turn (no "tool"-role branch to render it). Keep
            # this as a last resort and try the next repair in the ladder --
            # fold_tool_messages_into_user rewrites the "tool" role away
            # entirely, which the same template's "user" branch usually
            # renders fine.
            if silent_drop_fallback is None:
                silent_drop_fallback = rendered
        if silent_drop_fallback is not None:
            return silent_drop_fallback
        raise last_err

    def _compute_assistant_prefill(
        self,
        conversations: list[list[dict]],
        tools: list[dict] | None,
        enable_thinking: bool,
    ) -> list[str]:
        """Recover text some chat templates render as part of the *prompt*
        itself, not the completion -- e.g. Functionary's trailing ">>>"
        recipient marker, or Qwen3's empty "<think></think>" block. That
        text is the LAST thing before generation starts, so only decoding
        the newly generated tokens (as `generate()` below does) silently
        drops it -- which breaks any tool-call format anchored on it (e.g.
        Functionary's parser expects ">>>NAME" at the start of the call).
        Recovered as the text diff between rendering the same conversation
        with and without add_generation_prompt; best-effort, empty string
        per item on any failure (a missing prefill is a no-op, not a
        regression).
        """
        text_common = {
            "tools": tools if tools else None,
            "chat_template": self._chat_template,
            "tokenize": False,
        }
        try:
            no_gen = self.tokenizer.apply_chat_template(
                conversation=conversations,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **text_common,
            )
            with_gen = self.tokenizer.apply_chat_template(
                conversation=conversations,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **text_common,
            )
        except Exception:
            return [""] * len(conversations)
        prefills = [
            wg[len(ng) :] if wg.startswith(ng) else ""
            for ng, wg in zip(no_gen, with_gen, strict=False)
        ]
        # The generation prompt also opens the assistant turn ("<|im_start|>assistant\n",
        # "<start_of_turn>model\n"). That opener is not part of the answer: left in,
        # the answer reads "assistant\n{...}" once special tokens are stripped. Find it
        # by rendering a probe assistant turn and drop it where the prefill starts with it.
        sentinel = "agenttune-prefill-probe"
        try:
            with_msg = self.tokenizer.apply_chat_template(
                conversation=[
                    [*conv, {"role": "assistant", "content": sentinel}] for conv in conversations
                ],
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **text_common,
            )
        except Exception:
            return prefills
        out = []
        for ng, wm, prefill in zip(no_gen, with_msg, prefills, strict=False):
            idx = wm.find(sentinel, len(ng))
            opener = wm[len(ng) : idx] if wm.startswith(ng) and idx >= 0 else ""
            out.append(prefill[len(opener) :] if opener and prefill.startswith(opener) else prefill)
        return out

    def _parse_completion(self, token_ids: list[int], text: str) -> dict:
        """
        Parse a completion into a structured message dict.
        Mirrors GRPOTrainer._generate's parse_response path.
        """
        if self._can_parse:
            try:
                from trl.chat_template_utils import parse_response

                return parse_response(self.tokenizer, token_ids)
            except Exception:
                pass
        return {"role": "assistant", "content": text}

    # ── Main generate ──────────────────────────────────────────────────────────

    # def generate(
    #     self,
    #     prompts: Any,          # str | list[dict] | list[list[dict]]
    #     tools: List[Any],
    #     gen_cfg: Dict[str, Any],
    # ) -> Dict[str, Any]:

    #     tools_enabled = len(tools) > 0
    #     enable_thinking = gen_cfg.get("enable_thinking", False)

    #     if not prompts:
    #         return {"completions": [], "completions_structured": [], "logprobs": [], "metadata": {}}

    #     # Normalise into list of conversations
    #     if isinstance(prompts, str) or (isinstance(prompts, list) and isinstance(prompts[0], str)):
    #         raw = [prompts] if isinstance(prompts, str) else prompts
    #         conversations = [[{"role": "user", "content": p}] for p in raw]
    #     elif isinstance(prompts[0], dict) and "role" in prompts[0]:
    #         conversations = [prompts]
    #     else:
    #         conversations = prompts

    #     encoded = self._apply_chat_template(conversations, tools if tools_enabled else None, enable_thinking)

    #     # ── Move inputs to the right device(s) ───────────────────────────────
    #     encoded = self._move_inputs(encoded)

    #     input_len = encoded["input_ids"].shape[1]

    #     # ── Generate ──────────────────────────────────────────────────────────
    #     with torch.no_grad():
    #         output_ids = self.model.generate(
    #             **encoded,
    #             max_new_tokens=gen_cfg.get("max_new_tokens", 512),
    #             temperature=gen_cfg.get("temperature", 0.7),
    #             do_sample=gen_cfg.get("do_sample", True),
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #         )

    #     new_tokens = output_ids[:, input_len:]
    #     completion_ids_list = [ids.tolist() for ids in new_tokens]

    #     completions_text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    #     completions_raw = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)

    #     # ── Structured parse ──────────────────────────────────────────────────
    #     completions_structured = [
    #         self._parse_completion(ids, text)
    #         for ids, text in zip(completion_ids_list, completions_text)
    #     ]

    #     # ── Logprobs ──────────────────────────────────────────────────────────
    #     # output_ids may be on any device after generate(); bring logits to CPU
    #     # to avoid device-mismatch errors when indexing across shards.
    #     with torch.no_grad():
    #         logits = self.model(input_ids=output_ids).logits

    #     log_probs = F.log_softmax(logits.float(), dim=-1).cpu()
    #     output_ids_cpu = output_ids.cpu()
    #     new_tokens_cpu = new_tokens.cpu()

    #     batch_logprobs = []
    #     for i in range(output_ids_cpu.shape[0]):
    #         completion_tokens = new_tokens_cpu[i]
    #         completion_len = completion_tokens.shape[0]
    #         relevant = log_probs[i, input_len - 1: input_len - 1 + completion_len, :]
    #         token_lp = relevant[
    #             torch.arange(completion_len),
    #             completion_tokens,
    #         ].tolist()
    #         batch_logprobs.append(token_lp)

    #     return {
    #         "completions": completions_text,
    #         "completions_raw": completions_raw,
    #         "completions_structured": completions_structured,
    #         "logprobs": batch_logprobs,
    #         "metadata": {
    #             "backend": "transformers",
    #             "tools_enabled": tools_enabled,
    #             "enable_thinking": enable_thinking,
    #             "completion_ids": completion_ids_list,
    #             "can_parse": self._can_parse,
    #             "device": str(self._device),
    #             "multi_device": self._is_multi_device,
    #         },
    #     }
    def generate(
        self,
        prompts: Any,
        tools: list[Any],
        gen_cfg: dict[str, Any],
    ) -> dict[str, Any]:

        tools_enabled = len(tools) > 0
        enable_thinking = gen_cfg.get("enable_thinking", False)
        max_length = gen_cfg.get("max_length", 40960)
        max_new_tokens = gen_cfg.get("max_new_tokens", 512)
        prompt_budget = max_length - max_new_tokens  # tokens available for prompt

        if not prompts:
            return {"completions": [], "completions_structured": [], "logprobs": [], "metadata": {}}

        # Normalise into list of conversations
        if isinstance(prompts, str) or (isinstance(prompts, list) and isinstance(prompts[0], str)):
            raw = [prompts] if isinstance(prompts, str) else prompts
            conversations = [[{"role": "user", "content": p}] for p in raw]
        elif isinstance(prompts[0], dict) and "role" in prompts[0]:
            conversations = [prompts]
        else:
            conversations = prompts

        assistant_prefills = self._compute_assistant_prefill(
            conversations, tools if tools_enabled else None, enable_thinking
        )

        encoded = self._apply_chat_template(
            conversations, tools if tools_enabled else None, enable_thinking
        )
        encoded = self._move_inputs(encoded)

        # ── Truncate from LEFT if prompt already too long ─────────────────────
        # Keeps most recent context (tool results, last assistant turn).
        # This is the safest place to enforce the limit — before generate()
        # sees the input, so no warning is ever raised.
        if encoded["input_ids"].shape[1] > prompt_budget:
            warnings.warn(
                f"Prompt length ({encoded['input_ids'].shape[1]} tokens) exceeds budget "
                f"({prompt_budget} = max_length {max_length} - max_new_tokens {max_new_tokens}). "
                f"Truncating left side to preserve recent context.",
                UserWarning,
            )
            for key in encoded:
                if isinstance(encoded[key], torch.Tensor) and encoded[key].dim() == 2:
                    encoded[key] = encoded[key][:, -prompt_budget:]

        input_len = encoded["input_ids"].shape[1]  # recompute after possible truncation

        # Some chat models (e.g. Llama 3.1/3.2) stop a tool-call turn on a
        # different token than their default text eos -- Llama's built-in
        # tool format ends with "<|eom_id|>", not "<|eot_id|>". That extra
        # stop id lives in the model's own generation_config.eos_token_id
        # (which HF sets from generation_config.json as a list), not in
        # tokenizer.eos_token_id (a single id). Passing only the tokenizer's
        # id here used to override that list, so generate() never saw
        # "<|eom_id|>" as a stop condition and ran to max_new_tokens,
        # repeating the same tool call over and over.
        eos_token_id = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id

        # ── Generate ──────────────────────────────────────────────────────────
        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=gen_cfg.get("temperature", 0.7),
                do_sample=gen_cfg.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        new_tokens = output_ids[:, input_len:]
        completion_ids_list = [ids.tolist() for ids in new_tokens]

        completions_text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        completions_raw = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
        # Put back any assistant-turn prefill the prompt itself supplied
        # (see _compute_assistant_prefill) so tool-call formats anchored on
        # it parse correctly.
        completions_text = [
            p + t for p, t in zip(assistant_prefills, completions_text, strict=False)
        ]
        completions_raw = [p + t for p, t in zip(assistant_prefills, completions_raw, strict=False)]

        completions_structured = [
            self._parse_completion(ids, text)
            for ids, text in zip(completion_ids_list, completions_text, strict=False)
        ]

        # ── Logprobs: KV-cache two-pass (avoids OOM on full sequence) ────────
        with torch.no_grad():
            prompt_out = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=True,
            )
            past_kv = prompt_out.past_key_values

            comp_out = self.model(
                input_ids=new_tokens,
                past_key_values=past_kv,
                use_cache=False,
            )
            logits = comp_out.logits  # (B, completion_len, vocab)

        log_probs = F.log_softmax(logits.float(), dim=-1).cpu()
        new_tokens_cpu = new_tokens.cpu()

        batch_logprobs = []
        for i in range(new_tokens_cpu.shape[0]):
            completion_tokens = new_tokens_cpu[i]
            completion_len = completion_tokens.shape[0]
            token_lp = log_probs[
                i,
                torch.arange(completion_len),
                completion_tokens,
            ].tolist()
            batch_logprobs.append(token_lp)

        # Free KV cache immediately — critical in multi-turn tool loops
        del prompt_out, comp_out, past_kv, logits, log_probs
        torch.cuda.empty_cache()

        return {
            "completions": completions_text,
            "completions_raw": completions_raw,
            "completions_structured": completions_structured,
            "logprobs": batch_logprobs,
            "metadata": {
                "backend": "transformers",
                "tools_enabled": tools_enabled,
                "enable_thinking": enable_thinking,
                "completion_ids": completion_ids_list,
                "can_parse": self._can_parse,
                "device": str(self._device),
                "multi_device": self._is_multi_device,
                "prompt_len": input_len,
            },
        }

    def is_available(self) -> bool:
        return True

    def _get_tokenizer(self):
        return self.tokenizer

    def _apply_template(self, conversation, tools, enable_thinking=False) -> str:
        common = {
            "conversation": conversation,
            "tools": tools if tools else None,
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if enable_thinking:
            try:
                return self.tokenizer.apply_chat_template(**common, enable_thinking=True)
            except TypeError:
                pass
        try:
            return self.tokenizer.apply_chat_template(**common)
        except TypeError:
            common.pop("tools", None)
            return self.tokenizer.apply_chat_template(**common)
