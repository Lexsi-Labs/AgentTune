import asyncio
import json
import logging

import litellm

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, TrainingExample
from agenttune.decide.closed_loop.replay_validator import ReplayValidator

logger = logging.getLogger(__name__)


_NAME_KEYS = ("name", "tool", "tool_name", "function", "action")
_ARG_KEYS = ("arguments", "args", "parameters", "params")


def _parse_tool_call(text: str) -> dict | None:
    """Parse a synthesized correction into a ``{name, arguments}`` tool call.

    Real LLMs emit tool calls in many shapes; this accepts the common ones so the
    TAC reward can validate the arguments rather than fall back to the lenient
    raw-string path:
      - ``{"name"|"tool"|"tool_name"|"action": <str>, "arguments"|"args"|"parameters": {...}}``
      - ``{"function": <str>, "order_id": 42}``  (name as string; args are the
        remaining top-level keys)
      - OpenAI-style ``{"function": {"name": ..., "arguments": ...}}``
    Returns ``None`` when ``text`` is not a JSON tool-call object, so the caller
    falls back to scoring the raw string. Pure/​stdlib — no side effects — so it
    stays import-safe for this litellm-free-at-import module.
    """
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    # OpenAI-style nested function object.
    if isinstance(obj.get("function"), dict):
        fn = obj["function"]
        return {"name": fn.get("name"), "arguments": fn.get("arguments", {})}
    # Name from the first string-valued name key.
    name = next((obj[k] for k in _NAME_KEYS if isinstance(obj.get(k), str) and obj[k]), None)
    if name is None:
        return None
    # Arguments: an explicit args key, else the remaining (non-meta) top-level keys.
    args = next((obj[k] for k in _ARG_KEYS if obj.get(k) is not None), None)
    if args is None:
        meta = set(_NAME_KEYS) | set(_ARG_KEYS)
        args = {k: v for k, v in obj.items() if k not in meta}
    return {"name": name, "arguments": args if args is not None else {}}


class TrainingExampleGenerator:
    """
    Generates training examples from classified failures using multi-completion sampling
    (for RL algorithms like GRPO), incorporating TAC and TER rewards.
    """

    def __init__(
        self,
        validator: ReplayValidator,
        model_name: str = "groq/llama-3.3-70b-versatile",
        api_base: str = None,
        tool_schemas: dict | None = None,
    ):
        # Imported lazily to avoid a circular import: trajectory_eval imports
        # closed_loop.contracts, which triggers closed_loop/__init__ -> full_loop
        # -> this module. Importing TrajectoryEvaluator at call time breaks the cycle.
        from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

        self.validator = validator
        self.model_name = model_name
        self.api_base = api_base
        # tool_schemas make the TAC reward meaningful: with them, a corrected tool
        # call's arguments are validated against the schema (well-formed → 1.0,
        # malformed → 0.0). Without them TAC can't validate, so we keep the lenient
        # legacy behaviour (score the raw string, TAC=1.0) rather than a misleading 0.
        self.tool_schemas = tool_schemas or {}
        self.evaluator = TrajectoryEvaluator(
            model_name=model_name, api_base=api_base, tool_schemas=self.tool_schemas
        )

    async def _determine_num_completions(self, failure: ClassifiedFailure) -> int:
        """Dynamically decide how many examples are needed based on failure complexity."""
        if failure.root_cause == "wrong_tool":
            return 2  # Simple error, small N
        elif failure.root_cause == "loop_collapse":
            return 4  # Complex reasoning error, larger N for RL
        return 1

    async def _synthesize_completions(self, failure: ClassifiedFailure, n: int) -> list[str]:
        """Attempt to synthesize N corrected responses via LLM concurrently."""
        prompt = f"""You are fixing an agent trajectory that failed due to: {failure.root_cause}.
Failure analysis: {failure.analysis}
Context: {json.dumps(failure.failure.context)[:2000]}

Please write the EXACT JSON tool call or output the agent SHOULD have generated instead.
Return ONLY the raw JSON or string text that represents the correct action."""

        async def fetch_one():
            try:
                # Use temperature to ensure diversity in the N completions
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                }
                if self.api_base:
                    kwargs["api_base"] = self.api_base
                resp = await litellm.acompletion(**kwargs)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Failed to synthesize correction: {e}")
                return None

        results = await asyncio.gather(*[fetch_one() for _ in range(n)])
        return [r for r in results if r is not None]

    def _build_prompt_history(self, failure: ClassifiedFailure) -> list[dict[str, str]]:
        return [{"role": "user", "content": "Task context leading to failure..."}]

    async def _process_single(self, failure: ClassifiedFailure) -> TrainingExample | None:
        if failure.root_cause not in ["wrong_tool", "loop_collapse"]:
            return None

        num_needed = await self._determine_num_completions(failure)
        prompt_history = self._build_prompt_history(failure)
        completions_text = await self._synthesize_completions(failure, num_needed)

        if not completions_text:
            return None

        # Create base example structure
        example = TrainingExample(
            trajectory_id=failure.failure.trajectory_id,
            original_failure_type=failure.failure.failure_type,
            root_cause=failure.root_cause,
            prompt=prompt_history,
            completions=[],
            rewards=[],
            salvaged_at_attempt=1,
            is_negative_only=False,
        )

        # Validate and score each completion to build the reward array
        for comp_text in completions_text:
            is_valid, tool_outputs = await self.validator.validate_example_with_trace(
                example, comp_text
            )

            # Parse the synthesized correction into a real {name, arguments} tool call
            # so TAC actually validates the corrected arguments against the schema. The
            # synthesis prompt asks for "the EXACT JSON tool call", so corrections ARE
            # JSON — passing the raw string instead hits _calculate_tac's lenient string
            # branch and scores TAC=1.0 for every completion, making the reward unable to
            # tell a well-formed correction from a malformed one. Only use the parsed
            # call when schemas are available to validate against; otherwise fall back to
            # the legacy raw-string behaviour (no schema ⇒ no meaningful validation).
            parsed_call = _parse_tool_call(comp_text)
            use_parsed = parsed_call is not None and bool(self.tool_schemas)
            trajectory = {
                "trajectory_id": example.trajectory_id,
                "tool_calls": [parsed_call if use_parsed else comp_text],
                "tool_outputs": tool_outputs,
            }

            # 1. TAC (Tool Argument Correctness)
            tac_score = self.evaluator._calculate_tac(trajectory)

            # 2. TER (Tool Efficacy Reward)
            ter_score = self.evaluator._calculate_ter(trajectory) if is_valid else 0.0

            # Combine rewards
            total_reward = tac_score + ter_score
            # Add severe penalty if validation fails entirely
            if not is_valid:
                total_reward -= 1.0

            example.completions.append([{"role": "assistant", "content": comp_text}])
            example.rewards.append(total_reward)

        if not example.completions:
            return None

        # Try to extract the original failing response as a fallback rejected candidate
        original_rejected = None
        try:
            messages = failure.failure.context.get("state_snapshot", {}).get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if "tool_calls" in msg:
                        content += "\n" + json.dumps(msg["tool_calls"])
                    original_rejected = [{"role": "assistant", "content": content.strip()}]
                    break
        except Exception as e:
            logger.debug(f"Could not extract original failure for {example.trajectory_id}: {e}")

        # Determine chosen and rejected
        if example.completions:
            best_idx = example.rewards.index(max(example.rewards))
            worst_idx = example.rewards.index(min(example.rewards))

            if example.rewards[best_idx] > example.rewards[worst_idx]:
                # If there's a strict difference, use the best and worst generated completions
                example.chosen = example.completions[best_idx]
                example.rejected = example.completions[worst_idx]
            elif original_rejected and example.rewards[best_idx] >= 0:
                # Fallback: if generations tied (or only 1 valid generation), use original failure
                example.chosen = example.completions[best_idx]
                example.rejected = original_rejected

        logger.info(
            f"Generated {len(example.completions)} scored completions for {example.trajectory_id}."
        )
        return example

    async def generate_batch(self, failures: list[ClassifiedFailure]) -> list[TrainingExample]:
        """
        Generate training examples for a batch of classified failures in parallel.
        """
        tasks = [self._process_single(f) for f in failures]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
