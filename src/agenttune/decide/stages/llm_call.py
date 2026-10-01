"""LLM call stage for model inference with optional agentic tool use."""

import time
from typing import Any

from agenttune.decide.stages.base import StageHandler
from agenttune.decide.state import PipelineState


class LLMCallStage(StageHandler):
    """
    Stage for calling an LLM with optional agentic tool use.

    Supports:
    - Simple LLM call: prompt → response
    - Agentic loop: prompt → (model + tools) → multi-step reasoning
    - JSON schema validation and iteration loops
    - Tool caching via rollout engine

    YAML schema for simple call:
        - id: extract_info
          type: llm_call
          prompt: "Extract customer data: {input_text}"
          output_schema: {type: object, properties: {...}}
          max_iterations: 3

    YAML schema for agentic:
        - id: research_agent
          type: llm_call
          prompt: "Research this company: {input_text}"
          tools: [web_search, sql_query]
          max_steps: 5
          output_schema: {type: object, properties: {...}}
    """

    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute LLM call stage (simple or agentic).

        Args:
            state: Pipeline state with stage_outputs for context
            stage_config: Optional stage configuration override

        Returns:
            Dictionary with keys:
            - output: The LLM response or final agent result
            - conversation: Full conversation history if agentic
            - tool_calls: List of tool calls made (if agentic)
            - latency_ms: Total execution time
            - cost_usd: Estimated API cost
            - error: Error message if execution failed
        """
        # Use provided config or instance config
        config = stage_config or self.stage_config

        # Get model - check stage config first, then global config
        model = config.get("model") or config.get("default_model")
        if not model and hasattr(self, "global_config"):
            model = self.global_config.get("default_model")
        if not model:
            raise ValueError("No model specified and no default_model in config")

        # Interpolate prompt
        prompt = config.get("prompt", "")
        interpolated_prompt = self._interpolate(prompt, state)

        start_time = time.time()

        # Check if this is agentic (has tools)
        tools = config.get("tools")
        if tools:
            result = await self._execute_agentic(model, interpolated_prompt, tools, state, config)
        else:
            result = await self._execute_simple(model, interpolated_prompt, state, config)

        # Add latency
        result["latency_ms"] = int((time.time() - start_time) * 1000)

        return result

    async def _execute_simple(
        self, model: str, prompt: str, state: PipelineState, config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Execute simple (non-agentic) LLM call with JSON validation."""
        cfg = config or self.stage_config
        # Support both output_schema and json_schema
        output_schema = cfg.get("output_schema", cfg.get("json_schema", {}))
        max_iterations = cfg.get("max_iterations", 1)
        state.stage_iterations.get(cfg["id"], 0) + 1

        # Retry loop for JSON validation
        for attempt in range(max_iterations):
            try:
                response = await self._call_model(model, prompt)
            except Exception as e:
                return {
                    "output": None,
                    "error": str(e),
                    "cost_usd": 0.0,
                }

            # Attempt JSON validation
            try:
                output = self._validate_json(response, output_schema)
                return {
                    "output": output,
                    "cost_usd": 0.0,  # TODO: track actual costs
                }
            except ValueError as e:
                # JSON validation failed
                if attempt < max_iterations - 1:
                    # More attempts remaining, continue loop
                    continue
                elif "on_parse_error" in cfg:
                    # Return instruction to loop back
                    return {
                        "output": None,
                        "failed": True,
                        "goto": cfg.get("on_parse_error"),
                        "inject": f"JSON parsing failed: {str(e)}. Please provide valid JSON.",
                        "error": str(e),
                        "cost_usd": 0.0,
                    }
                else:
                    # No more attempts and no on_parse_error handler
                    return {
                        "output": None,
                        "error": f"JSON validation failed: {str(e)}",
                        "cost_usd": 0.0,
                    }

        return {
            "output": None,
            "error": "Max iterations reached without valid JSON",
            "cost_usd": 0.0,
        }

    async def _execute_agentic(
        self,
        model: str,
        prompt: str,
        tool_names: list[str],
        state: PipelineState,
        config: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Execute agentic LLM call with tool use.

        Requires rollout_engine to be configured (via self.rollout_engine).
        Falls back to simple call if rollout engine not available.

        Args:
            model: Model name/path
            prompt: Initial prompt
            tool_names: List of tool names to use
            state: Pipeline state
            config: Optional configuration override

        Returns:
            Dict with final output, conversation history, and tool calls
        """
        cfg = config or self.stage_config
        # Check if rollout engine is available
        if not hasattr(self, "rollout_engine") or self.rollout_engine is None:
            # Fallback: log warning and run simple LLM call
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Agentic mode requested but rollout_engine not configured. "
                "Falling back to simple LLM call. "
                "Configure rollout_engine in config.yaml to enable tool use."
            )
            return await self._execute_simple(model, prompt, state, config)

        try:
            from agenttune.agentic.rollout_engines.rollout_factory import (
                create_rollout_fn,
            )

            max_steps = cfg.get("max_steps", 5)

            # Create rollout function
            rollout_fn = create_rollout_fn(
                rollout_engine=self.rollout_engine,
                tools=[{"name": t} for t in tool_names],  # Minimal tool specs
                max_steps=max_steps,
                system_prompt="You are a helpful assistant with access to tools.",
            )

            # Execute rollout
            result = rollout_fn([prompt])

            # Extract final response
            final_response = result.get("responses", [prompt])[0]
            conversation = result.get("conversations", [[]])[0]
            tool_calls = result.get("tool_calls", [])

            # Validate output if schema provided
            output_schema = cfg.get("output_schema", {})
            try:
                output = self._validate_json(final_response, output_schema)
            except ValueError:
                output = final_response  # Use raw response if validation fails

            return {
                "output": output,
                "conversation": conversation,
                "tool_calls": tool_calls,
                "cost_usd": 0.0,
            }

        except Exception as e:
            return {
                "output": None,
                "error": f"Agentic execution failed: {str(e)}",
                "cost_usd": 0.0,
            }
