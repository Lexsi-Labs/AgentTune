"""Router stage for conditional dispatch based on LLM output."""

import time
from typing import Any

from agenttune.decide.stages.base import StageHandler, flatten_stage_outputs
from agenttune.decide.stages.rules import SafeEvaluator
from agenttune.decide.state import PipelineState


class RouterStage(StageHandler):
    """
    Stage for conditional routing based on LLM output.

    Calls LLM and routes to different stages based on output conditions.
    """

    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute router stage.

        Args:
            state: Pipeline state
            stage_config: Stage configuration (optional, uses self.stage_config if not provided)

        Returns:
            Dictionary with routing decision
        """
        start_time = time.time()

        # Get model (optional - router can be rules-based)
        config = stage_config or self.stage_config
        model = config.get("model") or config.get("default_model")
        if not model and hasattr(self, "global_config"):
            model = self.global_config.get("default_model")

        output = None
        if model:
            # LLM-based routing
            prompt = config.get("prompt", "")
            interpolated_prompt = self._interpolate(prompt, state)

            try:
                response = await self._call_model(model, interpolated_prompt)
            except Exception as e:
                return {
                    "output": None,
                    "error": str(e),
                    "latency_ms": int((time.time() - start_time) * 1000),
                }

            # Validate JSON output
            output_schema = config.get("output_schema", {})
            try:
                output = self._validate_json(response, output_schema)
            except ValueError as e:
                return {
                    "output": None,
                    "error": f"JSON validation failed: {str(e)}",
                    "latency_ms": int((time.time() - start_time) * 1000),
                }
        else:
            # Rules-based routing - use state.stage_outputs. Copy it: the caller stores
            # this return value back into state.stage_outputs[stage_id], and handing back
            # the live dict would insert it into itself (a circular reference that then
            # crashes json.dumps() in audit logging).
            output = dict(state.stage_outputs)

        # Evaluate on_result conditions for routing
        on_result = config.get("on_result", [])
        evaluator = SafeEvaluator()

        # Flatten context for evaluator if no model (rules-based routing)
        eval_context = output
        if not model and isinstance(output, dict):
            eval_context = flatten_stage_outputs(output)

        for condition_path in on_result:
            condition = condition_path.get("condition")
            goto = condition_path.get("goto")

            if not condition or not goto:
                continue

            try:
                if evaluator.eval(condition, eval_context):
                    return {
                        "output": output,
                        "goto": goto,
                        "latency_ms": int((time.time() - start_time) * 1000),
                    }
            except Exception:
                # Skip this condition if evaluation fails
                continue

        # Default: use default field or stay
        default = config.get("default")
        return {
            "output": output,
            "goto": default,
            "latency_ms": int((time.time() - start_time) * 1000),
        }
