"""Tool call stage for external tool integration."""

import time
from typing import Any

from agenttune.agentic.tools.executor import ToolExecutor
from agenttune.agentic.tools.registry import ToolRegistry
from agenttune.decide.stages.base import StageHandler
from agenttune.decide.state import PipelineState


class ToolCallStage(StageHandler):
    """
    Stage for calling built-in and custom tools from AgentTune.

    Integrates with agenttune.agentic.tools for SQL, web search,
    file I/O, code execution, GitHub, Slack, Playwright, FinQA, etc.

    YAML schema:
        - id: lookup_customer
          type: tool_call
          tool: sql_query
          args:
            query: "SELECT * FROM customers WHERE id = '{extract.output.customer_id}'"
          next: risk_score
    """

    def __init__(self, stage_config: dict[str, Any] | None = None):
        super().__init__(stage_config)
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute tool call stage.

        Interpolates arguments with pipeline context, executes the named tool,
        and returns structured output.

        Args:
            state: Pipeline state with stage_outputs for context
            stage_config: Stage configuration (optional, uses self.stage_config if not provided)

        Returns:
            Dictionary with keys:
            - output: Tool result
            - latency_ms: Execution time
            - cost_usd: Estimated API cost if applicable
            - error: Error message if execution failed

        Raises:
            KeyError: If tool not found in registry
            ValueError: If required arguments missing
        """
        config = stage_config or self.stage_config
        tool_name = config.get("tool")
        if not tool_name:
            return {
                "output": None,
                "error": "Missing required 'tool' key in stage config",
            }

        # Interpolate arguments with pipeline context
        raw_args = config.get("args", {})
        if not isinstance(raw_args, dict):
            return {
                "output": None,
                "error": f"Tool args must be dict, got {type(raw_args).__name__}",
            }

        interpolated_args = self._interpolate_dict(raw_args, state)

        # Check if tool is registered
        try:
            self.registry.get(tool_name)
        except KeyError:
            available = ", ".join(self.registry.list_tools())
            return {
                "output": None,
                "error": f"Tool '{tool_name}' not found. Available: {available}",
            }

        # Execute with timeout
        start_time = time.time()
        timeout_sec = config.get("timeout_sec", 30)

        try:
            result = await self.executor.run(tool_name, interpolated_args, timeout_sec=timeout_sec)
            latency_ms = int((time.time() - start_time) * 1000)

            return {
                "output": result,
                "latency_ms": latency_ms,
                "cost_usd": 0.0,  # TODO: track actual tool costs
            }

        except TimeoutError:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "output": None,
                "error": f"Tool execution timed out after {timeout_sec}s",
                "latency_ms": latency_ms,
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "output": None,
                "error": f"Tool execution failed: {str(e)}",
                "latency_ms": latency_ms,
            }

    def _interpolate_dict(self, data: dict[str, Any], state: PipelineState) -> dict[str, Any]:
        """Recursively interpolate dict values with pipeline context."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._interpolate(value, state)
            elif isinstance(value, dict):
                result[key] = self._interpolate_dict(value, state)
            elif isinstance(value, list):
                result[key] = [
                    (
                        self._interpolate(v, state)
                        if isinstance(v, str)
                        else self._interpolate_dict(v, state) if isinstance(v, dict) else v
                    )
                    for v in value
                ]
            else:
                result[key] = value
        return result
