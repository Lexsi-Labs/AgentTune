import asyncio
from typing import Any

from .base import ToolResult
from .registry import ToolRegistry


class ToolExecutor:
    """Executes tools from the registry with timeout support."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def run(
        self,
        tool_name: str,
        args: dict[str, Any],
        timeout_sec: int = 30,
    ) -> Any:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments as keyword arguments
            timeout_sec: Maximum execution time in seconds (default 30)

        Returns:
            Tool output on success

        Raises:
            TimeoutError: If execution exceeds timeout_sec
            KeyError: If tool not found in registry
            Exception: If tool execution fails
        """
        tool = self.registry.get(tool_name)

        async def _run() -> Any:
            result = tool.execute(**args)
            if asyncio.iscoroutine(result):
                result = await result

            if not isinstance(result, ToolResult):
                return result

            if not result.success:
                raise Exception(result.error or "Tool execution failed")

            return result.output

        try:
            return await asyncio.wait_for(_run(), timeout=timeout_sec)
        except TimeoutError:
            raise TimeoutError(f"Tool execution timed out after {timeout_sec}s")  # noqa: B904
