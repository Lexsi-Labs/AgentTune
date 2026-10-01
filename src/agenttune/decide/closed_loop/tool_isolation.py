"""
OpenEnv L1 Tool Isolation — Path B, Week 3
==========================================

Run a tool OFF the training process so a crashing / slow / untrusted tool
cannot take down the rollout loop or read the training process's memory.

This is "Level 1" isolation: a separate OS *process* on the same host (via
``concurrent.futures.ProcessPoolExecutor``). Levels 2–3 (containers, remote
hosts) are a later workstream and explicitly out of this sprint.

Design
------
``IsolatedTool`` wraps any ``BaseTool`` and presents the SAME ``BaseTool``
interface (``name``, ``description``, ``to_schema``, ``execute``), so the
rollout loop is unchanged — it cannot tell an isolated tool from a local one.

``execute`` dispatches the wrapped tool's call to a worker process with a
timeout. **Fallback guarantee:** if isolation fails for any reason (process
pool unavailable, worker crash, timeout, un-picklable tool), it falls back to
running the tool IN-PROCESS so the loop never blocks. Every result carries
``metadata["isolation"]`` = ``"process"`` or ``"in_process_fallback"`` so the
behaviour is observable.

A tool run in a child process must be picklable (top-level class, no live
sockets). Tools that hold unpicklable state (e.g. an open WebSocket like the
OpenEnv adapter) are detected up-front and run in-process — still safe, just
not isolated. ``require_isolation=True`` turns a failed isolation into an error
ToolResult instead of a silent fallback, for callers that must not run
untrusted code in-process.
"""

from __future__ import annotations

import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from agenttune.agentic.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# Module-level worker so it is importable/picklable by the child process.
def _run_tool_in_worker(tool: BaseTool, kwargs: dict[str, Any]) -> ToolResult:
    """Execute a tool inside the worker process and return its ToolResult."""
    return tool.execute(**kwargs)


class IsolatedTool(BaseTool):
    """Wrap a ``BaseTool`` so its ``execute`` runs in a separate process.

    Falls back to in-process execution if isolation is not possible, unless
    ``require_isolation=True``.
    """

    def __init__(
        self,
        tool: BaseTool,
        timeout_s: float = 30.0,
        require_isolation: bool = False,
    ) -> None:
        self._tool = tool
        self.name = getattr(tool, "name", tool.__class__.__name__)
        self.description = getattr(tool, "description", "")
        self._timeout_s = timeout_s
        self._require_isolation = require_isolation
        # Decide once whether this tool can even be shipped to a worker.
        self._picklable = self._check_picklable(tool)
        if not self._picklable:
            logger.warning(
                "IsolatedTool('%s'): wrapped tool is not picklable; calls will run "
                "in-process (still safe, not isolated).",
                self.name,
            )

    # -- schema passthrough -------------------------------------------------

    def _parameters(self) -> dict:
        return self._tool._parameters()

    def to_schema(self) -> dict:
        return self._tool.to_schema()

    # -- execution ----------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        # Tool can't be pickled → can't cross a process boundary.
        if not self._picklable:
            if self._require_isolation:
                return ToolResult(
                    success=False,
                    output={"error": "isolation required but tool is not picklable"},
                    error="isolation_unavailable",
                    metadata={"isolation": "unavailable", "tool": self.name},
                )
            return self._run_in_process(kwargs, reason="not_picklable")

        # Try real process isolation.
        try:
            with ProcessPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_tool_in_worker, self._tool, kwargs)
                result = future.result(timeout=self._timeout_s)
            if isinstance(result, ToolResult):
                result.metadata = {
                    **(result.metadata or {}),
                    "isolation": "process",
                    "tool": self.name,
                }
                return result
            # Defensive: a tool that returned a non-ToolResult.
            return ToolResult(
                success=True, output=result, metadata={"isolation": "process", "tool": self.name}
            )
        except FutureTimeout:
            msg = f"isolated tool '{self.name}' timed out after {self._timeout_s}s"
            logger.warning(msg)
            if self._require_isolation:
                return ToolResult(
                    success=False,
                    output={"error": msg},
                    error="timeout",
                    metadata={"isolation": "process", "tool": self.name},
                )
            return self._run_in_process(kwargs, reason="timeout")
        except Exception as exc:  # noqa: BLE001 — any isolation failure → fallback
            logger.warning("IsolatedTool('%s') process execution failed: %s", self.name, exc)
            if self._require_isolation:
                return ToolResult(
                    success=False,
                    output={"error": str(exc)},
                    error="isolation_failed",
                    metadata={"isolation": "process", "tool": self.name},
                )
            return self._run_in_process(kwargs, reason="isolation_error")

    # -- fallback -----------------------------------------------------------

    def _run_in_process(self, kwargs: dict[str, Any], reason: str) -> ToolResult:
        """Run the wrapped tool in this process so the loop never blocks."""
        try:
            result = self._tool.execute(**kwargs)
            if isinstance(result, ToolResult):
                result.metadata = {
                    **(result.metadata or {}),
                    "isolation": "in_process_fallback",
                    "fallback_reason": reason,
                    "tool": self.name,
                }
                return result
            return ToolResult(
                success=True,
                output=result,
                metadata={
                    "isolation": "in_process_fallback",
                    "fallback_reason": reason,
                    "tool": self.name,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output={"error": str(exc)},
                error=str(exc),
                metadata={
                    "isolation": "in_process_fallback",
                    "fallback_reason": reason,
                    "tool": self.name,
                },
            )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _check_picklable(tool: BaseTool) -> bool:
        try:
            pickle.dumps(tool)
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"<IsolatedTool name='{self.name}' picklable={self._picklable}>"


def isolate_tools(
    tools,
    timeout_s: float = 30.0,
    require_isolation: bool = False,
):
    """Wrap each tool in an :class:`IsolatedTool` (convenience for a tool list)."""
    return [
        IsolatedTool(t, timeout_s=timeout_s, require_isolation=require_isolation) for t in tools
    ]
