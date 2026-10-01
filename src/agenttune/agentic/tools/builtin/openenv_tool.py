"""
OpenEnv Tool Adapter — Layer 1
==============================

Wraps a single remote OpenEnv tool as a local ``BaseTool`` so the AgentTune
rollout loop can call it synchronously with zero changes to rollout_factory.py.

Usage (Python):
    from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

    tools, handle = create_openenv_tools(base_url="http://localhost:8000")
    try:
        rollout_fn = create_rollout_fn(tools=tools, ...)
        results = rollout_fn(prompts)
    finally:
        handle.close()

Usage (YAML via TrainerConfigBridge):
    training:
      tools:
        - type: openenv
          base_url: "http://localhost:8000"
          name_prefix: "sandbox_"   # optional, avoids collision with builtins
          tool_filter: ["echo_message"]  # optional subset
          connect_timeout_s: 10
          message_timeout_s: 60

Design notes:
- All OpenEnv imports are lazy (guarded by require_openenv()).  A default install
  that never calls create_openenv_tools() will never import openenv.
- execute() is a plain def (not async).  SyncEnvClient runs async I/O on a
  background thread, so the hot loop's "str(fn(**kwargs))" path works as-is.
- On RuntimeError from the remote env, execute() returns
  ToolResult(success=False, output={"error": ...}) so the rollout loop's
  failure-counter trips correctly.
- All tools built from the same create_openenv_tools() call share one persistent
  WebSocket session.  Parallel rollouts that need isolation should call
  create_openenv_tools() once per worker (see concurrency note in README).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 2 — OpenEnvTool(BaseTool)
# ---------------------------------------------------------------------------


class OpenEnvTool(BaseTool):
    """
    Wraps one remote OpenEnv tool as a local BaseTool.

    Constructor args:
        client        SyncEnvClient wrapping an MCPToolClient (.sync())
        remote_name   Name of the tool on the OpenEnv server
        name          Local tool name (used by ToolRegistry + schema)
        description   Human-readable description forwarded from the remote tool
        input_schema  JSON-Schema dict from the remote tool's Tool.input_schema
        connect_timeout_s  Timeout that was used to build the client (metadata only)
        message_timeout_s  Per-call timeout forwarded as metadata only (the
                           SyncEnvClient blocks the thread; underlying httpx/ws
                           timeouts are set at client-construction time)
    """

    def __init__(
        self,
        client: Any,
        remote_name: str,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
    ) -> None:
        self._client = client
        self._remote_name = remote_name
        self.name = name
        self.description = description
        self._input_schema = input_schema
        self._connect_timeout_s = connect_timeout_s
        self._message_timeout_s = message_timeout_s

    # ------------------------------------------------------------------
    # Schema — Phase 2.1: inject the remote tool's input_schema verbatim
    # ------------------------------------------------------------------

    def _parameters(self) -> dict[str, Any]:
        return self._input_schema

    # ------------------------------------------------------------------
    # Execute — Phase 2.1 / 2.3 / 2.4
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:  # plain def, NOT async
        """
        Call the remote OpenEnv tool synchronously.

        The SyncEnvClient runs the async I/O on a dedicated background thread
        so this method is safe to call from AgentTune's sync rollout hot loop.

        Arguments are forwarded verbatim to the remote tool.  The rollout loop
        may pass arguments as a JSON string on TypeError; that case is handled
        in rollout_factory.py:90-104 before this method is called, so we
        receive a plain dict here.

        Returns ToolResult with:
          success=True,  output=<raw result>  on success
          success=False, output={"error": <msg>}, error=<msg>  on any failure
          (the "error" key makes rollout_factory.py count this as a failure)
        """
        t0 = time.monotonic()
        try:
            result = self._client.call_tool(self._remote_name, **kwargs)
            latency_ms = (time.monotonic() - t0) * 1000
            logger.debug(
                "openenv_tool name=%s remote=%s latency_ms=%.1f success=True",
                self.name,
                self._remote_name,
                latency_ms,
            )
            return ToolResult(
                success=True,
                output=result,
                metadata={"latency_ms": latency_ms, "remote_name": self._remote_name},
            )
        except RuntimeError as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            err_msg = str(exc)
            logger.warning(
                "openenv_tool name=%s remote=%s latency_ms=%.1f success=False error=%r",
                self.name,
                self._remote_name,
                latency_ms,
                err_msg,
            )
            return ToolResult(
                success=False,
                output={"error": err_msg},
                error=err_msg,
                metadata={"latency_ms": latency_ms, "remote_name": self._remote_name},
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "openenv_tool name=%s remote=%s unexpected error=%r",
                self.name,
                self._remote_name,
                err_msg,
            )
            return ToolResult(
                success=False,
                output={"error": err_msg},
                error=err_msg,
                metadata={"latency_ms": latency_ms, "remote_name": self._remote_name},
            )

    def __repr__(self) -> str:
        return f"<OpenEnvTool name='{self.name}' remote='{self._remote_name}'>"


# ---------------------------------------------------------------------------
# Phase 3 — Lifecycle handle
# ---------------------------------------------------------------------------


class OpenEnvHandle:
    """
    Context-manager / close() handle for a group of OpenEnvTools.

    All tools returned by create_openenv_tools() share the same underlying
    SyncEnvClient session.  Close this handle when the tools are no longer
    needed to release the WebSocket connection and the background event loop.

    Usage::

        tools, handle = create_openenv_tools(base_url="http://localhost:8000")
        try:
            ...
        finally:
            handle.close()

        # Or as a context manager:
        with OpenEnvHandle(sync_client) as handle:
            tools = handle.tools
            ...
    """

    def __init__(self, sync_client: Any, tools: list[OpenEnvTool]) -> None:
        self._client = sync_client
        self.tools = tools
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            try:
                self._client.close()
            except Exception as exc:
                logger.warning("OpenEnvHandle.close() error (best-effort): %s", exc)
            finally:
                self._closed = True

    def __enter__(self) -> OpenEnvHandle:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<OpenEnvHandle tools={[t.name for t in self.tools]} " f"closed={self._closed}>"


# ---------------------------------------------------------------------------
# Phase 3 — Factory
# ---------------------------------------------------------------------------


def create_openenv_tools(
    base_url: str | None = None,
    *,
    connect_timeout_s: float = 10.0,
    message_timeout_s: float = 60.0,
    tool_filter: list[str] | None = None,
    name_prefix: str | None = None,
) -> tuple[list[OpenEnvTool], OpenEnvHandle]:
    """
    Connect to a running OpenEnv server and return one OpenEnvTool per remote tool.

    All returned tools share a single persistent SyncEnvClient session (one
    WebSocket connection).  Close the returned ``OpenEnvHandle`` when done to
    release resources.

    Args:
        base_url: HTTP/WS URL of the OpenEnv server, e.g. "http://localhost:8000".
        connect_timeout_s: Timeout for establishing the WebSocket connection.
        message_timeout_s: Per-message timeout for tool calls.
        tool_filter: Optional list of remote tool names to include.  If None,
            all tools exposed by the server are wrapped.
        name_prefix: Optional string prepended to each local tool name to avoid
            collisions with built-in tools (e.g. "sandbox_" → "sandbox_echo_message").

    Returns:
        (tools, handle) where ``tools`` is a list of OpenEnvTool instances and
        ``handle`` is an OpenEnvHandle whose ``.close()`` tears down the session.

    Raises:
        ImportError: If openenv is not installed (with install hint).
        ValueError: If base_url is missing or tool_filter names are not found.
        RuntimeError: If the server is unreachable or list_tools() returns empty.
    """
    from agenttune.utils.optional import require_openenv

    require_openenv()

    # Lazy import — only reached when openenv is installed
    from openenv.core.mcp_client import MCPToolClient

    if not base_url:
        raise ValueError(
            "base_url is required for create_openenv_tools(). "
            "Pass the HTTP URL of your running OpenEnv server, e.g. "
            "'http://localhost:8000'."
        )

    # Warn if pointing at a non-local URL — potential egress of training data
    if not _is_local_url(base_url):
        logger.warning(
            "openenv base_url=%r points to a remote host. "
            "Ensure credentials and training data are not leaked over the network.",
            base_url,
        )

    logger.info(
        "create_openenv_tools: connecting to %s (connect_timeout=%.1fs, " "message_timeout=%.1fs)",
        base_url,
        connect_timeout_s,
        message_timeout_s,
    )

    async_client = MCPToolClient(
        base_url=base_url,
        connect_timeout_s=connect_timeout_s,
        message_timeout_s=message_timeout_s,
    )
    sync_client = async_client.sync()

    try:
        sync_client.connect()
    except Exception as exc:
        sync_client.close()
        raise RuntimeError(
            f"Could not connect to OpenEnv server at {base_url!r}: {exc}\n"
            "Ensure the server is running (e.g. uvicorn server.app:app --port 8000)."
        ) from exc

    # Discover tools
    remote_tools = sync_client.list_tools()

    if not remote_tools:
        sync_client.close()
        raise RuntimeError(
            f"OpenEnv server at {base_url!r} returned no tools. "
            "Check that the environment is running and has MCP tools registered."
        )

    # Apply filter
    if tool_filter is not None:
        available_names = {t.name for t in remote_tools}
        missing = set(tool_filter) - available_names
        if missing:
            sync_client.close()
            raise ValueError(
                f"tool_filter names not found on server: {sorted(missing)}. "
                f"Available: {sorted(available_names)}"
            )
        remote_tools = [t for t in remote_tools if t.name in tool_filter]

    prefix = name_prefix or ""
    tools: list[OpenEnvTool] = []
    for rt in remote_tools:
        local_name = f"{prefix}{rt.name}"
        tool = OpenEnvTool(
            client=sync_client,
            remote_name=rt.name,
            name=local_name,
            description=rt.description,
            input_schema=rt.input_schema,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
        )
        tools.append(tool)
        logger.info(
            "create_openenv_tools: registered tool '%s' (remote: '%s')",
            local_name,
            rt.name,
        )

    handle = OpenEnvHandle(sync_client=sync_client, tools=tools)
    return tools, handle


def register_openenv_tools(
    base_url: str | None = None,
    *,
    connect_timeout_s: float = 10.0,
    message_timeout_s: float = 60.0,
    tool_filter: list[str] | None = None,
    name_prefix: str | None = None,
) -> tuple[list[OpenEnvTool], OpenEnvHandle]:
    """
    Convenience wrapper: create_openenv_tools() + push each tool into ToolRegistry.

    Returns the same (tools, handle) tuple so callers can close the handle.
    """
    from ..registry import ToolRegistry

    tools, handle = create_openenv_tools(
        base_url=base_url,
        connect_timeout_s=connect_timeout_s,
        message_timeout_s=message_timeout_s,
        tool_filter=tool_filter,
        name_prefix=name_prefix,
    )
    for tool in tools:
        ToolRegistry.register_custom(tool)
    return tools, handle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_local_url(url: str) -> bool:
    """Return True if the URL host is localhost / 127.x / ::1."""
    import urllib.parse

    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")
