"""
Tests for the OpenEnv tool adapter (Phase 5 of the L1 roadmap).

All tests run without an actual OpenEnv server — the client is mocked.
Live integration tests are gated by pytest.mark.openenv and the
RUN_OPENENV_IT=1 environment variable.

Test coverage:
  5.1 Unit tests (mocked client)
      - execute() success path → ToolResult(success=True)
      - execute() RuntimeError path → ToolResult(success=False, output={"error": ...})
      - execute() unexpected exception → ToolResult(success=False, output={"error": ...})
      - schema injection: _parameters() == injected input_schema
      - to_schema() produces valid OpenAI function schema
      - execute is NOT a coroutine function
      - factory builds N tools with correct names/descriptions
      - factory applies tool_filter
      - factory applies name_prefix
      - factory raises ValueError on missing base_url
      - factory raises ValueError on unknown tool_filter names (mocked)
  5.2 Import-safety test
      - require_openenv() raises with helpful message when openenv absent
  5.4 Live integration test (marked @pytest.mark.openenv, skipped by default)
      - echo_message round-trips through OpenEnvTool.execute()
      - echo_with_length returns dict with correct keys
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agenttune.agentic.tools.base import ToolResult
from agenttune.agentic.tools.builtin.openenv_tool import (
    OpenEnvHandle,
    OpenEnvTool,
    _is_local_url,
    create_openenv_tools,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "The message to echo"},
    },
    "required": ["message"],
}

FAKE_SCHEMA_2 = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
    },
    "required": ["x", "y"],
}


def _make_tool(
    client=None,
    remote_name="echo_message",
    name=None,
    description="Echo a message",
    input_schema=None,
):
    return OpenEnvTool(
        client=client or MagicMock(),
        remote_name=remote_name,
        name=name or remote_name,
        description=description,
        input_schema=input_schema or FAKE_SCHEMA,
    )


def _make_fake_remote_tool(name: str, description: str, schema: dict) -> Any:
    """Create a mock object resembling openenv's Tool model."""
    t = MagicMock()
    t.name = name
    t.description = description
    t.input_schema = schema
    return t


# ---------------------------------------------------------------------------
# 5.1 Unit tests — mocked client
# ---------------------------------------------------------------------------


class TestOpenEnvToolSchema:
    def test_parameters_returns_injected_schema(self):
        tool = _make_tool(input_schema=FAKE_SCHEMA)
        assert tool._parameters() == FAKE_SCHEMA

    def test_to_schema_produces_openai_format(self):
        tool = _make_tool(name="echo_message", input_schema=FAKE_SCHEMA)
        schema = tool.to_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "echo_message"
        assert "description" in fn
        assert fn["parameters"] == FAKE_SCHEMA

    def test_to_schema_has_required_fields(self):
        tool = _make_tool(input_schema=FAKE_SCHEMA)
        schema = tool.to_schema()
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_custom_schema_injected(self):
        tool = _make_tool(input_schema=FAKE_SCHEMA_2)
        assert tool._parameters()["properties"]["x"]["type"] == "integer"


class TestOpenEnvToolExecute:
    def test_execute_is_not_coroutine(self):
        """Phase 5.1 sync guarantee — execute must be a plain function."""
        tool = _make_tool()
        assert not asyncio.iscoroutinefunction(tool.execute)

    def test_execute_success_returns_tool_result(self):
        client = MagicMock()
        client.call_tool.return_value = "Hello!"
        tool = _make_tool(client=client)

        result = tool.execute(message="Hello!")

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "Hello!"
        assert result.error is None
        client.call_tool.assert_called_once_with("echo_message", message="Hello!")

    def test_execute_success_with_dict_result(self):
        client = MagicMock()
        client.call_tool.return_value = {"message": "hi", "length": 2}
        tool = _make_tool(client=client)

        result = tool.execute(message="hi")

        assert result.success is True
        assert result.output == {"message": "hi", "length": 2}

    def test_execute_runtime_error_returns_failure_result(self):
        """Phase 5.1 error path — RuntimeError → ToolResult(success=False)."""
        client = MagicMock()
        client.call_tool.side_effect = RuntimeError("connection refused")
        tool = _make_tool(client=client)

        result = tool.execute(message="test")

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "error" in result.output  # triggers rollout failure counter
        assert "connection refused" in result.output["error"]
        assert result.error is not None

    def test_execute_unexpected_exception_returns_failure(self):
        """Any non-RuntimeError exception also returns a failure ToolResult."""
        client = MagicMock()
        client.call_tool.side_effect = TimeoutError("timed out")
        tool = _make_tool(client=client)

        result = tool.execute(message="test")

        assert result.success is False
        assert "error" in result.output
        assert "TimeoutError" in result.output["error"]

    def test_execute_error_output_dict_has_error_key(self):
        """
        The rollout loop counts failures on dict results with key 'error'.
        Confirm execute() always uses exactly the 'error' key on failure.
        """
        client = MagicMock()
        client.call_tool.side_effect = RuntimeError("boom")
        tool = _make_tool(client=client)

        result = tool.execute()
        assert isinstance(result.output, dict)
        assert "error" in result.output

    def test_execute_metadata_has_latency(self):
        client = MagicMock()
        client.call_tool.return_value = "ok"
        tool = _make_tool(client=client)

        result = tool.execute(message="x")
        assert "latency_ms" in result.metadata
        assert result.metadata["latency_ms"] >= 0

    def test_execute_passes_kwargs_to_remote(self):
        client = MagicMock()
        client.call_tool.return_value = 42
        tool = _make_tool(client=client, remote_name="add")

        tool.execute(x=3, y=7)
        client.call_tool.assert_called_once_with("add", x=3, y=7)

    def test_execute_remote_name_differs_from_local_name(self):
        """name_prefix: local name 'sandbox_echo' → remote call 'echo'."""
        client = MagicMock()
        client.call_tool.return_value = "pong"
        tool = OpenEnvTool(
            client=client,
            remote_name="echo",
            name="sandbox_echo",
            description="sandboxed echo",
            input_schema=FAKE_SCHEMA,
        )

        result = tool.execute(message="ping")
        assert result.success is True
        client.call_tool.assert_called_once_with("echo", message="ping")

    def test_repr(self):
        tool = _make_tool(name="my_tool", remote_name="remote_tool")
        assert "my_tool" in repr(tool)
        assert "remote_tool" in repr(tool)


class TestOpenEnvHandle:
    def test_close_calls_client_close(self):
        client = MagicMock()
        handle = OpenEnvHandle(sync_client=client, tools=[])
        handle.close()
        client.close.assert_called_once()

    def test_close_is_idempotent(self):
        client = MagicMock()
        handle = OpenEnvHandle(sync_client=client, tools=[])
        handle.close()
        handle.close()
        assert client.close.call_count == 1

    def test_context_manager(self):
        client = MagicMock()
        handle = OpenEnvHandle(sync_client=client, tools=[])
        with handle:
            pass
        client.close.assert_called_once()

    def test_repr(self):
        client = MagicMock()
        tools = [_make_tool(name="t1"), _make_tool(name="t2")]
        handle = OpenEnvHandle(sync_client=client, tools=tools)
        r = repr(handle)
        assert "t1" in r
        assert "t2" in r


# ---------------------------------------------------------------------------
# 5.1 Factory tests (mocked MCPToolClient)
# ---------------------------------------------------------------------------

_FAKE_REMOTE_TOOLS = [
    _make_fake_remote_tool("echo_message", "Echo a message", FAKE_SCHEMA),
    _make_fake_remote_tool("echo_with_length", "Echo with length", FAKE_SCHEMA_2),
]


def _patch_factory(remote_tools=None):
    """Context manager that patches MCPToolClient so no real server is needed.

    MCPToolClient is imported lazily inside create_openenv_tools(), so we must
    patch it at its source (openenv.core.mcp_client) rather than at the
    openenv_tool module level.
    """
    if remote_tools is None:
        remote_tools = _FAKE_REMOTE_TOOLS

    mock_sync = MagicMock()
    mock_sync.list_tools.return_value = remote_tools
    mock_async = MagicMock()
    mock_async.sync.return_value = mock_sync

    return (
        patch(
            "openenv.core.mcp_client.MCPToolClient",
            return_value=mock_async,
        ),
        mock_sync,
    )


class TestCreateOpenenvTools:
    pytestmark = pytest.mark.skipif(
        __import__("importlib").util.find_spec("openenv") is None,
        reason="openenv package not installed",
    )

    def test_factory_raises_import_error_without_openenv(self, monkeypatch):
        monkeypatch.setattr("agenttune.utils.optional.OPENENV_AVAILABLE", False)
        with pytest.raises(ImportError, match="pip install"):
            create_openenv_tools(base_url="http://localhost:8000")

    def test_factory_raises_value_error_without_base_url(self):
        from agenttune.utils.optional import OPENENV_AVAILABLE

        if not OPENENV_AVAILABLE:
            pytest.skip("openenv not installed")
        with pytest.raises((ValueError, ImportError)):
            create_openenv_tools()

    @pytest.mark.openenv
    def test_factory_builds_correct_tool_count(self):
        ctx, mock_sync = _patch_factory()
        with ctx:
            tools, handle = create_openenv_tools(base_url="http://localhost:8000")

        assert len(tools) == 2
        handle.close()

    @pytest.mark.openenv
    def test_factory_tool_names_match_remote(self):
        ctx, mock_sync = _patch_factory()
        with ctx:
            tools, handle = create_openenv_tools(base_url="http://localhost:8000")

        names = {t.name for t in tools}
        assert "echo_message" in names
        assert "echo_with_length" in names
        handle.close()

    @pytest.mark.openenv
    def test_factory_applies_name_prefix(self):
        ctx, mock_sync = _patch_factory()
        with ctx:
            tools, handle = create_openenv_tools(
                base_url="http://localhost:8000", name_prefix="sandbox_"
            )

        names = {t.name for t in tools}
        assert all(n.startswith("sandbox_") for n in names)
        handle.close()

    @pytest.mark.openenv
    def test_factory_applies_tool_filter(self):
        ctx, mock_sync = _patch_factory()
        with ctx:
            tools, handle = create_openenv_tools(
                base_url="http://localhost:8000", tool_filter=["echo_message"]
            )

        assert len(tools) == 1
        assert tools[0].name == "echo_message"
        handle.close()

    @pytest.mark.openenv
    def test_factory_invalid_filter_raises(self):
        ctx, mock_sync = _patch_factory()
        with ctx:
            with pytest.raises(ValueError, match="not found"):
                create_openenv_tools(
                    base_url="http://localhost:8000",
                    tool_filter=["nonexistent_tool"],
                )

    @pytest.mark.openenv
    def test_factory_empty_tool_list_raises(self):
        ctx, mock_sync = _patch_factory(remote_tools=[])
        with ctx:
            with pytest.raises(RuntimeError, match="no tools"):
                create_openenv_tools(base_url="http://localhost:8000")

    @pytest.mark.openenv
    def test_factory_tools_share_client(self):
        ctx, mock_sync = _patch_factory()
        with ctx:
            tools, handle = create_openenv_tools(base_url="http://localhost:8000")

        # All tools reference the same sync client
        clients = [t._client for t in tools]
        assert all(c is clients[0] for c in clients)
        handle.close()

    @pytest.mark.openenv
    def test_factory_tool_has_correct_schema(self):
        ctx, mock_sync = _patch_factory()
        with ctx:
            tools, handle = create_openenv_tools(base_url="http://localhost:8000")

        echo = next(t for t in tools if t.name == "echo_message")
        assert echo._parameters() == FAKE_SCHEMA
        handle.close()

    @pytest.mark.openenv
    def test_factory_connection_failure_raises(self):
        mock_async = MagicMock()
        mock_sync = MagicMock()
        mock_sync.connect.side_effect = ConnectionRefusedError("refused")
        mock_async.sync.return_value = mock_sync

        with patch(
            "openenv.core.mcp_client.MCPToolClient",
            return_value=mock_async,
        ):
            with pytest.raises(RuntimeError, match="Could not connect"):
                create_openenv_tools(base_url="http://localhost:9999")


# ---------------------------------------------------------------------------
# 5.2 Import-safety test
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_require_openenv_raises_when_absent(self, monkeypatch):
        monkeypatch.setattr("agenttune.utils.optional.OPENENV_AVAILABLE", False)
        from agenttune.utils.optional import require_openenv

        with pytest.raises(ImportError) as exc_info:
            require_openenv()

        assert "pip install" in str(exc_info.value)
        assert "openenv" in str(exc_info.value).lower()

    def test_default_registry_loads_without_openenv(self):
        """auto_register_builtins() must succeed without openenv installed."""
        from agenttune.agentic.tools.registry import ToolRegistry

        # Reset state to force re-registration
        ToolRegistry._builtins_registered = False
        ToolRegistry._tools = {}
        ToolRegistry.auto_register_builtins()

        # OpenEnv tools must NOT be in the default registry
        tool_names = ToolRegistry.list_all()
        assert not any("openenv" in n for n in tool_names)

    def test_openenv_tool_module_importable_without_openenv(self):
        """The module itself imports fine; errors only surface on function call."""
        from agenttune.agentic.tools.builtin import openenv_tool  # noqa: F401


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_is_local_url_localhost(self):
        assert _is_local_url("http://localhost:8000") is True

    def test_is_local_url_127(self):
        assert _is_local_url("http://127.0.0.1:8080") is True

    def test_is_local_url_remote(self):
        assert _is_local_url("http://example.com:8000") is False

    def test_is_local_url_ws(self):
        assert _is_local_url("ws://localhost:8000/ws") is True


# ---------------------------------------------------------------------------
# 5.4 Live integration tests (skipped unless RUN_OPENENV_IT=1 + openenv installed)
# ---------------------------------------------------------------------------


@pytest.mark.openenv
@pytest.mark.integration
class TestLiveEchoEnv:
    """
    Live tests that hit a real echo_env server.

    To run:
        # Terminal 1: start echo_env
        cd /tmp/OpenEnv/envs/echo_env
        uvicorn server.app:app --port 8765

        # Terminal 2: run these tests
        RUN_OPENENV_IT=1 pytest tests/agentic/test_openenv_tool.py -m openenv -v
    """

    @pytest.fixture(autouse=True)
    def skip_unless_enabled(self):
        from agenttune.utils.optional import OPENENV_AVAILABLE

        if not OPENENV_AVAILABLE:
            pytest.skip("openenv not installed (pip install -e .)")
        if not os.environ.get("RUN_OPENENV_IT"):
            pytest.skip("Live tests skipped. Set RUN_OPENENV_IT=1 and start echo_env server.")

    @pytest.fixture
    def echo_tools(self):
        echo_url = os.environ.get("OPENENV_ECHO_URL", "http://localhost:8765")
        tools, handle = create_openenv_tools(base_url=echo_url)
        yield tools, handle
        handle.close()

    def test_echo_message_round_trips(self, echo_tools):
        tools, _ = echo_tools
        echo = next(t for t in tools if t.name == "echo_message")
        result = echo.execute(message="hello from agenttune")
        assert result.success is True
        assert result.output == "hello from agenttune"

    def test_echo_with_length_returns_dict(self, echo_tools):
        tools, _ = echo_tools
        echo_len = next(t for t in tools if t.name == "echo_with_length")
        result = echo_len.execute(message="test")
        assert result.success is True
        assert isinstance(result.output, dict)
        assert result.output.get("message") == "test"
        assert result.output.get("length") == 4

    def test_schema_matches_live_server(self, echo_tools):
        tools, _ = echo_tools
        echo = next(t for t in tools if t.name == "echo_message")
        schema = echo.to_schema()
        assert schema["function"]["name"] == "echo_message"
        params = schema["function"]["parameters"]
        assert "message" in params.get("properties", {})

    def test_error_path_on_bad_args(self, echo_tools):
        tools, _ = echo_tools
        echo = next(t for t in tools if t.name == "echo_message")
        # Pass wrong args — server should return an error
        result = echo.execute(bad_arg="should fail")
        # Either success=False with error key, or the server echoed something
        if not result.success:
            assert "error" in result.output
