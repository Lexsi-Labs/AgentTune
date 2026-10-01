"""Tests for ToolExecutor."""

import asyncio

import pytest

from agenttune.agentic.tools.base import BaseTool, ToolResult
from agenttune.agentic.tools.executor import ToolExecutor
from agenttune.agentic.tools.registry import ToolRegistry


class MockSyncTool(BaseTool):
    """Mock synchronous tool for testing."""

    name = "mock_sync_tool"
    description = "Mock sync tool"

    def execute(self, value: str) -> ToolResult:
        return ToolResult(success=True, output=f"processed: {value}")


class MockAsyncTool(BaseTool):
    """Mock asynchronous tool for testing."""

    name = "mock_async_tool"
    description = "Mock async tool"

    async def execute(self, value: str) -> ToolResult:
        await asyncio.sleep(0.01)
        return ToolResult(success=True, output=f"async: {value}")


class MockSlowTool(BaseTool):
    """Mock tool that takes a long time."""

    name = "mock_slow_tool"
    description = "Mock slow tool"

    async def execute(self, delay: float) -> ToolResult:
        await asyncio.sleep(delay)
        return ToolResult(success=True, output="done")


class MockFailingTool(BaseTool):
    """Mock tool that fails."""

    name = "mock_failing_tool"
    description = "Mock failing tool"

    def execute(self) -> ToolResult:
        return ToolResult(success=False, output=None, error="Tool intentionally failed")


class MockRawOutputTool(BaseTool):
    """Mock tool that returns raw output (not ToolResult)."""

    name = "mock_raw_output_tool"
    description = "Mock tool with raw output"

    def execute(self, text: str) -> str:
        return f"raw: {text}"


@pytest.fixture
def registry():
    """Create a registry with mock tools."""
    reg = ToolRegistry()
    reg.register(MockSyncTool())
    reg.register(MockAsyncTool())
    reg.register(MockSlowTool())
    reg.register(MockFailingTool())
    reg.register(MockRawOutputTool())
    return reg


@pytest.fixture
def executor(registry):
    """Create an executor with the mock registry."""
    return ToolExecutor(registry)


@pytest.mark.asyncio
async def test_sync_tool_execution(executor):
    """Test executing a synchronous tool."""
    result = await executor.run("mock_sync_tool", {"value": "test"})
    assert result == "processed: test"


@pytest.mark.asyncio
async def test_async_tool_execution(executor):
    """Test executing an asynchronous tool."""
    result = await executor.run("mock_async_tool", {"value": "test"})
    assert result == "async: test"


@pytest.mark.asyncio
async def test_tool_not_found(executor):
    """Test executing a non-existent tool."""
    with pytest.raises(KeyError):
        await executor.run("nonexistent_tool", {})


@pytest.mark.asyncio
async def test_timeout_handling(executor):
    """Test that timeout is raised when tool takes too long."""
    with pytest.raises(TimeoutError):
        await executor.run("mock_slow_tool", {"delay": 1.0}, timeout_sec=0.1)


@pytest.mark.asyncio
async def test_tool_failure(executor):
    """Test that tool execution failure is raised as exception."""
    with pytest.raises(Exception, match="Tool intentionally failed"):
        await executor.run("mock_failing_tool", {})


@pytest.mark.asyncio
async def test_tool_success_within_timeout(executor):
    """Test successful tool execution within timeout."""
    result = await executor.run("mock_slow_tool", {"delay": 0.05}, timeout_sec=1.0)
    assert result == "done"


@pytest.mark.asyncio
async def test_raw_output_tool(executor):
    """Test tool that returns raw output instead of ToolResult."""
    result = await executor.run("mock_raw_output_tool", {"text": "hello"})
    assert result == "raw: hello"


@pytest.mark.asyncio
async def test_executor_with_empty_args(executor):
    """Test executor with no arguments raises an exception."""
    with pytest.raises(Exception):
        await executor.run("mock_failing_tool", {})


@pytest.mark.asyncio
async def test_multiple_tool_executions(executor):
    """Test executing multiple tools in sequence."""
    result1 = await executor.run("mock_sync_tool", {"value": "first"})
    result2 = await executor.run("mock_async_tool", {"value": "second"})

    assert result1 == "processed: first"
    assert result2 == "async: second"


@pytest.mark.asyncio
async def test_concurrent_tool_executions(executor):
    """Test executing multiple tools concurrently."""
    results = await asyncio.gather(
        executor.run("mock_sync_tool", {"value": "one"}),
        executor.run("mock_async_tool", {"value": "two"}),
        executor.run("mock_sync_tool", {"value": "three"}),
    )

    assert results == [
        "processed: one",
        "async: two",
        "processed: three",
    ]


@pytest.mark.asyncio
async def test_timeout_error_message(executor):
    """Test that timeout error has a descriptive message."""
    try:
        await executor.run("mock_slow_tool", {"delay": 1.0}, timeout_sec=0.1)
    except TimeoutError as e:
        assert "timed out" in str(e)
        assert "0.1" in str(e)


@pytest.mark.asyncio
async def test_default_timeout(executor):
    """Test that default timeout is applied."""
    result = await executor.run("mock_slow_tool", {"delay": 0.05})
    assert result == "done"
