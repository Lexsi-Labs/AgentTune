"""Regression tests for _run_tools_parallel async handling.

Guards against a NameError in the async branch: the gather step used to be
unreachable code inside `_call_one` and referenced an undefined `_gather`, so
any async tool raised `NameError: name '_gather' is not defined`.
"""

import asyncio

from agenttune.agentic.rollout_engines.rollout_factory import _run_tools_parallel


async def _atool(x):
    return f"async:{x}"


def _stool(x):
    return f"sync:{x}"


def test_async_and_sync_tools_no_running_loop():
    res = _run_tools_parallel([("a", _atool, {"x": 1}), ("s", _stool, {"x": 2})])
    assert res == [("a", "async:1"), ("s", "sync:2")]


def test_async_tool_inside_running_loop():
    async def main():
        return _run_tools_parallel([("a", _atool, {"x": 9})])

    assert asyncio.run(main()) == [("a", "async:9")]


def test_sync_only_fast_path():
    res = _run_tools_parallel([("s", _stool, {"x": 3})])
    assert res == [("s", "sync:3")]
