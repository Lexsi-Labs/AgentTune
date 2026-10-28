"""Tests for ToolRegistry's lazy builtin loading."""

import subprocess
import sys

import pytest

from agenttune.agentic.tools import BaseTool, ToolRegistry, ToolResult


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(ToolRegistry, "_tools", {})


class SandboxedReadFile(BaseTool):
    name = "read_file"
    description = "Read a file inside a sandbox."

    def _parameters(self) -> dict:
        return {"type": "object", "properties": {"path": {"type": "string"}}}

    def execute(self, path: str) -> ToolResult:
        return ToolResult(success=True, output="sandboxed")


def test_stdlib_tool_loads_without_langchain_community():
    # Fresh interpreter with langchain_community made unimportable: fetching a stdlib
    # tool must not import any other builtin module.
    script = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == "langchain_community":
            raise ImportError(f"blocked {name}")

sys.meta_path.insert(0, Block())
from agenttune.agentic.tools import ToolRegistry

assert ToolRegistry.get("read_file").name == "read_file"
assert "agenttune.agentic.tools.builtin.slack" not in sys.modules
try:
    ToolRegistry.get("slack")
except ImportError:
    pass
else:
    raise AssertionError("slack loaded without langchain_community")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_list_all_includes_builtins_before_first_get():
    names = ToolRegistry.list_all()
    assert {"read_file", "run_bash", "slack", "sql_database"} <= set(names)
    assert ToolRegistry._tools == {}


def test_custom_tool_replaces_builtin_of_same_name():
    ToolRegistry.register_custom(SandboxedReadFile())
    assert isinstance(ToolRegistry.get("read_file"), SandboxedReadFile)

    ToolRegistry.auto_register_builtins()
    assert isinstance(ToolRegistry.get("read_file"), SandboxedReadFile)
    assert ToolRegistry.list_all().count("read_file") == 1


def test_get_registers_builtin_once():
    first = ToolRegistry.get("run_python")
    assert ToolRegistry.get("run_python") is first


def test_unknown_tool_raises_key_error_listing_available():
    with pytest.raises(KeyError, match="read_file"):
        ToolRegistry.get("no_such_tool")
