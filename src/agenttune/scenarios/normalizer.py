"""
agenttune.agentic.scenarios.normalizer
=======================================
Converts *any* tool representation into a plain ToolInfo dict that the
scenario generator can serialise into its prompt.

Accepted inputs (per tool)
--------------------------
1. Plain Python callable with type-annotated signature
   - Uses transformers.utils.get_json_schema when available, else inspects
     annotations manually.
2. agenttune.agentic.tools.base.BaseTool subclass instance
   - Calls .to_schema() / uses .name, .description, .parameters
3. Raw dict  {"name": ..., "description": ..., "parameters": ...}
4. MCP MCPTool / MCPResource -- imported lazily (no hard dependency)
5. Any object with .name + .description attributes (duck-typed)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

# -----------------------------------------------------------------------------
# Public data classes
# -----------------------------------------------------------------------------


@dataclass
class ToolInfo:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ResourceInfo:
    name: str
    description: str
    uri: str | None = None
    mime_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "description": self.description}
        if self.uri:
            d["uri"] = self.uri
        if self.mime_type:
            d["mime_type"] = self.mime_type
        return d


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _python_type_to_json(tp) -> str:
    """Small type mapping covering the common cases."""
    origin = getattr(tp, "__origin__", None)
    if origin is not None:
        if origin is list:
            return "array"
        return "string"
    _MAP = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        bytes: "string",
        dict: "object",
        list: "array",
        type(None): "null",
    }
    return _MAP.get(tp, "string")


def _schema_from_callable(fn) -> dict[str, Any]:
    """
    Extract a JSON-schema-style dict from a plain Python function.

    Priority:
    1. transformers.utils.get_json_schema  (richest, handles docstrings)
    2. Manual inspection of inspect.signature + type annotations
    """
    # Path 1 -- transformers (best quality)
    try:
        from transformers.utils import get_json_schema

        schema = get_json_schema(fn)
        fn_def = schema.get("function", schema)
        return {
            "name": fn_def.get("name", fn.__name__),
            "description": fn_def.get("description", inspect.getdoc(fn) or ""),
            "parameters": fn_def.get("parameters", {}),
        }
    except Exception:
        pass

    # Path 2 -- manual fallback
    sig = inspect.signature(fn)
    hints = {}
    try:
        import typing

        hints = typing.get_type_hints(fn)
    except Exception:
        pass

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        type_hint = hints.get(pname)
        prop: dict[str, Any] = {}
        if type_hint is not None:
            prop["type"] = _python_type_to_json(type_hint)
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        properties[pname] = prop

    return {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _normalize_one_tool(tool: Any) -> ToolInfo:
    """Normalise a single tool from any supported form."""

    # 1. Already a ToolInfo
    if isinstance(tool, ToolInfo):
        return tool

    # 2. Plain dict
    if isinstance(tool, dict):
        return ToolInfo(
            name=tool.get("name", "unknown"),
            description=tool.get("description", ""),
            parameters=tool.get("parameters", tool.get("inputSchema", {})),
        )

    # 3. agenttune BaseTool
    try:
        from agenttune.agentic.tools.base import BaseTool

        if isinstance(tool, BaseTool):
            if hasattr(tool, "to_schema"):
                schema = tool.to_schema()
                fn_def = schema.get("function", schema)
                return ToolInfo(
                    name=fn_def.get("name", tool.name),
                    description=fn_def.get("description", getattr(tool, "description", "")),
                    parameters=fn_def.get("parameters", getattr(tool, "parameters", {})),
                )
            return ToolInfo(
                name=tool.name,
                description=getattr(tool, "description", ""),
                parameters=getattr(tool, "parameters", {}),
            )
    except ImportError:
        pass

    # 4. MCP MCPTool (optional dep)
    try:
        from art.mcp.types import MCPTool

        if isinstance(tool, MCPTool):
            d = tool.to_dict()
            return ToolInfo(
                name=d.get("name", ""),
                description=d.get("description", ""),
                parameters=d.get("parameters", d.get("inputSchema", {})),
            )
    except ImportError:
        pass

    # 5. Plain callable
    if callable(tool):
        info = _schema_from_callable(tool)
        return ToolInfo(
            name=info["name"],
            description=info["description"],
            parameters=info["parameters"],
        )

    # 6. Duck-typed fallback
    if hasattr(tool, "name"):
        return ToolInfo(
            name=str(tool.name),
            description=str(getattr(tool, "description", "")),
            parameters=getattr(tool, "parameters", getattr(tool, "inputSchema", {})),
        )

    raise TypeError(
        f"Cannot normalise tool of type {type(tool).__name__!r}. "
        "Pass a callable, dict, BaseTool, MCPTool, or ToolInfo."
    )


def _normalize_one_resource(res: Any) -> ResourceInfo:
    """Normalise a single resource from any supported form."""

    if isinstance(res, ResourceInfo):
        return res

    if isinstance(res, dict):
        return ResourceInfo(
            name=res.get("name", "unknown"),
            description=res.get("description", ""),
            uri=res.get("uri"),
            mime_type=res.get("mime_type", res.get("mimeType")),
        )

    # MCP MCPResource
    try:
        from art.mcp.types import MCPResource

        if isinstance(res, MCPResource):
            d = res.to_dict()
            return ResourceInfo(
                name=d.get("name", ""),
                description=d.get("description", ""),
                uri=d.get("uri"),
                mime_type=d.get("mimeType"),
            )
    except ImportError:
        pass

    # Duck-typed fallback
    if hasattr(res, "name"):
        return ResourceInfo(
            name=str(res.name),
            description=str(getattr(res, "description", "")),
            uri=getattr(res, "uri", None),
            mime_type=getattr(res, "mime_type", getattr(res, "mimeType", None)),
        )

    raise TypeError(f"Cannot normalise resource of type {type(res).__name__!r}.")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def normalize_tools(tools: list[Any]) -> list[ToolInfo]:
    """
    Convert a mixed list of tools into ToolInfo objects.

    Accepts any combination of callables, dicts, BaseTool, MCPTool, ToolInfo.
    """
    return [_normalize_one_tool(t) for t in tools]


def normalize_resources(resources: list[Any] | None) -> list[ResourceInfo]:
    """Convert a mixed list of resources into ResourceInfo objects."""
    if not resources:
        return []
    return [_normalize_one_resource(r) for r in resources]
