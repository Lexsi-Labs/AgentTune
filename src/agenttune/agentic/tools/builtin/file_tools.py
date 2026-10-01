import os

from ..base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file at a given path."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path"],
        }

    def execute(self, path: str, encoding: str = "utf-8") -> ToolResult:
        try:
            with open(path, encoding=encoding) as f:
                content = f.read()
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file at a given path."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "default": "w"},
            },
            "required": ["path", "content"],
        }

    def execute(self, path: str, content: str, mode: str = "w") -> ToolResult:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, mode) as f:
                f.write(content)
            return ToolResult(success=True, output=f"Written to {path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List files and directories at a given path."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
            },
            "required": [],
        }

    def execute(self, path: str = ".") -> ToolResult:
        try:
            entries = os.listdir(path)
            return ToolResult(success=True, output=entries)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
