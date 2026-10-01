import subprocess

from ..base import BaseTool, ToolResult


class GrepTool(BaseTool):
    name = "grep"
    description = "Search for a pattern in files using grep."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "recursive": {"type": "boolean", "default": True},
            },
            "required": ["pattern"],
        }

    def execute(self, pattern: str, path: str = ".", recursive: bool = True) -> ToolResult:
        try:
            flags = "-r" if recursive else ""
            result = subprocess.run(
                f"grep {flags} '{pattern}' {path}",
                shell=True,
                capture_output=True,
                text=True,
            )
            return ToolResult(success=True, output=result.stdout)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
