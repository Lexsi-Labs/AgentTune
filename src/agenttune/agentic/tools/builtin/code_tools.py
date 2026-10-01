import os
import subprocess
import sys
import tempfile

from ..base import BaseTool, ToolResult


class RunPythonTool(BaseTool):
    name = "run_python"
    description = "Execute a Python code string and return stdout/stderr."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["code"],
        }

    def execute(self, code: str, timeout: int = 30) -> ToolResult:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            os.unlink(tmp_path)
            output = result.stdout + result.stderr
            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="Timeout exceeded")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class RunBashTool(BaseTool):
    name = "run_bash"
    description = "Execute a bash command and return stdout/stderr."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["command"],
        }

    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=result.stderr if result.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="Timeout exceeded")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
