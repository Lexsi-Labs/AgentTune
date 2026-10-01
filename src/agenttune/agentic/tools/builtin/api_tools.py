from ..base import BaseTool, ToolResult


class HttpGetTool(BaseTool):
    name = "http_get"
    description = "Make an HTTP GET request to a URL."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "headers": {"type": "object", "default": {}},
                "timeout": {"type": "integer", "default": 10},
            },
            "required": ["url"],
        }

    def execute(self, url: str, headers: dict = None, timeout: int = 10) -> ToolResult:
        try:
            import requests

            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            return ToolResult(
                success=resp.ok,
                output=resp.text,
                error=None if resp.ok else f"HTTP {resp.status_code}",
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class HttpPostTool(BaseTool):
    name = "http_post"
    description = "Make an HTTP POST request to a URL with a JSON body."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "body": {"type": "object", "default": {}},
                "headers": {"type": "object", "default": {}},
                "timeout": {"type": "integer", "default": 10},
            },
            "required": ["url"],
        }

    def execute(
        self, url: str, body: dict = None, headers: dict = None, timeout: int = 10
    ) -> ToolResult:
        try:
            import requests

            resp = requests.post(url, json=body or {}, headers=headers or {}, timeout=timeout)
            return ToolResult(
                success=resp.ok,
                output=resp.text,
                error=None if resp.ok else f"HTTP {resp.status_code}",
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
