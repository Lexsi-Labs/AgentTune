from langchain_community.tools import DuckDuckGoSearchResults

from ..base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web using DuckDuckGo and return relevant results."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                },
                "output_format": {
                    "type": "string",
                    "default": "list",
                    "enum": ["list", "json", "string"],
                    "description": "Format of the returned results.",
                },
                "backend": {
                    "type": "string",
                    "default": "text",
                    "enum": ["text", "news"],
                    "description": "Search backend. Use 'news' to search news articles only.",
                },
            },
            "required": ["query"],
        }

    def execute(
        self,
        query: str,
        output_format: str = "list",
        backend: str = "text",
    ) -> ToolResult:
        try:
            search = DuckDuckGoSearchResults(
                output_format=output_format,
                backend=backend,
            )
            results = search.invoke(query)
            return ToolResult(success=True, output=results)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
