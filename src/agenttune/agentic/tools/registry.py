from .base import BaseTool


class ToolRegistry:
    """
    Central registry for all tools.
    Builtins are auto-registered on first use.
    Users can add tools via register_custom().

    To add a new builtin category:
        1. Create src/agenttune/agentic/tools/builtin/my_tools.py
        2. Add import + register calls in auto_register_builtins()
        3. Done
    """

    _tools: dict[str, BaseTool] = {}
    _builtins_registered: bool = False

    @classmethod
    def register(cls, tool: BaseTool):
        cls._tools[tool.name] = tool

    @classmethod
    def register_custom(cls, tool: BaseTool):
        """Explicit entry point for user-provided tools."""
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool:
        if not cls._builtins_registered:
            cls.auto_register_builtins()
        if name not in cls._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {cls.list_all()}")
        return cls._tools[name]

    @classmethod
    def get_many(cls, names: list[str]) -> list[BaseTool]:
        return [cls.get(n) for n in names]

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._tools.keys())

    @classmethod
    def list_tools(cls) -> list[str]:
        """Alias for list_all() for backward compatibility."""
        return cls.list_all()

    @classmethod
    def auto_register_builtins(cls):
        """Register all builtin tools. Called automatically on first get()."""
        from .builtin.api_tools import HttpGetTool, HttpPostTool
        from .builtin.code_tools import RunBashTool, RunPythonTool
        from .builtin.file_tools import ListDirTool, ReadFileTool, WriteFileTool
        from .builtin.github import GitHubTool
        from .builtin.playright import PlaywrightTool
        from .builtin.search_tools import GrepTool
        from .builtin.slack import SlackTool
        from .builtin.sql import SQLDatabaseTool
        from .builtin.web_search_tool import WebSearchTool

        builtins = [
            ReadFileTool(),
            WriteFileTool(),
            ListDirTool(),
            RunPythonTool(),
            RunBashTool(),
            GrepTool(),
            HttpGetTool(),
            HttpPostTool(),
            WebSearchTool(),
            SlackTool(),
            SQLDatabaseTool(),
            GitHubTool(),
            PlaywrightTool(),
        ]
        for tool in builtins:
            cls.register(tool)
        cls._builtins_registered = True
