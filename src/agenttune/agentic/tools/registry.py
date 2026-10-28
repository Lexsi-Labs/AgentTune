import importlib

from .base import BaseTool

# Builtin tool name -> (module under .builtin, class name). Each builtin is imported
# on first get() of its name, so fetching one tool never imports another tool's
# dependencies (e.g. langchain_community for slack/github/playwright/sql/web_search).
_BUILTINS: dict[str, tuple[str, str]] = {
    "read_file": ("file_tools", "ReadFileTool"),
    "write_file": ("file_tools", "WriteFileTool"),
    "list_dir": ("file_tools", "ListDirTool"),
    "run_python": ("code_tools", "RunPythonTool"),
    "run_bash": ("code_tools", "RunBashTool"),
    "grep": ("search_tools", "GrepTool"),
    "http_get": ("api_tools", "HttpGetTool"),
    "http_post": ("api_tools", "HttpPostTool"),
    "web_search": ("web_search_tool", "WebSearchTool"),
    "slack": ("slack", "SlackTool"),
    "sql_database": ("sql", "SQLDatabaseTool"),
    "github": ("github", "GitHubTool"),
    "playwright_browser": ("playright", "PlaywrightTool"),
}


class ToolRegistry:
    """
    Central registry for all tools.
    Builtins are imported and registered one at a time, on first get() of their name.
    Users can add tools via register_custom(); a tool registered under a builtin's
    name replaces that builtin.

    To add a new builtin category:
        1. Create src/agenttune/agentic/tools/builtin/my_tools.py
        2. Add a name -> (module, class) entry per tool to _BUILTINS
        3. Done
    """

    _tools: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool):
        cls._tools[tool.name] = tool

    @classmethod
    def register_custom(cls, tool: BaseTool):
        """Explicit entry point for user-provided tools."""
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool:
        if name not in cls._tools and name in _BUILTINS:
            cls._register_builtin(name)
        if name not in cls._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {cls.list_all()}")
        return cls._tools[name]

    @classmethod
    def get_many(cls, names: list[str]) -> list[BaseTool]:
        return [cls.get(n) for n in names]

    @classmethod
    def list_all(cls) -> list[str]:
        """Registered tools plus builtins that get() imports on first fetch."""
        return list(dict.fromkeys([*cls._tools, *_BUILTINS]))

    @classmethod
    def list_tools(cls) -> list[str]:
        """Alias for list_all() for backward compatibility."""
        return cls.list_all()

    @classmethod
    def auto_register_builtins(cls):
        """Register every builtin not already registered, importing all of their modules.

        get() doesn't need this. Use it to load the whole builtin set up front; it
        requires every builtin's dependencies to be installed.
        """
        for name in _BUILTINS:
            if name not in cls._tools:
                cls._register_builtin(name)

    @classmethod
    def _register_builtin(cls, name: str):
        module_name, class_name = _BUILTINS[name]
        module = importlib.import_module(f".builtin.{module_name}", __package__)
        cls.register(getattr(module, class_name)())
