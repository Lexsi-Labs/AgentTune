from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Base class for all tools (builtin and user-defined).

    To add a custom tool:
        class MyTool(BaseTool):
            name = "my_tool"
            description = "Does something useful"

            def _parameters(self):
                return {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"]
                }

            def execute(self, input: str) -> ToolResult:
                return ToolResult(success=True, output=f"processed: {input}")
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def _parameters(self) -> dict:
        """Override to define input schema."""
        return {"type": "object", "properties": {}, "required": []}

    def to_schema(self) -> dict:
        """OpenAI-compatible tool schema. Override if needed."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters(),
            },
        }

    def __repr__(self):
        return f"<Tool name='{self.name}'>"
