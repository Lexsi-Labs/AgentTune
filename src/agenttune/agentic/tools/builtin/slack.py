from langchain_community.agent_toolkits import SlackToolkit

from ..base import BaseTool, ToolResult


class SlackTool(BaseTool):
    name = "slack"
    description = (
        "Interact with Slack to read channels, get messages, and send or schedule messages."
    )

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The Slack action to perform.",
                    "enum": [
                        "get_channel",
                        "get_message",
                        "send_message",
                        "schedule_message",
                    ],
                },
                "input": {
                    "type": "string",
                    "description": "The input for the action (e.g. channel ID, message text, or JSON string with required fields).",
                },
            },
            "required": ["action", "input"],
        }

    def execute(self, action: str, input: str) -> ToolResult:
        try:
            toolkit = SlackToolkit()

            tools = {tool.name: tool for tool in toolkit.get_tools()}

            if action not in tools:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action '{action}'. Available: {list(tools.keys())}",
                )

            result = tools[action].run(input)
            return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
