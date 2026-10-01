import asyncio

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser

from ..base import BaseTool, ToolResult


class PlaywrightTool(BaseTool):
    name = "playwright_browser"
    description = "Control a browser to navigate websites, extract text, click elements, and interact with dynamic web pages."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The browser action to perform.",
                    "enum": [
                        "navigate_browser",
                        "previous_page",
                        "click_element",
                        "extract_text",
                        "extract_hyperlinks",
                        "get_elements",
                        "current_webpage",
                    ],
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to. Required for 'navigate_browser'.",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for 'click_element' or 'get_elements'.",
                },
                "attributes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attributes to extract for 'get_elements' (e.g. ['innerText', 'href']).",
                },
            },
            "required": ["action"],
        }

    def execute(
        self,
        action: str,
        url: str | None = None,
        selector: str | None = None,
        attributes: list | None = None,
    ) -> ToolResult:
        try:
            result = asyncio.run(self._async_execute(action, url, selector, attributes))
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def _async_execute(
        self,
        action: str,
        url: str | None,
        selector: str | None,
        attributes: list | None,
    ) -> str:
        async_browser = create_async_playwright_browser()
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
        tools_by_name = {tool.name: tool for tool in toolkit.get_tools()}

        if action not in tools_by_name:
            raise ValueError(f"Unknown action '{action}'. Available: {list(tools_by_name.keys())}")

        tool = tools_by_name[action]

        if action == "navigate_browser":
            if not url:
                raise ValueError("'url' is required for navigate_browser.")
            result = await tool.arun({"url": url})

        elif action == "click_element":
            if not selector:
                raise ValueError("'selector' is required for click_element.")
            result = await tool.arun({"selector": selector})

        elif action == "get_elements":
            if not selector:
                raise ValueError("'selector' is required for get_elements.")
            input_data = {"selector": selector}
            if attributes:
                input_data["attributes"] = attributes
            result = await tool.arun(input_data)

        elif action in ("extract_text", "extract_hyperlinks", "current_webpage", "previous_page"):
            result = await tool.arun({})

        else:
            raise ValueError(f"Unhandled action: '{action}'.")

        return result
