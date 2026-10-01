from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

from ..base import BaseTool, ToolResult


class GitHubTool(BaseTool):
    name = "github"
    description = "Interact with a GitHub repository to manage issues, pull requests, and files."

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The GitHub action to perform.",
                    "enum": [
                        "Get Issues",
                        "Get Issue",
                        "Comment on Issue",
                        "List open pull requests (PRs)",
                        "Get Pull Request",
                        "Create Pull Request",
                        "Create File",
                        "Read File",
                        "Update File",
                        "Delete File",
                        "Create a new branch",
                        "Set active branch",
                        "List branches in this repository",
                        "Search issues and pull requests",
                        "Search code",
                    ],
                },
                "input": {
                    "type": "string",
                    "description": "The input required for the action (e.g. issue number, file path, branch name, search query).",
                },
            },
            "required": ["action", "input"],
        }

    def execute(self, action: str, input: str) -> ToolResult:
        try:
            github = GitHubAPIWrapper()
            toolkit = GitHubToolkit.from_github_api_wrapper(github)

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
