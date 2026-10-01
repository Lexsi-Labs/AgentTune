# Tool Library

AgentTune ships a small library of `BaseTool` subclasses under
`agenttune.agentic.tools.builtin`, plus the infrastructure to register, discover, and
execute them (`ToolRegistry`, `ToolExecutor`) and to write your own. Every example on this
page was run against the real source in this repo, not inferred from docstrings.

**Read this first: the one gotcha that affects every tool below:**

!!! warning "`ToolRegistry.get()` can crash on an unrelated missing dependency"
    Calling `ToolRegistry.get(name)` for **any** tool, including the pure-stdlib
    `read_file`, triggers `auto_register_builtins()`, which unconditionally imports
    *every* builtin module at once, including the ones that need `langchain_community`
    (Slack, GitHub, Playwright, SQL, web search). Without `langchain_community` installed,
    fetching `read_file` through the registry raises `ImportError` even though
    `ReadFileTool` itself has no such dependency:

    ```pycon
    >>> from agenttune.agentic.tools.registry import ToolRegistry
    >>> ToolRegistry.get("read_file")
    Traceback (most recent call last):
      ...
    ModuleNotFoundError: No module named 'langchain_community'
    ```

    A base `pip install -e .` (which pulls in `langchain-community` per `pyproject.toml`)
    does **not** hit this. Confirmed live, `ToolRegistry.get("read_file")` returns cleanly
    in that install. This only bites in a leaner, partial, or pinned environment where
    `langchain_community` was left out.

    It even trips on your own custom tools; `register_custom()` doesn't touch
    `auto_register_builtins()`, but the very next `.get()` call (yours or anyone else's)
    still does. See [Known Issues](../community/known-issues.md) and the workaround in
    [Troubleshooting](troubleshooting.md).

    **The reliable workaround, if you only need the zero-dependency tools:** import the
    tool classes directly from their `builtin/` module instead of going through the
    registry: `ReadFileTool`, `WriteFileTool`, `ListDirTool`, `RunPythonTool`,
    `RunBashTool`, and `GrepTool` all import cleanly on their own, with no
    `langchain_community` in their module-level imports.

## Zero-dependency tools

These six need nothing beyond the Python standard library (`GrepTool` also shells out to
the system `grep` binary). Import them directly, no registry, no extras, no API keys.

### `ReadFileTool`

Reads a file. Real signature: `execute(path: str, encoding: str = "utf-8") -> ToolResult`.

```python
from agenttune.agentic.tools.builtin.file_tools import ReadFileTool

tool = ReadFileTool()
result = tool.execute(path="README.md")
print(result.success, result.output[:80])
# True '<p align="center">\n  <picture>\n    <source media="(prefers-color-scheme: da'
```

### `WriteFileTool`

Writes content to a file, creating parent directories as needed. Real signature:
`execute(path: str, content: str, mode: str = "w") -> ToolResult`.

```python
from agenttune.agentic.tools.builtin.file_tools import WriteFileTool

tool = WriteFileTool()
result = tool.execute(path="/tmp/notes/todo.txt", content="buy milk\n")
print(result.output)
# Written to /tmp/notes/todo.txt
```

Pass `mode="a"` to append instead of overwrite.

### `ListDirTool`

Lists entries in a directory. Real signature: `execute(path: str = ".") -> ToolResult`.

```python
from agenttune.agentic.tools.builtin.file_tools import ListDirTool

tool = ListDirTool()
result = tool.execute(path=".")
print(result.output)
# ['.git', 'CODE_OF_CONDUCT.md', 'LICENSE', 'colab', 'config.yaml', ...]
```

### `RunPythonTool`

Writes a code string to a temp `.py` file and runs it with `subprocess.run([sys.executable,
tmp_path], ...)`, returning combined stdout+stderr. Real signature: `execute(code: str,
timeout: int = 30) -> ToolResult`.

```python
from agenttune.agentic.tools.builtin.code_tools import RunPythonTool

tool = RunPythonTool()
result = tool.execute(code="print(sum(range(10)))", timeout=10)
print(result.success, result.output)
# True 45
```

`result.success` is `False` and `result.error` is set to stderr whenever the script's
return code is non-zero; a timeout instead returns `error="Timeout exceeded"`.

### `RunBashTool`

Runs a shell command via `subprocess.run(command, shell=True, ...)`. Real signature:
`execute(command: str, timeout: int = 30) -> ToolResult`.

```python
from agenttune.agentic.tools.builtin.code_tools import RunBashTool

tool = RunBashTool()
result = tool.execute(command="echo hi && wc -l README.md")
print(result.output)
```

Because `shell=True`, this executes whatever string the model produces; treat it the same
way you'd treat any other unsandboxed shell access from an LLM.

### `GrepTool`

Shells out to the system `grep` binary. Real signature: `execute(pattern: str, path: str =
".", recursive: bool = True) -> ToolResult`.

```python
from agenttune.agentic.tools.builtin.search_tools import GrepTool

tool = GrepTool()
result = tool.execute(pattern="ToolRegistry", path="src/agenttune/agentic/tools")
print(result.output)
```

`pattern` and `path` are interpolated straight into a shell string
(`f"grep {flags} '{pattern}' {path}"`) with no escaping; treat `pattern`/`path` as
untrusted input the same way you would `RunBashTool`'s `command`.

## Needs `langchain_community`

These six wrap `langchain_community` toolkits/utilities. `langchain-community` is listed as
a base dependency in `pyproject.toml`, but in a lean, partial, or pinned environment it may
simply not be present, which is exactly the failure mode in the warning above.
`HttpGetTool`/`HttpPostTool` only need
`requests` (already a transitive dependency of almost everything else here), not
`langchain_community` itself, but they still get imported by
`auto_register_builtins()` alongside the tools that do.

### `HttpGetTool` / `HttpPostTool`

Plain `requests` wrappers, no `langchain_community` needed for these two specifically.
Real signatures:

- `HttpGetTool.execute(url: str, headers: Dict = None, timeout: int = 10) -> ToolResult`
- `HttpPostTool.execute(url: str, body: Dict = None, headers: Dict = None, timeout: int = 10) -> ToolResult`

```python
from agenttune.agentic.tools.builtin.api_tools import HttpGetTool, HttpPostTool

get_tool = HttpGetTool()
result = get_tool.execute(url="https://api.github.com/zen")
print(result.success, result.output)
# True 'Approachable is better than simple.'

post_tool = HttpPostTool()
result = post_tool.execute(url="https://httpbin.org/post", body={"hello": "world"})
print(result.success)
```

`result.success` mirrors `resp.ok`; a non-2xx status is reported as failure with
`error="HTTP {status_code}"`, the response body still lands in `output`.

### `SQLDatabaseTool`

The most feature-dense tool in the library: query/inspect any SQLAlchemy-URI database via
LangChain's `ListSQLDatabaseTool`/`InfoSQLDatabaseTool`/`QuerySQLDatabaseTool`, build a
SQLite DB from a HuggingFace dataset, or use the built-in Enron-email / FinQA table
loaders and FTS5 email search helpers.

Needs: `langchain_community`, `pandas`. `build_enron_db`/`create_from_dataset` additionally
need the `datasets` library.

Construct it once with a `db_uri`; you never need to pass `db_uri` again:

```python
from agenttune.agentic.tools.builtin.sql import SQLDatabaseTool

tool = SQLDatabaseTool("sqlite:///demo.db")

# Load any HuggingFace dataset straight into a table
tool.create_from_dataset("qgallouedec/biogrid", table_name="interactions")

print(tool.list_tables().output)
# ['interactions']
print(tool.schema("interactions").output)

result = tool.query("SELECT * FROM interactions LIMIT 5")
print(result)   # a list of tuples, or {"error": ...} on failure
```

Real `execute()` signature (the generic entry point behind all the convenience methods):

```python
def execute(
    self, action: str, input: str = "", table_name: str = "data", split: str = "train",
    db_uri: Optional[str] = None, tables_root: str = "data/data/company_tables",
    max_rows: int = 50_000, inbox: str = "", keywords: Optional[List[str]] = None,
    from_addr: str = "", to_addr: str = "", sent_after: str = "", sent_before: str = "",
    max_results: int = 10, message_id: str = "", min_emails: int = 10,
) -> ToolResult: ...
```

`action` is one of: `sql_db_list_tables`, `sql_db_schema`, `sql_db_query`,
`create_from_dataset`, `build_finqa_db`, `build_enron_db`, `search_inbox`, `read_email`,
`list_senders`.

Two things worth knowing about the design:

- **`.query(sql_command)` is the method meant to be handed to a trainer directly**:
  `create_agentic_trainer(tools=[tool.query], ...)`. It returns a plain list of tuples (or
  an error dict), not a `ToolResult`, deliberately, so the model sees a clean result in
  context. `.make_query_tool(name, description)` wraps the same thing under a custom name if
  you want the model to see a specific tool name in its schema (e.g. `query_biogrid`
  instead of the generic `query`).
- **Email/Enron actions are their own FTS5-backed sub-API**: `build_enron_db()` downloads
  `corbt/enron-emails` from HuggingFace and builds a full-text-searchable SQLite schema
  (`emails` / `recipients` / `emails_fts`); `search_inbox(inbox, keywords, ...)` and
  `read_email(message_id)` query it. `search_inbox` requires `inbox` and at least one
  keyword, and caps `max_results` at 10.

```python
tool = SQLDatabaseTool("sqlite:///enron.db")
tool.build_enron_db(max_rows=50_000)

hits = tool.search_inbox(
    inbox="jeff.skilling@enron.com",
    keywords=["quarterly", "earnings"],
    max_results=5,
)
print(hits)
```

### `WebSearchTool`

DuckDuckGo search via `langchain_community.tools.DuckDuckGoSearchResults`. Needs
`langchain_community` **and**, verified live, a separately-installed `ddgs` package (the
current name for what used to be `duckduckgo-search`); `langchain_community` does not pull
this in as a hard dependency, so a base install raises `result.success=False,
result.error="Could not import ddgs python package. Please install it with \`pip install -U
ddgs\`."` until you `pip install ddgs` yourself. Also needs network access, no API key.

Real signature: `execute(query: str, output_format: str = "list", backend: str = "text") ->
ToolResult`. `backend="news"` searches news articles instead of general web results.

```python
from agenttune.agentic.tools.builtin.web_search_tool import WebSearchTool

tool = WebSearchTool()
result = tool.execute(query="AgentTune agentic RL training", output_format="list")
print(result.output)
```

### `GitHubTool`

Wraps `langchain_community.agent_toolkits.github.toolkit.GitHubToolkit` /
`GitHubAPIWrapper`. Real signature: `execute(action: str, input: str) -> ToolResult`; both
`action` and `input` are required.

Needs `langchain_community` **and** GitHub credentials in the environment
(`GitHubAPIWrapper` reads them from env vars, e.g. `GITHUB_APP_ID` /
`GITHUB_APP_PRIVATE_KEY` or a personal-access-token equivalent, plus `GITHUB_REPOSITORY`).

```python
from agenttune.agentic.tools.builtin.github import GitHubTool

tool = GitHubTool()
result = tool.execute(action="Get Issues", input="")
print(result.output)
```

Full `action` enum: `Get Issues`, `Get Issue`, `Comment on Issue`,
`List open pull requests (PRs)`, `Get Pull Request`, `Create Pull Request`, `Create File`,
`Read File`, `Update File`, `Delete File`, `Create a new branch`, `Set active branch`,
`List branches in this repository`, `Search issues and pull requests`, `Search code`.

**`GitHubTool` has zero test coverage in this repo and has never been exercised with real
credentials here**; see [Known Issues](../community/known-issues.md) before relying on it
in production.

### `SlackTool`

Wraps `langchain_community.agent_toolkits.SlackToolkit`. Real signature: `execute(action:
str, input: str) -> ToolResult`.

Needs `langchain_community` and a Slack bot token in the environment that
`SlackToolkit` picks up (`SLACK_BOT_TOKEN`, plus any additional scopes the specific action
needs).

```python
from agenttune.agentic.tools.builtin.slack import SlackTool

tool = SlackTool()
result = tool.execute(action="send_message", input='{"channel": "C0123", "message": "hi"}')
```

Full `action` enum: `get_channel`, `get_message`, `send_message`, `schedule_message`.

**`SlackTool` has zero test coverage in this repo and has never been exercised with real
credentials here**; see [Known Issues](../community/known-issues.md).

### `PlaywrightTool`

Wraps `langchain_community.agent_toolkits.PlayWrightBrowserToolkit` over a real headless
browser. Needs `langchain_community` (a base dependency), the `playwright` Python package,
**and** the browser binaries (`playwright install`); the latter two are not pulled in by
the base install.

Real signature:

```python
def execute(
    self, action: str,
    url: Optional[str] = None, selector: Optional[str] = None,
    attributes: Optional[list] = None,
) -> ToolResult: ...
```

`action` enum: `navigate_browser`, `previous_page`, `click_element`, `extract_text`,
`extract_hyperlinks`, `get_elements`, `current_webpage`. `url` is required for
`navigate_browser`; `selector` is required for `click_element` and `get_elements`;
`attributes` is an optional list for `get_elements` (e.g. `["innerText", "href"]`).

```python
from agenttune.agentic.tools.builtin.playright import PlaywrightTool
# Note: the module file is named playright.py (typo) — the class is PlaywrightTool.

tool = PlaywrightTool()
nav = tool.execute(action="navigate_browser", url="https://example.com")
text = tool.execute(action="extract_text")
print(text.output[:200])
```

Every call spins up a fresh `create_async_playwright_browser()` + toolkit internally (via
`asyncio.run(...)` around an async helper); there's no persistent browser session held
across calls, so state like navigation history only lives within one `execute()` call's
underlying async browser unless you drive it yourself.

**`PlaywrightTool` has zero test coverage in this repo and has never been exercised for
real anywhere in this codebase**; see [Known Issues](../community/known-issues.md).

## OpenEnv: remote sandboxed tools

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is a protocol + server for running tool
calls in an isolated remote environment (a container, a VM, a sandboxed process) instead of
in-process on the machine running your training loop. AgentTune's `openenv_tool.py`
(`agenttune.agentic.tools.builtin.openenv_tool`) is a thin adapter: it connects to a running
OpenEnv server over MCP, discovers whatever tools that server exposes, and wraps each one as
a local `BaseTool`, so the rest of AgentTune's rollout machinery never needs to know the
tool call actually went over a network.

`openenv>=0.3.1`, `httpx`, and `websockets` are base dependencies of `agenttune`, so a plain
`pip install -e .` already covers this, no extra install step. All `openenv` imports inside
`openenv_tool.py` are still lazy; importing `agenttune.agentic.tools.builtin.openenv_tool`
itself never touches the package; only actually calling `create_openenv_tools()` does
(guarded by `require_openenv()`, which raises `ImportError` with a reinstall hint if
`openenv` is somehow missing).

### Connecting to a running server

The real factory signature (`create_openenv_tools`, verified against
`agenttune/agentic/tools/builtin/openenv_tool.py`):

```python
def create_openenv_tools(
    base_url: Optional[str] = None,
    *,
    connect_timeout_s: float = 10.0,
    message_timeout_s: float = 60.0,
    tool_filter: Optional[List[str]] = None,
    name_prefix: Optional[str] = None,
) -> Tuple[List[OpenEnvTool], OpenEnvHandle]: ...
```

`base_url` and `name_prefix` **are** the real parameter names, confirmed directly from
source, not assumed. With a server already running (e.g. `uvicorn server.app:app --port
8000`, per the error message the function raises if it can't connect):

```python
from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

tools, handle = create_openenv_tools(
    base_url="http://localhost:8000",
    name_prefix="sandbox_",       # optional — avoids name collisions with builtins
    tool_filter=["echo_message"],  # optional — only wrap these remote tools
)

try:
    for tool in tools:
        print(tool.name, "->", tool.description)
    result = tools[0].execute(text="hello")   # kwargs forwarded verbatim to the remote tool
    print(result.success, result.output)
finally:
    handle.close()   # releases the WebSocket connection + background event loop
```

`create_openenv_tools` raises `ValueError` if `base_url` is missing or a `tool_filter` name
doesn't exist on the server, and `RuntimeError` if the server is unreachable or returns zero
tools. Every tool built from one `create_openenv_tools()` call shares a single persistent
`SyncEnvClient` session (one WebSocket); for parallel rollout workers that need isolation,
call `create_openenv_tools()` once per worker, not once globally.

`OpenEnvTool.execute(**kwargs)` is a plain (non-async) method; the underlying
`SyncEnvClient` runs the async I/O on a background thread, so it's safe to call from
AgentTune's synchronous rollout hot loop. On failure it returns
`ToolResult(success=False, output={"error": ...}, error=...)` rather than raising, so the
rollout loop's failure-counter sees it correctly.

### Registering discovered tools globally

`register_openenv_tools(...)` takes the same keyword arguments as `create_openenv_tools`
and additionally pushes every discovered tool into `ToolRegistry` via
`ToolRegistry.register_custom()`:

```python
from agenttune.agentic.tools.builtin.openenv_tool import register_openenv_tools

tools, handle = register_openenv_tools(base_url="http://localhost:8000")
# tools are now also fetchable via ToolRegistry.get(tool.name)
```

### Wiring OpenEnv tools into a rollout

`create_openenv_tools()` returns plain `BaseTool` instances, which is exactly what
`create_rollout_fn(tools=[...])` expects:

```python
from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

tools, handle = create_openenv_tools(base_url="http://localhost:8000")
try:
    rollout_fn = create_rollout_fn(
        rollout_backend="api",
        api_model="gpt-4o-mini",
        tools=tools,
        max_steps=10,
    )
    result = rollout_fn(["Use the tools to solve the task."])
finally:
    handle.close()
```

### `OpenEnvHandle`

The lifecycle object returned alongside the tools, also usable as a context manager:

```python
from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools, OpenEnvHandle

tools, handle = create_openenv_tools(base_url="http://localhost:8000")
with handle:
    ...  # handle.close() runs automatically on exit
```

Note there's a *separate* OpenEnv integration in this codebase at the harness level
(`agenttune.agentic.harness_openenv.OpenEnvHarness`), which wraps a real
`openenv.core.Environment` (`reset`/`step`) for the agentic-spine's strategy/harness/replay
machinery rather than exposing individual remote tools. See the
[Local Notebooks](../notebooks/local-notebook.md) index for that path, and [Python API:
Agentic Spine](agentic-spine.md) for how it fits the rest of the
spine. This page's `OpenEnvTool` is the tool-level adapter, not the harness-level one; pick
whichever matches how your training loop is structured.

## Writing your own tool

Every tool, builtin or custom, subclasses `BaseTool` (`agenttune.agentic.tools.base`).
The real abstract base:

```python
class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult: ...

    def _parameters(self) -> Dict:
        """Override to define input schema."""
        return {"type": "object", "properties": {}, "required": []}

    def to_schema(self) -> Dict:
        """OpenAI-compatible tool schema. Override if needed."""
        ...
```

`ToolResult` is a plain dataclass: `success: bool`, `output: Any`, `error: Optional[str] =
None`, `metadata: Dict[str, Any] = {}`.

The only thing you're required to implement is `execute()`. Overriding `_parameters()` is
optional but strongly recommended; it's what `to_schema()` uses to build the
OpenAI-compatible tool schema a model actually sees, so without it the model gets a tool
with an empty parameter list.

```python
from agenttune.agentic.tools.base import BaseTool, ToolResult

class UppercaseTool(BaseTool):
    name = "uppercase"
    description = "Uppercase a string."

    def _parameters(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to uppercase"}},
            "required": ["text"],
        }

    def execute(self, text: str) -> ToolResult:
        return ToolResult(success=True, output=text.upper())
```

Verified end to end:

```python
tool = UppercaseTool()
print(tool.execute(text="hi there").output)
# HI THERE
print(tool.to_schema())
# {'type': 'function', 'function': {'name': 'uppercase', 'description': 'Uppercase a
#  string.', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string',
#  ...}}, 'required': ['text']}}}
```

You don't have to route custom tools through `ToolRegistry` at all; a bare list of
`BaseTool` instances (or even plain Python callables, if you're building the tool schema
yourself, as `finqa_tool.py`'s `build_finqa_tools()` does) works directly with
`tools=[...]` on `create_rollout_fn`/`create_agentic_trainer`.

## `ToolRegistry` and `ToolExecutor`

`ToolRegistry` (`agenttune.agentic.tools.registry`) is a classmethod-only, process-global
registry; there's no instance to construct.

```python
from agenttune.agentic.tools.registry import ToolRegistry

ToolRegistry.register_custom(UppercaseTool())   # explicit entry point for your own tools
tool = ToolRegistry.get("uppercase")             # KeyError if not found
names = ToolRegistry.list_all()                  # (list_tools() is an alias)
```

`register_custom()` just does `cls._tools[tool.name] = tool`; it doesn't touch
`auto_register_builtins()`. But the very next `.get()` call, yours or anyone else's in the
same process, checks `cls._builtins_registered` and, if it's still `False`, imports every
builtin module before looking your tool up. This is the same failure mode as the warning at
the top of this page, just reached via a different path: **registering your own tool does
not protect you from the `langchain_community` ImportError on the first `.get()` call.** See
[Troubleshooting](troubleshooting.md) for the workaround (setting
`ToolRegistry._builtins_registered = True` yourself before the first `.get()`, if you don't
need the langchain-backed builtins).

`ToolExecutor` (`agenttune.agentic.tools.executor`) wraps registry lookup + execution with a
timeout, and normalizes both sync and async `execute()` methods:

```python
import asyncio
from agenttune.agentic.tools.executor import ToolExecutor

executor = ToolExecutor(ToolRegistry)
output = asyncio.run(executor.run("uppercase", {"text": "hi there"}, timeout_sec=10))
print(output)
# HI THERE
```

`ToolExecutor.run(tool_name, args, timeout_sec=30)` returns the tool's raw `output` on
success (unwrapping `ToolResult`), raises `Exception(result.error)` if `result.success` is
`False`, raises `KeyError` if the tool isn't registered, and raises `TimeoutError` if
execution exceeds `timeout_sec`. If `execute()` returns a coroutine, it's awaited
automatically. Write your custom tool's `execute()` as either a plain function or an
`async def`; `ToolExecutor` handles both.

## Other tool-adjacent modules

Two more files under `builtin/` are worth knowing about even though they aren't registered
in `auto_register_builtins()`:

- **`FileIngestionTool`** (`builtin/file_ingestion.py`): generates synthetic directories of
  small CSV/TXT/LOG files for training file-reading agents, sized so a full
  `list_dir → read_file × 2 → run_python → write_file` rollout stays comfortably under
  common `max_completion_length` limits. `tool.execute(action="generate", n_samples=200)`
  builds the dataset; `action="list_dir"`/`"read_file"` are the read-only tool-callable
  actions. Pure stdlib + `datasets`.
- **`finqa_tool.py`**: not a `BaseTool` at all; `build_finqa_tools(db_tool)` returns three
  plain Python callables (`list_tables`, `get_table_schema`, `query_finqa_tables`) bound to
  a `SQLDatabaseTool` instance, plus a standalone `calculator(expression, variables=None)`
  function, all shaped to be handed directly to `create_agentic_trainer(tools=[...])`
  without wrapping them in `BaseTool` at all, a reminder that `tools=` accepts plain
  callables with a docstring just as readily as `BaseTool` instances.

See the [Local Notebooks](../notebooks/local-notebook.md) index for the tested tools running
end to end, and for the OpenEnv notebook.
