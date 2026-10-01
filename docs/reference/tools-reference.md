# Python API: Tools

`agenttune.agentic.tools`: the tool-calling primitives every agent strategy and rollout
loop dispatches through: a `BaseTool` ABC, a process-wide `ToolRegistry`, an async
`ToolExecutor`, and 13 auto-registered builtins plus 3 more that exist but aren't
auto-registered. This page is the terse signature/params reference. For worked, runnable
examples of each tool see [User Guide: Tool Library](../user-guide/tool-library.md); for
where tools fit in the rest of the agentic spine see
[Python API: Agentic Spine](../user-guide/agentic-spine.md).

## `BaseTool` / `ToolResult`: the base contract

```python
from agenttune.agentic.tools.base import BaseTool, ToolResult

@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

`BaseTool` (`tools/base.py`) is an ABC with two class attributes and three methods:

| Member | Signature | Required? |
|---|---|---|
| `name` | `str` class attribute | Yes: the key it registers under |
| `description` | `str` class attribute | Yes: goes straight into the schema |
| `execute(**kwargs)` | `-> ToolResult` | **Abstract**: every subclass must implement it |
| `_parameters()` | `-> Dict` | Override to declare the JSON-Schema `parameters` block. Default is `{"type": "object", "properties": {}, "required": []}` |
| `to_schema()` | `-> Dict` | **Not** abstract; reusable as-is |

`to_schema()` is the one genuinely reusable piece worth calling out on its own: it builds an
OpenAI-compatible function-calling schema straight from `name`/`description`/`_parameters()`
with no dependency on the registry:

```python
{
    "type": "function",
    "function": {
        "name": self.name,
        "description": self.description,
        "parameters": self._parameters(),
    },
}
```

Any `BaseTool` subclass, builtin or your own, gets a working `.to_schema()` for free the
moment `_parameters()` is implemented; you never need to hand-write the OpenAI wrapper.

Minimal custom tool (from the class docstring):

```python
class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"

    def _parameters(self):
        return {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}

    def execute(self, input: str) -> ToolResult:
        return ToolResult(success=True, output=f"processed: {input}")
```

## `ToolRegistry` and `ToolExecutor`

`ToolRegistry` (`tools/registry.py`) is a classmethod-only registry; there's no instance,
just class-level state:

| Method | What it does |
|---|---|
| `ToolRegistry.register(tool)` | Adds/overwrites `tool.name` in the registry |
| `ToolRegistry.register_custom(tool)` | Identical to `register()`: the documented entry point for user tools |
| `ToolRegistry.get(name)` | Triggers `auto_register_builtins()` on first call ever, then looks up `name`; raises `KeyError` (listing everything registered) if not found |
| `ToolRegistry.get_many(names)` | `[cls.get(n) for n in names]` |
| `ToolRegistry.list_all()` / `.list_tools()` | Same thing; `list_tools()` is a back-compat alias |
| `ToolRegistry.auto_register_builtins()` | Imports and instantiates all 13 builtins below with zero constructor args, once |

!!! warning "`ToolRegistry.get()` can fail on an unrelated missing dependency"
    `auto_register_builtins()` imports **every** builtin module unconditionally, including
    the `langchain_community`-dependent ones (SQL/web-search/Slack/GitHub/Playwright).
    Without `langchain_community` installed, even `ToolRegistry.get("read_file")`, pure
    stdlib, raises `ImportError`. See [Known Issues](../community/known-issues.md).

Also note: the registry instantiates `SQLDatabaseTool()` with no `db_uri`. A copy fetched via
`ToolRegistry.get("sql_database")` will raise `ValueError` on every action until you
construct your own instance with a URI (`SQLDatabaseTool("sqlite:///mydb.db")`) and
`register_custom()` it; the registry's stateless auto-instantiation doesn't fit a tool that
needs constructor state.

`ToolExecutor` (`tools/executor.py`) wraps a registry with timeout + `ToolResult` unwrapping:

```python
class ToolExecutor:
    def __init__(self, registry: ToolRegistry): ...
    async def run(self, tool_name: str, args: Dict[str, Any], timeout_sec: int = 30) -> Any: ...
```

`run()` looks up the tool, calls `tool.execute(**args)` (awaiting it if it returned a
coroutine), and then: if the result isn't a `ToolResult`, returns it as-is; if it is one and
`success=False`, raises `Exception(result.error or "Tool execution failed")`; otherwise
returns `result.output`. The whole call is wrapped in `asyncio.wait_for`, converting a real
timeout into `TimeoutError(f"Tool execution timed out after {timeout_sec}s")`.

## The 13 auto-registered builtins

These are exactly what `ToolRegistry.auto_register_builtins()` instantiates, fetchable by
name via `ToolRegistry.get(...)` immediately, no manual registration needed. Actions/params
are **not uniform** across tools; each row is that tool's real `execute()` signature.

| Class | File | `execute()`: actions / params | What it does | Needs |
|---|---|---|---|---|
| `ReadFileTool` (`read_file`) | `file_tools.py` | `(path: str, encoding: str = "utf-8")` | Reads a file, returns contents as a string | Nothing (stdlib) |
| `WriteFileTool` (`write_file`) | `file_tools.py` | `(path: str, content: str, mode: str = "w")` | Writes a file, creating parent dirs via `os.makedirs` | Nothing |
| `ListDirTool` (`list_dir`) | `file_tools.py` | `(path: str = ".")` | `os.listdir(path)` | Nothing |
| `RunPythonTool` (`run_python`) | `code_tools.py` | `(code: str, timeout: int = 30)` | Writes `code` to a temp `.py` file, runs it with `sys.executable`, returns combined stdout+stderr | Nothing (no sandbox, runs arbitrary code as your own process) |
| `RunBashTool` (`run_bash`) | `code_tools.py` | `(command: str, timeout: int = 30)` | `subprocess.run(command, shell=True, ...)` | Nothing (no sandbox) |
| `GrepTool` (`grep`) | `search_tools.py` | `(pattern: str, path: str = ".", recursive: bool = True)` | Shells out to `grep [-r] '{pattern}' {path}`; `pattern`/`path` are interpolated unescaped into the shell string | System `grep` binary |
| `HttpGetTool` (`http_get`) | `api_tools.py` | `(url: str, headers: Dict = None, timeout: int = 10)` | `requests.get` | `requests` |
| `HttpPostTool` (`http_post`) | `api_tools.py` | `(url: str, body: Dict = None, headers: Dict = None, timeout: int = 10)` | `requests.post(json=body)` | `requests` |
| `WebSearchTool` (`web_search`) | `web_search_tool.py` | `(query: str, output_format: str = "list", backend: str = "text")`; `backend` ∈ `text`\|`news` | DuckDuckGo search via `langchain_community`'s `DuckDuckGoSearchResults` | `langchain_community` + network |
| `SQLDatabaseTool` (`sql_database`) | `sql.py` | See dedicated section below: 9 actions, not a flat schema | Query/inspect SQL, build demo DBs, or search a pre-built email inbox | `langchain_community` + `pandas`; `datasets` for the dataset/Enron actions |
| `GitHubTool` (`github`) | `github.py` | `(action: str, input: str)`; `action` is one of the 15 real tool names from `GitHubToolkit.get_tools()` (`"Get Issues"`, `"Create Pull Request"`, `"Read File"`, `"Search code"`, ...) | Delegates to LangChain's GitHub toolkit | `langchain_community` + a GitHub token via `GitHubAPIWrapper`'s env vars. **Zero test coverage, never exercised with real credentials in this repo**; see [Known Issues](../community/known-issues.md) |
| `SlackTool` (`slack`) | `slack.py` | `(action: str, input: str)`; `action` ∈ `get_channel`\|`get_message`\|`send_message`\|`schedule_message` | Delegates to LangChain's `SlackToolkit` | `langchain_community` + a Slack token. Same untested caveat |
| `PlaywrightTool` (`playwright_browser`) | `playright.py` *(sic, filename typo in the repo)* | `(action: str, url=None, selector=None, attributes=None)`; `action` ∈ `navigate_browser`\|`previous_page`\|`click_element`\|`extract_text`\|`extract_hyperlinks`\|`get_elements`\|`current_webpage` | Drives a headless Playwright browser via LangChain's toolkit | `langchain_community` + `playwright` + browser binaries. Same untested caveat. Calls `asyncio.run()` internally; don't call `execute()` from inside an already-running event loop |

### `SQLDatabaseTool` in detail

Constructor takes the URI once (`SQLDatabaseTool(db_uri="sqlite:///enron.db")`), and every
call after that reuses it unless you pass an explicit override:

```python
execute(
    action: str,               # required — see enum below
    input: str = "",
    table_name: str = "data",
    split: str = "train",
    db_uri: Optional[str] = None,
    tables_root: str = "data/data/company_tables",
    max_rows: int = 50_000,
    inbox: str = "", keywords: Optional[List[str]] = None,
    from_addr: str = "", to_addr: str = "",
    sent_after: str = "", sent_before: str = "",
    max_results: int = 10, message_id: str = "", min_emails: int = 10,
) -> ToolResult
```

`action` is one of: `sql_db_list_tables`, `sql_db_schema`, `sql_db_query`,
`create_from_dataset`, `build_finqa_db`, `build_enron_db`, `search_inbox`, `read_email`,
`list_senders`. The first three delegate to LangChain's `ListSQLDatabaseTool` /
`InfoSQLDatabaseTool` / `QuerySQLDatabaseTool`; the email actions (`search_inbox`,
`read_email`, `list_senders`) run hand-written SQLite/FTS5 queries against a schema this
tool itself creates via `build_enron_db`.

Beyond `execute()`, the class exposes convenience methods worth knowing about since they're
built to be passed directly as plain-callable tools (not through the registry):

| Method | Signature | Notes |
|---|---|---|
| `.query(sql_command)` | `-> list` | Universal read-only query, unwraps the `ToolResult` into a plain list of tuples (or `{"error": ...}`). Doesn't support FTS5 `MATCH`; use `search_inbox` for that. Docstring literally shows passing it straight into a trainer: `create_agentic_trainer(tools=[tool.query], ...)` |
| `.make_query_tool(name, description)` | `-> callable` | Returns a *renamed* closure over `.query()` with a custom docstring; use when the model needs to see a specific tool name in its schema instead of the generic `query` |
| `.create_from_dataset(dataset_name, table_name="data", split="train")` | `-> ToolResult` | Loads a HuggingFace dataset and writes it into a table; only supports `sqlite:///` URIs |
| `.list_tables()` / `.schema(tables="")` | `-> ToolResult` | Thin wrappers around the `sql_db_list_tables` / `sql_db_schema` actions |
| `.build_enron_db(max_rows=50_000)` | `-> ToolResult` | Downloads `corbt/enron-emails` from HF and builds the `emails`/`recipients`/`emails_fts` schema the search actions expect |
| `.build_finqa_db(tables_root="data/data/company_tables")` | `-> ToolResult` | Builds a SQLite DB from FinQA's per-company `tables_cleaned_all_companies.json` files. A near-duplicate of the free function `build_finqa_db()` in `finqa_tool.py` (below) exists: same logic, two places |

## Real but not auto-registered

Three more tool-shaped things live under `tools/builtin/` but are **absent** from
`auto_register_builtins()`; `ToolRegistry.get(...)` will never find them. Use them by
importing directly.

| Name | File | How to get it | What it does | Needs |
|---|---|---|---|---|
| `OpenEnvTool` | `openenv_tool.py` | `create_openenv_tools(base_url, ...) -> (list[OpenEnvTool], OpenEnvHandle)`, or `register_openenv_tools(...)` (same, plus pushes each into `ToolRegistry.register_custom()`) | Wraps one remote OpenEnv MCP tool per instance. `execute(**kwargs)` is a plain `def` (not async) that forwards kwargs verbatim to `SyncEnvClient.call_tool()` on a background thread, returning `ToolResult(success=False, output={"error": ...})` on `RuntimeError` so rollout failure-counters trip correctly | A running OpenEnv server (openenv itself is a base dependency, already installed). `OpenEnvHandle.close()` (or the context-manager form) must be called to release the WebSocket session |
| `FileIngestionTool` | `file_ingestion.py` | `FileIngestionTool(base_dir=None, seed=42, system_prompt=None)`, instantiate directly | `execute(action, n_samples=200, sample_dir=None, filename=None)`, `action` ∈ `generate`\|`list_dir`\|`read_file`. Generates synthetic quarterly-report directories (`revenue.csv`/`expenses.csv`/`notes.txt`/`server.log`) deliberately sized so a full `list_dir→read_file×2→run_python→write_file` rollout stays under ~580 tokens, avoiding TRL's GRPO agentic trainer's completion_mask/tool_mask shape-mismatch on long rollouts. Also exposes `.get_dataset()` and the module-level `generate_file_ingestion_dataset(...)` | `datasets` |
| FinQA tool functions | `finqa_tool.py` | `build_finqa_tools(db_tool)` and `calculator`, plain functions, not `BaseTool` subclasses at all | See below | `asteval`, `huggingface_hub`, `pandas`, `datasets` |

`finqa_tool.py` is meant to be dropped straight into `create_agentic_trainer(tools=[...])`
rather than routed through the registry:

```python
db_tool = SQLDatabaseTool("sqlite:///finqa.db")
list_tables, get_table_schema, query_finqa_tables = build_finqa_tools(db_tool)
trainer = create_agentic_trainer(tools=[list_tables, get_table_schema, query_finqa_tables, calculator], ...)
```

- **`build_finqa_tools(db_tool: SQLDatabaseTool)`** returns `(list_tables, get_table_schema,
  query_finqa_tables)`, three closures bound to `db_tool`, each returning plain strings
  (`query_finqa_tables` JSON-encodes its rows) so the model can read them directly.
  `query_finqa_tables`'s exact name is important: [rewards-reference.md](rewards-reference.md)'s
  `sql_grounding_reward` greps the completion text for the literal string
  `"query_finqa_tables"`; renaming this function silently breaks that reward.
- **`calculator(expression: str, variables: dict = {})`**: standalone, no SQL dependency.
  Evaluates arithmetic via `asteval`, normalizing `$`/`€`/`£`, `%`, `K`/`M`/`B`/`T` suffixes,
  thousands-commas, unicode minus/full-width digits, and named placeholders (e.g.
  `"Total Cash 2023"` → sanitized to a valid identifier). Returns a `float`, or a string
  starting with `"Error:"` on failure; never raises.
- **`load_finqa_datasets(data_dir=..., force_download=False)`** downloads and extracts
  `rLLM/rLLM-FinQA-Dataset` from the HF Hub and returns `(train_dataset, val_dataset)` ready
  for `create_agentic_trainer`.

## See also

- [User Guide: Tool Library](../user-guide/tool-library.md): tutorial-style worked examples
  for each tool above.
- [Python API: Agentic Spine](../user-guide/agentic-spine.md): where tools sit relative
  to strategies, harnesses, and rollout engines.
- [Known Issues](../community/known-issues.md): the `ToolRegistry.get()` import-order gotcha,
  and the untested `GitHubTool`/`SlackTool`/`PlaywrightTool` caveat, in full.
