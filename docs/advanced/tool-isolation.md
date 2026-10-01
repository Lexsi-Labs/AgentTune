# Tool isolation: process-level vs. network-level

Two independent, real mechanisms let a tool run somewhere other than directly inside the
training/inference process, for two different threat models. This page compares them
directly; for OpenEnv's actual setup steps (installing the extra, starting a server), see
[User Guide: Tool Library](../user-guide/tool-library.md); this page assumes that's done and
focuses on what each mechanism actually guarantees.

| | `IsolatedTool` (process-level) | OpenEnv (network-level) |
|---|---|---|
| **Where the tool runs** | A separate OS *process* on the **same machine**, via `concurrent.futures.ProcessPoolExecutor` | A **real remote server** (or a separate local process reachable over HTTP/WebSocket) |
| **Isolation boundary** | Process boundary: separate memory space, crash-isolated, but same host, same filesystem, same network namespace | Network boundary: a genuinely separate machine/container is possible; the tool never shares memory, filesystem, or process table with the caller at all |
| **Setup cost** | Zero, no extra dependency, no server to run. Works with any `BaseTool` today. | Needs a running OpenEnv server reachable at a URL (`openenv` itself is already a base dependency) |
| **Failure mode if unavailable** | Automatic in-process fallback: the tool still runs, just not isolated | Hard failure: `RuntimeError` if the server is unreachable, `ImportError` if `openenv` isn't installed |
| **What it protects against** | A tool that crashes, hangs, or leaks memory, contained to a disposable child process | Everything `IsolatedTool` protects against, *plus* a tool the training process should never be able to reach directly (different trust domain, different machine, different egress rules) |
| **Timeout handling** | Real: `future.result(timeout=timeout_s)`, catches `FutureTimeout` | Best-effort: `message_timeout_s` is set at client-construction time; the wrapper itself does not re-check it per call |
| **Real implementation** | `decide/closed_loop/tool_isolation.py`: `IsolatedTool`, `isolate_tools()` | `agentic/harness_openenv.py` (`OpenEnvHarness`, gym-like `reset()`/`step()`) and `agentic/tools/builtin/openenv_tool.py` (`OpenEnvTool`, wraps one remote tool as a `BaseTool`) |

## `IsolatedTool`: process-level isolation

The module docstring in `decide/closed_loop/tool_isolation.py` states the goal plainly:

> Run a tool OFF the training process so a crashing / slow / untrusted tool cannot take down
> the rollout loop or read the training process's memory. This is "Level 1" isolation: a
> separate OS *process* on the same host... Levels 2–3 (containers, remote hosts) are a later
> workstream and explicitly out of this sprint.

`IsolatedTool` wraps any `BaseTool` and presents the exact same interface (`name`,
`description`, `to_schema`, `execute`); the caller (a rollout loop, a strategy) cannot tell
an isolated tool from a local one:

```python
class IsolatedTool(BaseTool):
    def __init__(self, tool: BaseTool, timeout_s: float = 30.0, require_isolation: bool = False) -> None:
        self._tool = tool
        self.name = getattr(tool, "name", tool.__class__.__name__)
        self.description = getattr(tool, "description", "")
        self._timeout_s = timeout_s
        self._require_isolation = require_isolation
        self._picklable = self._check_picklable(tool)   # decided once, at wrap time
```

`_check_picklable` runs `pickle.dumps(tool)` up front; a child process needs a picklable
tool to receive it at all, so this is checked eagerly rather than discovered on first call.

### `execute()`: the real dispatch and timeout path

```python
def execute(self, **kwargs: Any) -> ToolResult:
    if not self._picklable:
        if self._require_isolation:
            return ToolResult(success=False, output={"error": "isolation required but tool is not picklable"},
                              error="isolation_unavailable", metadata={"isolation": "unavailable", "tool": self.name})
        return self._run_in_process(kwargs, reason="not_picklable")

    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_tool_in_worker, self._tool, kwargs)
            result = future.result(timeout=self._timeout_s)
        if isinstance(result, ToolResult):
            result.metadata = {**(result.metadata or {}), "isolation": "process", "tool": self.name}
            return result
        return ToolResult(success=True, output=result, metadata={"isolation": "process", "tool": self.name})
    except FutureTimeout:
        if self._require_isolation:
            return ToolResult(success=False, output={"error": msg}, error="timeout", ...)
        return self._run_in_process(kwargs, reason="timeout")
    except Exception as exc:
        if self._require_isolation:
            return ToolResult(success=False, output={"error": str(exc)}, error="isolation_failed", ...)
        return self._run_in_process(kwargs, reason="isolation_error")
```

A fresh `ProcessPoolExecutor(max_workers=1)` is spun up **per call**, the tool call is
submitted to the module-level worker function `_run_tool_in_worker` (top-level so it's
importable/picklable by the child), and `future.result(timeout=self._timeout_s)` is a real,
enforced timeout; `FutureTimeout` is caught explicitly and distinguished from any other
exception.

**The fallback guarantee is the whole design.** Every failure path (not picklable, timed
out, or any other exception during process dispatch) falls back to running the tool
**in-process** (`_run_in_process`, which just calls `self._tool.execute(**kwargs)` directly)
*unless* `require_isolation=True`, in which case the same failure becomes an error
`ToolResult` instead. Either way, every returned `ToolResult.metadata["isolation"]` records
exactly what happened: `"process"`, `"in_process_fallback"` (with a `"fallback_reason"` of
`"not_picklable"` / `"timeout"` / `"isolation_error"`), or `"unavailable"`, so isolation
status is always observable on the result, not silently assumed.

### Worked example

```python
from agenttune.decide.closed_loop.tool_isolation import isolate_tools
from agenttune.agentic.tools.builtin.file_tools import ReadFileTool

tools = isolate_tools([ReadFileTool()], timeout_s=5.0)
result = tools[0].execute(path="/etc/hostname")

print(result.success, result.metadata["isolation"])
# True 'process'   -- ReadFileTool is a plain class with no live sockets/connections,
#                     so it pickles fine and the read happens in a child process.
```

A tool that holds genuinely unpicklable state (the module docstring gives the OpenEnv adapter
itself as the example, it owns a live WebSocket) is detected up front by
`_check_picklable` and runs in-process automatically, with a logged warning, still safe
(same process crash-isolation you'd have anyway), just not actually isolated:

```python
tools = isolate_tools([my_websocket_backed_tool])
# logs: "IsolatedTool('my_tool'): wrapped tool is not picklable; calls will run
#        in-process (still safe, not isolated)."
result = tools[0].execute(**kwargs)
print(result.metadata)   # {'isolation': 'in_process_fallback', 'fallback_reason': 'not_picklable', ...}
```

`require_isolation=True` turns that same situation into a hard failure, no silent fallback,
for callers that must never run untrusted code in-process:

```python
tools = isolate_tools([my_websocket_backed_tool], require_isolation=True)
result = tools[0].execute(**kwargs)
print(result.success, result.error)   # False 'isolation_unavailable'
```

## OpenEnv: network-level isolation

OpenEnv gives up the "zero setup" property in exchange for a real, stronger isolation
boundary: the tool can live on a different machine entirely, not just a different process on
the same one. Two separate adapters exist depending on what shape of thing you're wrapping.

### `OpenEnvTool`: one remote tool as a `BaseTool`

`agentic/tools/builtin/openenv_tool.py`'s `create_openenv_tools()` connects to a running
OpenEnv server and returns one `OpenEnvTool` per remote tool the server exposes; each one
still satisfies the same `BaseTool` interface the rollout loop already expects:

```python
from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

tools, handle = create_openenv_tools(base_url="http://localhost:8000")
try:
    result = tools[0].execute(some_arg="value")   # forwarded to the remote tool verbatim
finally:
    handle.close()   # releases the WebSocket session
```

Every real OpenEnv import in this file is lazy, guarded by `require_openenv()`; a default
install that never calls `create_openenv_tools()` never imports `openenv` at all, so this
module loads fine without the extra. `execute()` is a plain (non-async) method; the
underlying `SyncEnvClient` runs the async I/O on a background thread, and any `RuntimeError`
from the remote environment is turned into `ToolResult(success=False, output={"error": ...},
error=...)` so a failing remote tool counts as a rollout failure the same way a local
exception would, rather than crashing the loop. There's also a real safety check baked in:
`create_openenv_tools` warns explicitly if `base_url` isn't local (`_is_local_url` checks for
`localhost`/`127.x`/`::1`): "Ensure credentials and training data are not leaked over the
network", since a genuinely remote server is exactly the point of this isolation level, and
exactly the thing that needs a second look before you point training data at it.

### `OpenEnvHarness`: a whole gym-like environment as a `Harness`

`agentic/harness_openenv.py` adapts a full OpenEnv `Environment` object (anything exposing
`reset()`/`step(action)`) to the spine's `Harness` interface, so `run_episode`,
`run_conformance`, and `replay` all work against it unchanged; see
[Python API: Agentic Spine](../user-guide/agentic-spine.md).
The env object is injected, not constructed here; `OpenEnvHarness` itself never imports
`openenv`, which is why the class is testable today with a hand-rolled fake env, no GPU/
network/openenv install required:

```python
class OpenEnvHarness(Harness):
    def __init__(self, env, *, action_space=None, action_adapter=None, max_steps=None) -> None:
        self._env = env
        self.capabilities = HarnessCapabilities(
            supports_snapshot=False,        # remote envs have no in-process snapshot/restore
            supports_streaming=False,
            supports_tool_boundary_interrupt=False,
            supports_stepwise_turns=True,
            max_steps=max_steps,
        )
```

`step()` records the **original dict action** into its own `event_log` before converting it
via `action_adapter` for the real `env.step()` call, so `replay()`/`to_eval_dict()` still
round-trip the action a caller actually issued, even though the environment itself received a
converted, env-native action object. `capabilities.supports_snapshot=False` is set
unconditionally and is a real, structural limitation, not a placeholder: a remote environment
has no in-process state to snapshot and restore, so this always reads `False` for any
`OpenEnvHarness`, regardless of what the wrapped env can do; `run_conformance` sees a
consistent, honest `False` here rather than something that might work depending on the
backend.

### Via YAML: `TrainerConfigBridge`

Both `IsolatedTool` and OpenEnv tools are reachable from a `TrainerConfigBridge` YAML config
(`decide/trainer_config_bridge.py`, see [Production compositions](production-compositions.md)
for the full bridge). An OpenEnv tool block:

```yaml
training:
  tools:
    - type: openenv
      base_url: "http://localhost:8000"
      name_prefix: "sandbox_"
      tool_filter: ["echo_message"]
      connect_timeout_s: 10
      message_timeout_s: 60
```

`TrainerConfigBridge._build_openenv_tools` builds the tools, then registers the returned
`OpenEnvHandle.close` via `atexit`, so the WebSocket session is released cleanly even if
training crashes, and additionally stashes the handle on `self._openenv_handles` so
`close_openenv_handles()` can close it explicitly and promptly once training finishes, without
waiting for interpreter exit.

### Observability and concurrency

Both mechanisms make their behavior visible on the `ToolResult` they return, but they surface
different things, because they're watching for different failure modes:

| | `IsolatedTool` | `OpenEnvTool` |
|---|---|---|
| **Always-present metadata** | `metadata["isolation"]` (`"process"` / `"in_process_fallback"` / `"unavailable"`), `metadata["tool"]` | `metadata["latency_ms"]` (wall-clock time of the remote call, measured with `time.monotonic()`), `metadata["remote_name"]` |
| **On failure** | `metadata["fallback_reason"]` (`"not_picklable"` / `"timeout"` / `"isolation_error"`) when it fell back | `ToolResult(success=False, output={"error": ...}, error=...)`: a `RuntimeError` from the remote env is caught specifically; any other exception is caught and reported as `f"{type(exc).__name__}: {exc}"` |
| **What it's watching for** | Whether isolation itself succeeded | Whether the remote call itself succeeded, and how long it took |

`OpenEnvTool.execute` logs every call at `debug` level on success and `warning`/`error` on
failure, always including `latency_ms`, useful for spotting a remote tool that's degrading
before it starts timing out outright.

Concurrency has a real, stated constraint worth knowing before you scale up rollout workers:
"All tools built from the same `create_openenv_tools()` call share one persistent WebSocket
session," per the module docstring. Two consequences follow directly from that:

- **Parallel rollouts needing isolation from each other should call `create_openenv_tools()`
  once per worker**, not share one call's tools across workers, otherwise concurrent workers
  serialize on the same underlying session.
- **`OpenEnvHandle` is the one thing responsible for tearing that session down.** It supports
  both explicit `.close()` and use as a context manager (`with OpenEnvHandle(...) as handle:`);
  either releases the WebSocket connection and the background event loop the
  `SyncEnvClient` runs on. Forgetting to close it leaks a live connection and a thread for the
  lifetime of the process; this is exactly why `TrainerConfigBridge._build_openenv_tools`
  registers `handle.close` with `atexit` as a fallback on top of `close_openenv_handles()`.

`IsolatedTool` has no equivalent shared-session concern; it opens a brand-new
`ProcessPoolExecutor(max_workers=1)` for every single `execute()` call and tears it down
immediately after (the `with ProcessPoolExecutor(...) as pool:` block), so there's no
persistent state across calls to leak. The tradeoff is the opposite one: paying full process
spin-up cost on every call, rather than amortizing a connection across many calls the way
OpenEnv's persistent session does.

## Choosing between them

Reach for `IsolatedTool` (`isolate_tools([...])`) when the concern is a **badly-behaved tool
on your own machine**, one that might hang, leak memory, or crash the process, and you want
that contained with zero extra infrastructure; the automatic in-process fallback means it's
always safe to wrap a tool this way even if some of your tools happen to be unpicklable.
Reach for OpenEnv when the concern is a **genuinely untrusted execution environment** that
should never share a process, filesystem, or network namespace with the training process at
all, which needs a server actually running somewhere (`openenv` is a base dependency). The two
compose: an `OpenEnvTool` itself holds a live WebSocket and is not picklable, so wrapping one
in `IsolatedTool` degrades to the safe, logged in-process fallback rather than double-isolating
it; the network boundary OpenEnv already provides is the real isolation in that case.

See [User Guide: Tool Library](../user-guide/tool-library.md) for starting a server, and
the [Local Notebooks](../notebooks/local-notebook.md) index for a real OpenEnv
environment exercised end-to-end.
