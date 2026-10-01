# Known Issues

Found by reading the code directly, module by module, rather than trusting existing docs.
Most of what's below is narrow, one function, one edge case. A few are significant gaps
between what a reader might assume is wired up and what actually runs. Organized by how
much each should worry you.

## Looks like it works, doesn't, or gives a quietly wrong answer

- **Fetching *any* tool can crash on an unrelated missing dependency.**
  `ToolRegistry.get("read_file")` triggers `auto_register_builtins()`, which unconditionally
  imports every builtin tool module at once, including the `langchain_community`-dependent
  ones (Slack/GitHub/Playwright/SQL/web-search). Without `langchain_community` installed,
  even fetching the pure-stdlib `read_file` tool raises `ImportError`.
- **`DistilledJudge` silently returns an all-zero reward** if `transformers` isn't installed
  or its checkpoint fails to load; logged as a warning, not raised as an error.
- **`RougeMetric`/`BleuMetric` silently return `0.0`** if `rouge_score`/`nltk` aren't
  installed; the eval appears to succeed, the score is just meaningless.
- **`OutputStage._route_to_destination` is a dead no-op.** Harmless; real destination
  routing happens separately via `DestinationRouter.route()` after the whole pipeline
  finishes; but this specific method does nothing if you go looking at it.
- **`StageWiseRunner` silently ignores all conditional routing** (`on_result`/`rules`/
  `router`/`goto`/`next`); it just runs every stage top-to-bottom. Not a substitute for
  `GraphRunner` on any template with branching.
- **`decide/audit.py`'s `AuditReader.extract_dpo_pairs`** (and `DecideToTrainerBridge.extract_bco_labels`
  in `decide/training_bridge.py`, which wraps `AuditReader` but implements this one itself)
  look for fields (`human_feedback`, a top-level `verdict`) that `AuditWriter` never emits;
  always return an empty list against a real log. **`CollectRunner` is the working
  alternative**: it builds DPO/BCO training records directly from in-memory
  `PipelineState` during a real run, not by re-parsing the audit log afterward.
- **~~`agenttune train` cannot currently succeed~~ Partially fixed.** Its `--dataset` flag
  used to land in `train_dataset` (the pre-loaded-dataset path) instead of the alias
  `dataset` that routes through `DataManager`'s HF-dataset-id loading path; fixed, a
  dataset id passed via the CLI now actually loads. A new `--reward-funcs` flag (names
  from `REWARD_REGISTRY`, comma-separated) makes `grpo`/`rloo` runnable via the CLI too.
  Still no `--tools` flag; agentic tool-calling training isn't reachable from the CLI,
  only the Python API. See [User Guide: RL Training](../user-guide/rl-training.md).
- **A registry-fetched `SQLDatabaseTool()` has no database configured.**
  `ToolRegistry.auto_register_builtins()` instantiates it with no `db_uri`, but
  `.execute()` handles this cleanly, returning `ToolResult(success=False, error="No db_uri
  set. Pass it at construction: SQLDatabaseTool('sqlite:///mydb.db')")` rather than
  crashing. Still, you'll want to construct `SQLDatabaseTool(db_uri=...)` yourself rather
  than fetching the unconfigured instance from the registry.
