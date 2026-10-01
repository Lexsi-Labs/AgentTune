# Python API: DECIDE Engine

`agenttune.decide`: define a decision pipeline as YAML instead of code. See
[Concepts: DECIDE & the closed loop](../concepts/decide-and-closed-loop.md) for the
narrative introduction; this page is the standalone-usability inventory. Self-healing
specifically is covered separately in
[Python API: Self-Healing Closed Loop](closed-loop.md).

## The core entry point

```python
from agenttune.decide import GraphRunner
runner = GraphRunner.from_template("generic/text_classify", config_path="config.yaml")
state = runner.run_sync("some input")
```

`template_id` is a dotted path under `src/agenttune/decide/templates/` **without** the
`.yaml` extension (e.g. `"generic/text_classify"` resolves to
`decide/templates/generic/text_classify.yaml`); any string containing `/` and not
starting with one is resolved this way, so a literal file path like `"path/to/workflow.yaml"`
would resolve to the wrong location (`templates/path/to/workflow.yaml.yaml`). To load a
YAML file from an arbitrary location instead, pass an absolute path.

Compiles the YAML into a real `langgraph` `StateGraph`, handles observation-schema
validation, engine caching, step-limit enforcement, audit logging, and routes the final
decision via `DestinationRouter`. **Rating B**: needs whatever backend/API key the
template's stages specify; the orchestration itself has no hard dependency.

## Stage types (8, all registered and real)

| Stage | Needs a model? | Notes |
|---|---|---|
| `llm_call` | Yes (B) | Simple prompt→JSON, or agentic mode with tools |
| `llm_judge` | Yes (B) | Wraps `LLMJudge`, falls back to a raw call if that import fails |
| `rules` | **No (A)** | `SafeEvaluator`: AST-based safe expression evaluator (`==`, `<`, `and`, `or`, `not`, dotted lookups, blocks dunder access, no calls/imports). Usable standalone: `SafeEvaluator().eval("income > 5000", context)` |
| `router` | Only for LLM-based routing (B); rules-based routing needs nothing (A) | |
| `parallel` | Depends on sub-stages | Runs branches concurrently via `asyncio.gather` |
| `human_review` | **No (A)** | Real but prototype-grade: filesystem-poll pause/resume (`./pending_reviews/`, `./review_decisions/`); no auth, no UI |
| `tool_call` | Depends on the tool (B) | Delegates to the agentic tool registry |
| `output` | **No (A)** | Sets the final verdict/confidence/reason from literal values or field references |

`stages/base.py`'s `_interpolate` (prompt templating), `_validate_json`, and the module-level
`flatten_stage_outputs` are reusable pure-Python utilities on their own (**A**) if imported
directly.

## Templates: 24 real, loadable YAML workflows

`bfsi/` (12: KYC, fraud, loan approval, sanctions...), `generic/` (10: classification,
sentiment, summarization, entity extraction...), `custom/` (2), `example/` (1).

```bash
agenttune decide list
agenttune decide validate --template bfsi/kyc_triage
agenttune decide show --template bfsi/kyc_triage
```

`TemplateRegistry` (**A**) does discovery/indexing; `ConfigLoader` (**A**, pure YAML/dict
merging: `extends`, deep-merge, auto edge generation) does the load-time resolution every
runner depends on.

## Running episodes and collecting training data

- **`CollectRunner(runner).run(inputs)`**: runs N episodes, extracts DPO/BCO/GRPO-PPO-RLOO-
  shaped records **directly from in-memory `PipelineState`**. This is the recommended path
  for producing training data, unlike re-parsing the audit log afterward (see
  [Known Issues](../community/known-issues.md)), this one matches what actually gets
  produced. **Rating B** (needs the template's model to run episodes).
- **`EvalRunner(runner).run(test_set_path)`**: runs a labelled JSONL test set, computes
  accuracy/latency p50-p95/cost-per-decision/judge-score-mean, checks configured
  thresholds. **Rating B.**
- **`StageWiseRunner`**: an alternate, simpler execution path that runs every stage
  top-to-bottom, ignoring all conditional routing. Not a substitute for `GraphRunner` on any
  template with branching; see [Known Issues](../community/known-issues.md).

All 4 `agenttune decide run --mode` values (`inference`/`collect`/`eval`/`train`) are
genuinely implemented; see [CLI](cli.md).

## Config and deployment helpers: no model needed

- **`ModelDeploymentBridge` / `deploy_trained_model`** (`decide/model_deployment.py`):
  rewrites `config.yaml`'s `default_model`/`backend`/per-stage model fields, with automatic
  backup and `rollback_deployment` to undo. **Rating A**: genuinely usable right now to
  point a template at a local checkpoint, no training infra needed.
- **`AuditWriter` / `AuditReader`** (`decide/audit.py`): JSONL audit trail I/O. See
  [Known Issues](../community/known-issues.md) for what `AuditReader.extract_dpo_pairs`/
  `extract_bco_labels` can't do against a real log; use `CollectRunner` instead.
- **`TrainerConfigBridge`** (`decide/trainer_config_bridge.py`): the full YAML-driven
  builder for the agentic training stack. `get_peft_config()`/`build_reward_funcs()` are
  pure Python (**A**); `build_rollout_engine()`/`build_trainer()` need GPU (**B**).

## Destinations: where a decision gets written

`DestinationRouter.route(state, config)` dispatches automatically after every
`GraphRunner.run()`: `FileWriter` (**A**, stdlib), `WebhookSender` (**B**, needs `requests`
+ a reachable endpoint), `PostgresWriter` (**B**, needs `psycopg2` + a pre-existing table).
