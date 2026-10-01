# DECIDE Workflows

DECIDE is AgentTune's second system, separate from the agentic spine covered elsewhere
in this guide. Where the spine is about training an agent's policy, DECIDE is about
running a **decision workflow defined entirely in YAML**: extract some fields, apply
rules, ask a model, maybe fan out to several agents in parallel, route to a human if
you're unsure, and log every step to an audit trail. No training, no GPU, unless a stage
you configure actually calls a model.

This page is the practical how-to: write a template, understand the stage types, run it,
collect training data from it, deploy a retrained model into it. For the full stage-type
API inventory (every field, every "does this need a model" rating), see
[Reference: DECIDE Engine](../reference/decide-engine.md); this page won't duplicate that
table, it'll show you templates and stages actually being used.

## The building blocks

Four real classes do all the work, and they're worth knowing by name before you look at
YAML:

- **`ConfigLoader`** (`agenttune/decide/config.py`): loads a template YAML, resolves its
  `extends:` chain, deep-merges it with the global config, validates required fields, and
  auto-generates the `edges` list your stage's `next`/`on_result`/`rules` fields imply.
  Pure Python, no model needed.
- **`TemplateRegistry`** (`agenttune/decide/registry.py`): scans a directory of template
  YAMLs and indexes them by `id`, so `agenttune decide list` can enumerate them.
- **`GraphRunner`** (`agenttune/decide/graph_runner.py`): compiles a resolved config into
  a real `langgraph` `StateGraph`, one node per stage, and executes it. This is the thing
  you actually run.
- **`PipelineState`** (`agenttune/decide/state.py`): the state object threaded through
  every stage: `stage_outputs`, `step_history`, `verdict`/`verdict_label`/`confidence`/
  `reason`, `is_complete`, `error`, `elapsed_seconds`, `next_stage`.

```python
from agenttune.decide import GraphRunner

runner = GraphRunner.from_template("bfsi/kyc_triage", config_path="./config.yaml")
state = runner.run_sync("Applicant: Jane Doe, DOB 1985-03-12, income $85,000...")
print(state.verdict, state.verdict_label, state.confidence)
```

`GraphRunner.from_template(template_id, config_path)` is the real factory method;
`template_id` can be a registry-style path like `"bfsi/kyc_triage"` (resolved under
`agenttune/decide/templates/`) or a direct filesystem path to your own YAML.
`run_sync(input_text)` blocks until completion (it detects whether an event loop is
already running and dispatches to a thread pool if so); `await runner.run(input_text)` is
the native async form if you're already inside one.

`ConfigLoader.load` also guards against path traversal: a `template_id` containing `..`
raises `ValueError` immediately rather than resolving somewhere outside the templates
directory.

## Getting started: `agenttune decide init`

Before writing your own template, you need a `config.yaml`, the "environment config"
every template's `extends:` inherits from (model choice, episode/reward/collect/eval
settings, destinations, compliance flags):

```bash
agenttune decide init --output config.yaml
```

This literally copies `agenttune/decide/config.example.yaml` to the path you gave it (it
asks before overwriting an existing file). The generated file is thoroughly commented,
worth reading once in full, but the sections that matter most:

```yaml
default_model: "claude-haiku-4-5"      # what llm_call/llm_judge/router use if a stage
judge_model: "claude-opus-4-1"         #   doesn't specify its own `model:`

run_mode: inference   # inference | collect | eval | train — overridden by --mode

episode:
  n_episodes: 1        # only used by collect/eval/train modes
  batch_size: 1         # concurrent pipeline runs per batch
  shuffle_inputs: false
  seed: 42

reward:
  default_fn: verdict_binary   # verdict_binary | judge_score | rule_pass_rate | iteration_penalty
  normalize: true
  scale: [0.0, 1.0]

collect:
  algorithm: dpo              # dpo | bco | grpo | ppo | rloo — see §7
  stage_id: null               # null = the final output stage; or target one stage's I/O
  min_samples: 50
  output_path: ./collected_data.jsonl
  auto_train: false

eval:
  test_set: null                        # JSONL: {"input": ..., "expected_verdict": ...}
  metrics: [accuracy, latency_p50, latency_p95, cost_per_decision, judge_score_mean]
  thresholds: {accuracy: 0.80, latency_p95: 5000, cost_per_decision: 0.10}

training:
  trainer_config: null     # path to a trainer_config.yaml for TrainerConfigBridge
  auto_deploy: false         # if true, ModelDeploymentBridge repoints default_model after training
  deploy_stage_id: null      # or deploy to one stage only (A/B style)

audit:
  destination: file
  path: ./audit.jsonl

destinations:
  file: {enabled: true, path: ./decisions.jsonl, format: jsonl}
  postgres: {enabled: false, connection_string: "...", table: decisions}
  webhook: {enabled: false, url: "...", method: POST}
```

Every template's `extends: ../../../../../config.yaml` (relative to the template's own
location under `agenttune/decide/templates/`) pulls this file in as the base layer, then
overrides only what it needs. `ConfigLoader.deep_merge` does dict-recursive merging (list
fields replace wholesale rather than concatenating). If you're running templates from
outside the package, point `config_path=` at wherever you put your own `config.yaml`
instead; `extends:` in a template you write yourself doesn't have to be that exact
relative path.

If you'd rather see an already-filled-in, production-flavored config, the repo's own root
`config.yaml` is real and worth comparing: it deliberately pins `default_model` to an
open-source model (`Qwen/Qwen2.5-1.5B-Instruct`) and disables `judge_as_reward` and
`reward.default_fn: judge_score`, favoring the zero-API-call `verdict_binary` reward, a
reasonable default if you don't want every pipeline run making an extra model call just to
score itself.

## Templates: 24 real, loadable YAML workflows

`bfsi/` (12: KYC, fraud, loan approval, sanctions, card decline, claims triage...),
`generic/` (10: classification, sentiment, summarization, entity extraction, multi-judge
scoring...), `custom/` (2), `example/` (1). All discoverable and runnable as-is:

```bash
agenttune decide list                          # every template, name + version + description
agenttune decide list --category bfsi          # filter by top-level directory
agenttune decide validate --template bfsi/kyc_triage
agenttune decide show --template bfsi/kyc_triage    # dump the fully-resolved (post-extends) YAML
```

`validate` runs the same `ConfigLoader.load` your `GraphRunner.from_template` call would,
so a passing `validate` means the template will actually load. Structural errors (a
stage missing `id`/`type`, an unregistered `type`, a missing `id`/`name`/`version` at the
template level) are caught here, without needing a model.

Three real templates, chosen for how differently they're built:

### The simplest shape: `generic/entity_extract.yaml`

One `llm_call`, one `output`. This is close to the smallest template that does anything
useful:

```yaml
extends: ../../../../../config.yaml

name: "ENTITY EXTRACT"
id: "generic/entity_extract"
version: "1.0.0"
tags: ["generic"]

max_total_steps: 30

stages:
  - id: process
    type: llm_call
    description: "Extract entities from text"
    prompt: |
      Extract all named entities (persons, organizations, locations) from the given text.
      Respond ONLY with valid JSON (no markdown, no code blocks): {"entities": [{"type": "PERSON"|"ORG"|"LOC", "value": "..."}], "verdict": "PASS" | "FAIL"}

      Text: {input_text}
    output_schema:
      type: object
      properties:
        entities: {type: array, items: {type: object, properties: {type: {type: string}, value: {type: string}}, required: ["type", "value"]}}
        verdict: {type: string, enum: ["PASS", "FAIL"]}
      required: [entities, verdict]
    next: output_stage

  - id: output_stage
    type: output
    verdict_field: "process.output.verdict"
    destinations: [file]
```

`{input_text}` is the interpolation syntax stages use to reach the pipeline's raw input;
`{stage_id.output.field}` (used heavily in the next two examples) reaches another stage's
parsed output. `verdict_field: "process.output.verdict"` tells the `output` stage to pull
`state.verdict` from `process`'s parsed JSON rather than a literal value.

### A production-config template: `generic/multi_judge_score.yaml`

Same linear shape as above but chained five stages deep (preprocess → 3 independent
judges → aggregate → output) and, more importantly, a real `episode`/`reward`/`collect`/
`eval` block filled in, not left at global defaults:

```yaml
episode:
  n_episodes: 50
  batch_size: 2
  seed: 42

reward:
  stages:
    aggregate:    {fn: verdict_binary, weight: 1.0}
    output_high:  {fn: verdict_binary, weight: 0.3}
    output_low:   {fn: verdict_binary, weight: 0.3}
  final_fn: weighted_mean

collect:
  algorithm: bco              # binary labels: PASS / FAIL verdict
  stage_id: aggregate
  min_samples: 100
  output_path: ./collected/multi_judge_bco.jsonl

eval:
  metrics: [judge_score_mean, accuracy, latency_p50]
  thresholds: {judge_score_mean: 0.65, accuracy: 0.80}
```

The three judge stages (`judge_1`/`judge_2`/`judge_3`) each score a different axis
(accuracy, clarity, completeness) from the *same* `preprocess` output, deliberately kept
independent ("prevent anchoring bias" per the template's own description) before
`aggregate` combines them with a weighted formula in its prompt. This is the template to
copy if you're building an LLM-judge panel rather than a single classifier.

### A real BFSI production pipeline: `bfsi/kyc_triage.yaml`

This is the deep end: `rules` (deterministic gate) → `parallel` (fan-out to three
specialist agents) → `llm_judge` (synthesis) → conditional routing to either an `output`
stage or a `human_review` stage. Walked through stage-by-stage in the next section, since
it's also the best real example of `rules`/`router`-style branching in the repo.

## Stage types in depth: `rules` and `router`

The [reference table](../reference/decide-engine.md#stage-types-8-all-registered-and-real)
lists all 8 stage types; this section shows the two conditional-logic ones actually
working, end to end, because a straight `llm_call → output` chain (the examples above) is
the easy 80%; real pipelines branch.

### `rules`: deterministic, no model, no API cost

`RulesStage` evaluates a list of `condition` strings against the pipeline's accumulated
`stage_outputs`, using `SafeEvaluator`, a hand-written AST walker, not `eval()`. It
allows exactly `== != < <= > >= and or not`, dotted attribute lookups (`s0.output.score`),
and the JSON literals `true`/`false`/`null`; it explicitly rejects dunder access and
disallows any node type it doesn't recognize (no function calls, no imports, nothing that
executes arbitrary code). It's usable standalone, outside any pipeline:

```python
from agenttune.decide.stages.rules import SafeEvaluator

ok = SafeEvaluator().eval("income > 5000 and dob != null",
                          {"income": 62000, "dob": "1990-01-01"})
# True
```

Inside a pipeline, conditions read from `flatten_stage_outputs(state.stage_outputs)`,
which exposes each field three ways: `stage_id.output.field`, `stage_id.field`, and a bare
`field` (the last stage that produced that key wins on collisions). That's why
`bfsi/kyc_triage.yaml`'s rules can write bare `dob`/`income` instead of always qualifying
with a stage id:

```yaml
- id: quality_check
  type: rules
  description: "Validate extracted data meets minimum KYC requirements"
  rules:
    - condition: "dob != null and dob != ''"
      on_fail:
        goto: extract
        inject: "Date of birth is required. Please re-extract it from the document."
    - condition: "address != null and address != ''"
      on_fail:
        goto: extract
        inject: "Residential address is required. Please re-extract it."
    - condition: "income != null and income > 0"
      on_fail:
        goto: extract
        inject: "Income field is missing or invalid (must be > 0). Please re-extract."
  next: parallel_analysis
```

Rules are checked in order; the **first failing** condition short-circuits with
`{"goto": ..., "inject": ...}` (both `on_fail` and `on_failure` keys work; the code
checks `rule.get("on_failure") or rule.get("on_fail")`), routing back to `extract` instead
of falling through to `next`. The `inject` string becomes available to the target stage's
prompt as `{feedback}`, which is what makes `extract`'s prompt template
(`...Document:\n{input_text}\n\n{feedback}`) show the specific validation failure on
retry rather than just re-asking blind. If every rule passes, execution falls through to
`next` (`parallel_analysis` here) with `{"output": results, "goto": None}`.

### `router`: LLM-based or rules-based dispatch, same stage type

`RouterStage` has one code path with a fork in it: if the stage config has a `model` (its
own, or falling back to `global_config["default_model"]`), it calls that model, validates
the JSON reply against `output_schema`, then evaluates `on_result` conditions against the
*model's own output*. If there's no `model` at all, it skips the LLM call entirely and
evaluates `on_result` against `state.stage_outputs` (flattened) instead: pure rules-based
routing through the exact same stage type. `generic/document_classification.yaml` uses
both modes back to back:

```yaml
stages:
  - id: extract_features       # type: llm_call — pulls topic/sentiment/style/keywords
    ...
    next: rule_based_classification

  - id: rule_based_classification
    type: rules
    rules:
      - condition: "document_style == 'legal' and any(keyword in ['contract', 'agreement', 'clause'] for keyword in key_keywords)"
        on_fail: {goto: llm_judge, inject: "..."}
      - condition: "document_style == 'technical' and sentiment != 'negative'"
        on_fail: {goto: llm_judge, inject: "..."}
    next: confidence_check

  - id: confidence_check         # ← type: router, LLM-based
    type: router
    description: "Check if rule-based classification has high enough confidence to output directly"
    prompt: |
      Based on extracted features:
      - Topic: {extract_features.output.topic}
      - Style: {extract_features.output.document_style}

      Is there clear, high-confidence category assignment possible?
    output_schema:
      type: object
      properties:
        is_confident: {type: boolean}
        apparent_category: {type: string}
    model: claude-haiku-4-5
    next: output_result
    on_result:
      - condition: "not is_confident"
        goto: llm_judge          # ambiguous → escalate to a stronger judge

  - id: llm_judge                # the escalation path — full classification
    type: llm_judge
    ...
    next: output_result
    on_fail: output_result

  - id: output_result
    type: output
    verdict_field: "llm_judge.output.category"
    confidence_field: "llm_judge.output.confidence_score"
```

Here `confidence_check` is genuinely LLM-based (it has a `model:`): it asks a small,
cheap model whether the rule-based signal is already confident enough, and only escalates
to the expensive `llm_judge` stage (a bigger model, a longer prompt, full category
enumeration) when the answer is no. A router with the `model:` line deleted and an
`on_result` list referencing `stage_outputs` directly (see `RouterStage`'s rules-based
branch) gets you the second mode: free, deterministic branching on upstream fields,
identical stage type.

Either way, `on_result`'s first matching `condition` wins; if none match, execution falls
back to `default` (if set) or `next`.

### The rest, briefly: `parallel`, `human_review`, `tool_call`

`bfsi/kyc_triage.yaml`'s `parallel_analysis` stage fans out to three independent
`llm_call` branches and rejoins before continuing, with all branches run concurrently via
`asyncio.gather`. Each branch below is **flat**: the branch dict itself is the one stage
to run (`ParallelStage` also supports a `{id, stages: [...]}` nested-multi-stage-per-branch
shape, if a branch needs to run more than one stage internally, but no bundled template
currently needs that):

```yaml
- id: parallel_analysis
  type: parallel
  branches:
    - id: income_agent
      type: llm_call
      prompt: "Analyze the income verifiability for this KYC applicant. ..."
      output_schema: {type: object, properties: {score: {type: integer}, risk_level: {type: string}}}
    - id: credit_agent
      type: llm_call
      prompt: "Evaluate the credit and repayment risk for this applicant. ..."
    - id: compliance_agent
      type: llm_call
      prompt: "Perform a compliance and AML risk assessment for this applicant. ..."
  next: decision_judge
```

Downstream stages reference each branch's output directly by its branch `id`
(`{income_agent.output.score}`), exactly as if it were a top-level stage. `parallel`
merges branch outputs into `stage_outputs` keyed by branch id, so nothing downstream needs
to know the fan-out happened.

`human_review` is real but deliberately prototype-grade: it writes a JSON file to
`./pending_reviews/{review_id}.json` and polls `./review_decisions/{review_id}.txt` once a
second until a decision appears or `timeout_seconds` elapses. No auth, no UI, just a
filesystem handshake you (or a script, or a small internal tool) fulfill by dropping a
file:

```yaml
- id: output_human_review
  type: human_review
  prompt_for_human: |
    === KYC Manual Review Required ===
    Applicant: {extract.output.full_name}
    Judge Score: {decision_judge.output.score}/10
    Please Approve or Deny.
  timeout_seconds: 86400          # 24h window
  on_approved: output_approve
  on_denied: output_deny
```

`tool_call` delegates to the exact same tool registry the agentic spine uses
(`agenttune.agentic.tools`): same `ToolRegistry`/`ToolExecutor`, same
[known caveat](../community/known-issues.md#looks-like-it-works-doesnt-or-gives-a-quietly-wrong-answer)
about `auto_register_builtins()` importing every builtin (including the
`langchain_community`-dependent ones) on first use. From the stage's own docstring, a real
schema:

```yaml
- id: lookup_customer
  type: tool_call
  tool: sql_query
  args:
    query: "SELECT * FROM customers WHERE id = '{extract.output.customer_id}'"
  next: risk_score
```

Arguments are interpolated the same way prompts are (`{stage_id.output.field}` etc.)
before the tool executes, with a per-stage `timeout_sec` (default 30s).

## Running a pipeline

Python, as shown throughout this page. Give `extract` a document that actually has the
fields `quality_check` requires (name, DOB, address, income); a vague placeholder like
`"some input text"` just makes the extract/quality-check retry loop run to
`max_total_steps` before giving up with no verdict:

```python
from agenttune.decide import GraphRunner
runner = GraphRunner.from_template("bfsi/kyc_triage", config_path="config.yaml")
state = runner.run_sync(
    "Applicant: Jane Doe, DOB 1985-03-12, address 45 Oak Ave Denver, income $85,000 "
    "annual, employed at Globex Corp, provided passport and utility bill, PEP status: no."
)
print(state.verdict, state.confidence, state.reason)
```

Or the CLI, which additionally writes the result to a file:

```bash
agenttune decide run --template bfsi/kyc_triage \
  --input "Applicant: Jane Doe, DOB 1985-03-12, address 45 Oak Ave Denver, income \$85,000 annual, employed at Globex Corp, provided passport and utility bill, PEP status: no." \
  --output decision.json --config config.yaml
```

`--input` also accepts `@path/to/file`: a plain-text file (one line = one input) or a
JSONL file where each line is `{"input": "..."}` (falls back to the raw line if a line
isn't valid JSON or has no `input` key).

The CLI's `--mode` flag (or `run_mode:` in config, if `--mode` is omitted) picks one of
four real, distinct execution paths:

| Mode | What it does |
|---|---|
| `inference` (default) | One pipeline run. Writes `decision.json` + appends to `audit.jsonl`. |
| `collect` | Runs `episode.n_episodes` episodes, writes training records via `CollectRunner` to `collect.output_path` (see next section). |
| `eval` | Runs `EvalRunner` against `eval.test_set`, computes accuracy/latency/cost metrics, exits non-zero if any `eval.thresholds` are violated. |
| `train` | Runs `collect`, then (if `training.trainer_config` is set) builds and runs a trainer via `TrainerConfigBridge`, then (if `training.auto_deploy`) repoints `default_model` via `ModelDeploymentBridge`. |

`train` mode without `training.trainer_config` set prints a warning and skips training
entirely. It will still collect and write the dataset, so it's safe to leave unset while
you're still accumulating data.

## Collecting training data: `CollectRunner`

The recommended way to turn live pipeline runs into DPO/BCO/GRPO-shaped training records
is `CollectRunner`. It reads directly from each episode's in-memory `PipelineState`,
not from the audit log:

```python
from agenttune.decide import GraphRunner, CollectRunner

runner = GraphRunner.from_template("bfsi/kyc_triage", config_path="config.yaml")
runner.config["episode"]["n_episodes"] = 2   # see note below — the template defaults to 100
n_written = await CollectRunner(runner).run(inputs=[
    "Applicant: Jane Doe, DOB 1985-03-12, address 45 Oak Ave Denver, income $85,000 "
    "annual, employed at Globex Corp, provided passport and utility bill, PEP status: no.",
    "Applicant: Bob Lee, DOB 1970-11-02, address 9 Birch St Austin, income $42,000 "
    "annual, self_employed, provided passport, PEP status: no.",
])
```

(`CollectRunner.run` is async; call it with `await` inside an event loop, or
`asyncio.run(...)` from sync code; that's also exactly what `agenttune decide run --mode
collect` does under the hood.)

!!! warning "`CollectRunner` runs `episode.n_episodes` episodes, not `len(inputs)`"
    `inputs` is a *pool* that gets cycled, not a one-shot list: `run_episodes` always
    executes exactly `episode.n_episodes` pipeline runs (batched `episode.batch_size` at a
    time), wrapping around the input list if it's shorter. `bfsi/kyc_triage.yaml`
    deliberately overrides the global `episode.n_episodes: 1` up to **100** (with
    `batch_size: 4`), since it's meant to collect a real training set; passing it 2
    example inputs without also overriding `n_episodes` silently launches 100 real pipeline
    episodes (many minutes on a local model, not a hang). Override `n_episodes` down (as
    above) for a quick illustrative run, and back up for real collection.

The record shape depends on `collect.algorithm` in config (or the template's own
override):

- **`bco`** → `{"input": ..., "label": 1 if verdict in PASS_VERDICTS else 0, "verdict": ..., "pipeline_id": ...}`
- **`dpo`** → `{"input": ..., "chosen_output": ..., "rejected_output": ..., "verdict": ..., "pipeline_id": ...}`
  (splits on pass/fail today; human-feedback-pair extraction is the eventual richer
  source but isn't wired into `CollectRunner` yet)
- **`grpo`/`ppo`/`rloo`** → `{"input": ..., "output": ..., "reward": ..., "verdict": ..., "pipeline_id": ...}`,
  where `reward` comes from `reward.stages.<stage_id>.fn`. `judge_score` normalizes a
  0–10 judge score to `[0, 1]`, anything else falls back to a binary pass/fail.

`PASS_VERDICTS` (from `agenttune.decide.state`) is `{"PASS", "APPROVE", "COMPLETE"}`;
anything else counts as a fail for labelling purposes.

!!! note "Don't re-parse `audit.jsonl` for this"
    `AuditReader.extract_dpo_pairs`/`extract_bco_labels` (`decide/audit.py`) look for
    fields (`human_feedback`, a top-level `verdict` key) that `AuditWriter` never actually
    emits; they always return an empty list against a real log. `CollectRunner` is the
    real, working path; see
    [Known Issues](../community/known-issues.md#looks-like-it-works-doesnt-or-gives-a-quietly-wrong-answer)
    for the exact field mismatch.

## Stage-wise execution: `StageWiseRunner`, and when not to use it

`StageWiseRunner` is a second, simpler execution path: it runs every stage top-to-bottom
and returns a per-stage result list, useful for generating stage-specific training
examples one stage at a time:

```python
from agenttune.decide.stage_wise_runner import StageWiseRunner   # not re-exported from
                                                                  # agenttune.decide itself

runner = StageWiseRunner.from_template("bfsi/kyc_triage", config_path="config.yaml")
results = await runner.run_stage_wise("... document text ...")
# [{"stage_id": "extract", "output": {...}, "step": 1}, {"stage_id": "quality_check", ...}, ...]
```

!!! warning "It ignores all conditional routing"
    `StageWiseRunner` runs every stage in the template's `stages:` list in file order,
    unconditionally: it never evaluates `on_result`, `rules`, `router` decisions, or
    `goto`. On `bfsi/kyc_triage.yaml` it would run `output_approve`, `output_human_review`,
    *and* `output_deny` regardless of what `decision_judge` actually decided. It's fine
    for a template that's genuinely linear (`generic/entity_extract.yaml`); it is **not**
    a substitute for `GraphRunner` on anything with branching. See
    [Known Issues](../community/known-issues.md#looks-like-it-works-doesnt-or-gives-a-quietly-wrong-answer).

## After a decision runs: audit trail and destinations

Every stage execution and every completed run gets written to the audit log
(`AuditWriter`, path from `audit.path`, `./audit.jsonl` by default), one line per stage
plus a final completion record, keyed by `pipeline_id`/`stage_id` (also carried as
`trajectory_id`/`stage_name` aliases, the schema the self-healing failure detector reads;
see [User Guide: Self-Healing](self-healing.md) for how the closed loop consumes this
log).

Once a run finishes, `DestinationRouter.route(state, config)` dispatches the final
decision to every enabled entry under `destinations:`: `FileWriter` (stdlib, always
available), `WebhookSender` (needs `requests` and a reachable URL), `PostgresWriter`
(needs `psycopg2` and a pre-existing table matching your `table:` config). This happens
automatically after `GraphRunner.run()`/`run_sync()`; you don't call it yourself.

For what happens *next*, retraining off a stream of production decisions, gating a new
checkpoint before it goes live, deploying it back into a stage, see
[User Guide: Self-Healing](self-healing.md). The short version: the DECIDE-native
detection path (watching `audit.jsonl` for failures automatically) works against a real
audit log, and every downstream piece (classify, generate corrected examples, retrain,
gate, deploy) is real too. You can also feed the classifier/generator manually
constructed `Failure` objects when your failure signal doesn't come from a DECIDE audit
log at all, or skip detection entirely and use `CollectRunner` (above) if what you
actually want is training data rather than automated failure-driven retraining.

`ModelDeploymentBridge.deploy_trained_model(...)` (used by `train` mode's `auto_deploy`)
is the concrete mechanism for pointing a template at a freshly trained checkpoint: it
rewrites `config.yaml`'s `default_model`, or a specific stage's `model:` field if you pass
`deploy_stage_id`, with an automatic backup and a `rollback_deployment` to undo. It needs
no training infra to use directly; you can point it at any local checkpoint path right
now.

## Common pitfalls

- **Editing `agenttune/decide/templates/*.yaml` in place and expecting `extends:` to still
  resolve.** The relative `extends: ../../../../../config.yaml` path is computed from the
  template file's own location; copy a template out of the package tree and you'll need
  to fix (or replace) that path, or pass an absolute one.
- **Assuming `rules`/`router` conditions can call functions.** `SafeEvaluator` explicitly
  disallows function calls (that `any(... for ... in ...)` generator expression in
  `document_classification.yaml`'s rules stage is Python syntax in the YAML string, but
  note it's inside an `llm_call`'s prompt context describing intent to a human reader of
  the template. The actual evaluator only supports comparisons/boolean ops/dotted
  lookups; don't assume arbitrary Python works inside a real `condition:` string).
- **Expecting `human_review` to have any real UI or auth.** It's a filesystem poll loop,
  fine for a prototype or an internal script, not for anything user-facing without
  building the UI/auth layer yourself on top.
- **Calling `ToolRegistry.get(...)` (via a `tool_call` stage) and hitting an unrelated
  `ImportError`.** See the `tool_call` section above; it's a known, documented gap, not a
  configuration mistake on your part.
- **Path-traversal in `template_id`.** `ConfigLoader.load` explicitly rejects any
  `template_id` containing `..`; pass an absolute path if your template genuinely lives
  outside the package tree.

## See it run for real

The corresponding [Local Notebook](../notebooks/local-notebook.md) builds a real 2-stage
template (classify sentiment → decide an action) and runs it through real chained model calls,
specifically to prove stage 2 genuinely reads stage 1's real parsed output rather than
each stage being independently canned.

## Related reading

- [Reference: DECIDE Engine](../reference/decide-engine.md): the full 8-stage-type API
  inventory, dependency ratings, and the destinations/config-bridge reference.
- [Concepts: DECIDE & the Closed Loop](../concepts/decide-and-closed-loop.md): the
  narrative introduction to how DECIDE and the self-healing loop connect.
- [User Guide: Self-Healing](self-healing.md): what happens after a pipeline is deployed:
  detecting failures, retraining, gating, redeploying.
- [Known Issues](../community/known-issues.md): every gap referenced on this page, with
  the exact code-level cause.
