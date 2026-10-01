# Configuration

DECIDE pipelines read one global YAML config, `config.yaml`, at the
repo root, plus a per-template YAML that `extends: ../../config.yaml`. This page is a guide
to the sections in that file; the file itself (embedded below) is the source of truth.

!!! tip "This isn't for the agentic spine"
    `config.yaml` configures **DECIDE** (`agenttune.decide.GraphRunner`), the YAML decision
    pipeline engine. The agentic spine (`Project`, `EventLog`, strategies, harnesses) is
    pure Python and takes no global config file; see
    [Getting Started](../getting-started/installation.md) and
    [Concepts: DECIDE & the closed loop](../concepts/decide-and-closed-loop.md) for how the
    two relate.

## Sections at a glance

| # | Section | Purpose |
|---|---|---|
| 1 | `api_keys` | Deprecated placeholders; AgentTune Decide runs open-source models only (Qwen, LLaMA, …); no cloud API-key provider is used. |
| — | `default_model` / `judge_model` | The model used for `llm_call` stages and for LLM-judge scoring, respectively. |
| — | `max_total_steps` / `timeout_seconds` | Global guards: stop after N steps across all stages; per-execution wall-clock timeout. |
| 4 | `run_mode` | Which loop to run: `inference` (default) \| `collect` \| `eval` \| `train`. |
| 5 | `episode` | Episode/rollout loop config for `collect`/`eval`/`train` modes: `n_episodes`, `batch_size`, `shuffle_inputs`, `seed`. |
| 6 | `observation_schema` | What the pipeline accepts as input: `string` \| `json` \| `file`, with an optional JSON schema. |
| 7 | `reward` | How reward is computed per stage/episode: builtin `default_fn`s (`verdict_binary`, `judge_score`, `rule_pass_rate`, `iteration_penalty`) or a `custom_fn` module path. |
| 8 | `collect` | Training-data collection: `algorithm` (`dpo`\|`bco`\|`grpo`\|`ppo`\|`rloo`), `min_samples`, `output_path`, `auto_train`. |
| 9 | `eval` | Systematic evaluation against a labelled `test_set` JSONL: metrics + pass/fail `thresholds`. |
| 10 | `training` | Links `run_mode: train` to `trainer_config.yaml` via `TrainerConfigBridge`; `auto_deploy` / `deploy_stage_id`. |
| 11 | `audit` | Where the append-only audit log is written (`audit.jsonl` by default); the closed loop's only input. |
| 12 | `destinations` | Where decisions are written: `file`, `postgres`, or `webhook`. |
| 13 | `compliance` | `audit_required`, `human_review_required`, `log_prompts`, `log_responses`. |

## `run_mode`, explained by analogy

The header comment in `config.yaml` frames DECIDE as "the environment config for your LLM
decision pipelines", an OpenAI Gym analogue that replaces a Python `env` with a YAML
pipeline:

| Gym problem | DECIDE's answer |
|---|---|
| Python-only environments | YAML-driven pipeline; no code required |
| Python-only reward functions | `reward.fn`: `verdict_binary` \| `judge_score` \| `rule_pass_rate` \| `iteration_penalty` \| `custom` |
| No LLM in `step()` | `llm_call` / `llm_judge` stage types |
| No multi-step reasoning | `tools: [web_search, sql_query]` per stage |
| Fixed observation/action space | `output_schema` per stage + `observation_schema` |
| No episode batching | `episode.n_episodes` / `episode.batch_size` |
| No training-data collection | `collect.algorithm: dpo \| bco \| grpo \| ppo \| rloo` |
| Hard-coded done conditions | `on_result` conditions per stage |
| No human-in-the-loop | `human_review` stage type |
| No observability | `audit.jsonl` per execution |

## Per-template overrides

Templates under `src/agenttune/decide/templates/` (generic, custom, and BFSI use cases)
`extends: ../../config.yaml` and override only what differs; most commonly `run_mode`,
`reward`, and `observation_schema`. Validate a template's resolved config with
[`agenttune decide validate`](cli.md), or dump it with
[`agenttune decide show`](cli.md).

## The full file

```yaml
--8<-- "config.yaml"
```
