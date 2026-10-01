# CLI

Installing AgentTune (`pip install -e .`) registers the `agenttune` console script:

```bash
agenttune --help
```

It resolves to `agenttune.cli:main`, which dispatches on the first argument: `decide ...`
routes to the DECIDE sub-app; everything else routes to the top-level app
(`version`, `pipeline`, `train`).

## Top-level commands

### `agenttune version`

Print the installed AgentTune version.

### `agenttune pipeline`

Run a DECIDE decision pipeline once (inference mode) and print the verdict. Run from the
repo root (this defaults to loading `./config.yaml`):

```bash
agenttune pipeline --template bfsi/kyc_triage --input "some input text" --output decision.json
```

| Option | Default | Description |
|---|---|---|
| `--template` | *required* | Template ID (e.g. `bfsi/kyc_triage`) or path to a template YAML. |
| `--input` | *required* | Input text, or `@path` to read it from a file. |
| `--config` | `None` | Path to a global `config.yaml` (optional; see [Configuration](configuration.md)). |
| `--output` | `decision.json` | Where to write the decision JSON. |

Exits non-zero and prints the error if the pipeline run itself reports one.

### `agenttune train`

Train an agentic RL adapter via `agenttune.api.train_agentic`.

```bash
agenttune train --algorithm grpo --model Qwen/Qwen2.5-1.5B-Instruct --dataset my/dataset --output ./agenttune-run
```

| Option | Default | Description |
|---|---|---|
| `--algorithm` | *required* | `grpo` \| `dpo` \| `ppo` \| `rloo` \| `bco`. |
| `--model` | *required* | Base model name or path. |
| `--dataset` | *required* | Training dataset: HF dataset id or local path. |
| `--output` | `./agenttune-run` | Output directory for the trained adapter. |

Agentic training needs a GPU and a compatible `trl`/`torch` stack; the command validates
arguments and constructs the trainer immediately, but the run itself only executes in a
suitable environment. See [Getting Started](../getting-started/installation.md) for the
base install, which already includes everything needed for real training.

## `agenttune decide`: the DECIDE sub-app

```bash
agenttune decide <run|list|validate|show|init> [options]
```

### `agenttune decide run`

Run a decision pipeline.

| Option | Default | Description |
|---|---|---|
| `--template` | *required* | Template ID, e.g. `bfsi/kyc_triage`. |
| `--input` | *required* | Input text or `@filepath` (plain text or JSONL). |
| `--output` | `decision.json` | Output file path (inference mode). |
| `--config` | `None` | Config file path (optional). |
| `--mode` | `None` | `inference` \| `collect` \| `eval` \| `train`; overrides `run_mode` in `config.yaml` when set. |

Modes:

- **`inference`**: single-shot execution (default). Writes `decision.json` + `audit.jsonl`.
- **`collect`**: episode loop: runs `episode.n_episodes`, writes training records to
  `collect.output_path`.
- **`eval`**: run against a labelled test set and check thresholds.
- **`train`**: collect + auto-trigger the `TrainerConfigBridge`.

### `agenttune decide list`

List available templates, optionally filtered with `--category` (e.g. `bfsi`, `generic`).
Prints each template's ID, name, version, and description.

### `agenttune decide validate`

Validate a template's resolved YAML against its schema.

```bash
agenttune decide validate --template bfsi/kyc_triage
```

Prints the template's ID, name, version, and stage count on success (`✓`), or the
validation error on failure (`✗`, non-zero exit).

### `agenttune decide show`

Dump a template's fully-resolved YAML configuration (after `extends:` merging) to stdout.

```bash
agenttune decide show --template bfsi/kyc_triage
```

### `agenttune decide init`

Write a starter `config.yaml` from the packaged example.

```bash
agenttune decide init --output config.yaml
```
