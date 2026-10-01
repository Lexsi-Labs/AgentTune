# DECIDE Examples

Runnable examples of DECIDE decision pipelines, using `Qwen/Qwen2.5-1.5B-Instruct` by default.
For what DECIDE is and how it relates to the closed-loop self-healing system, see
[Concepts: DECIDE & the closed loop](../docs/concepts/decide-and-closed-loop.md).

## Quick start

Run from the repo root:

```bash
# Sync (simple, blocking execution)
python examples/decide_example_text_classify_sync.py
python examples/decide_example_multi_judge_sync.py
python examples/decide_example_iterative_sync.py

# Async (parallel execution, better for batches)
python examples/decide_example_text_classify_async.py
python examples/decide_example_multi_judge_async.py
python examples/decide_example_iterative_async.py
```

## Examples included

### Example 1 — Text classification

**Sync**: `decide_example_text_classify_sync.py` · **Async**: `decide_example_text_classify_async.py`

Classifies text into POSITIVE, NEGATIVE, or NEUTRAL. Use case: customer-feedback analysis,
content moderation, intent detection.

```
Verdict: POSITIVE
Confidence: 9/10
```

### Example 2 — Multi-judge consensus

**Sync**: `decide_example_multi_judge_sync.py` · **Async**: `decide_example_multi_judge_async.py`

Three independent judges score quality; the output is the consensus verdict. Use case: QA,
auditable scoring, decisions where a single judge's call isn't enough.

```
Consensus Verdict: PASS
Consensus Score: 8.0/10
  Judge 1 score: 8
  Judge 2 score: 8
  Judge 3 score: 8
```

### Example 3 — Iterative refinement

**Sync**: `decide_example_iterative_sync.py` · **Async**: `decide_example_iterative_async.py`

Generate → evaluate → loop while score < 7 → output. Use case: code generation, content
synthesis, anything where the first draft usually isn't good enough.

```
Iterations: 2
Final Quality Score: 8/10
```

### Feature YAMLs

Three standalone DECIDE templates, one per stage type, runnable directly with `GraphRunner`:

- `decide_feature_llm_call.yaml` — a single `llm_call` stage
- `decide_feature_rules.yaml` — deterministic `rules`-stage routing
- `decide_feature_parallel.yaml` — fan-out/fan-in with a `parallel` stage

## Sync vs. async

| Aspect | Sync | Async |
|---|---|---|
| Blocking | Yes | No |
| Best for | Exploration, REPL | Production, batch jobs |
| Parallelism | One input at a time | Multiple inputs concurrently |
| Setup | Simple | Needs `asyncio.run()` |

Use sync while exploring; switch to async for production/batch pipelines.

## Configuration

Examples read `config/config.yaml`. Default:

```yaml
default_model: "Qwen/Qwen2.5-1.5B-Instruct"
judge_model: "Qwen/Qwen2.5-1.5B-Instruct"
```

Swap models by editing that file, or point at a different one with `config_path=` when
constructing the `GraphRunner`.

## Troubleshooting

**"Config file not found"** — run from the repo root, not from inside `examples/`.

**"Module not found: agenttune"** — `pip install -e .` from the repo root (see
[Getting Started](../docs/getting-started/installation.md)).

**Slow first run** — the first call downloads the model; subsequent runs use the HF cache.
Set `HF_HOME` to control where it's cached.

**CUDA out of memory** — switch to a smaller model in `config/config.yaml`, or run on CPU.

## Where to next

- [`docs/features.md`](../docs/features.md) — every AgentTune capability, linked to a
  notebook that runs it for real.
- [`docs/notebooks/05_decide_business_rules.ipynb`](../docs/notebooks/05_decide_business_rules.ipynb) —
  narrated DECIDE walkthrough.
- [`examples/USE_CASES_decide.ipynb`](USE_CASES_decide.ipynb) — 10 production-style DECIDE
  business-rule workflows (support routing, spam/toxicity detection, resume screening, and
  more), run live in one notebook.
