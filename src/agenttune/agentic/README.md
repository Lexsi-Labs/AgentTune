# AgentTune — the agentic spine

`agenttune.agentic` is the **integration spine**: the connective tissue that moves one
artifact — a set of tasks on a *strategy* + *harness* — through the entire agentic
lifecycle and extracts value end to end.

## Thesis

> Everywhere else you can **run** an agent design, a memory system, or a harness. Here you
> **train, eval, distill, and heal** them — through **one trajectory schema**.

That one schema is the **two-tier `EventLog`**. Every stage of the lifecycle — a strategy
episode, a harness rollout, an RL rollout, an eval, a self-heal scan — reads and writes the
same normalized event stream, so a trajectory produced anywhere is consumable everywhere.
The `EventLog` is an *additive projection layer*: it wraps the existing `agentic.Trajectory`,
DECIDE `PipelineState`, and eval dicts without modifying them.

**The two tiers** (`events.py`):

| tier | source | carries | used for |
|------|--------|---------|----------|
| `full` | harness / rollout path | `token_span` + `logprobs` | trainable — SFT rows, GRPO masking, distillation |
| `light` | observational (`DictToolHarness`, DECIDE `PipelineState`, eval dicts) | events only | UI / eval / heal (read-only) |

A `light`-tier log can be evaluated and healed but **cannot be trained on** — the trainable
projections (`as_dataset_rows`, `masked_tokens`) raise if the tier is not `full`.

**Projection naming convention** (stable across 1.0): `as_*` returns a direct *cast* of the
log's own events (`as_dataset_rows`, `masked_tokens` — same data, different shape); `to_*`
builds a *new representation for a specific consumer* (`to_eval_dict` for the evaluator,
`to_audit_records` for the failure detector). The prefix tells you whether you are re-viewing
the log or translating it for something else.

## The lifecycle

```
build (AgentStrategy) → run (Harness) → collect_rollout → evaluate / evaluate_agentic
                                                        → train (SFT | GRPO) → distill → heal
```

`Project` (`project.py`) is the public lifecycle object that threads one artifact through
these stages, accumulating `EventLog` trajectories and emitting a typed `LifecycleEvent`
stream (`Project.events()`).

### Quickstart (GPU-free, verified)

Uses `DictToolHarness` + `ReActStrategy` + `Project` plus the deterministic
`DemoRolloutEngine` — no model, no network, no GPU.

```python
from agenttune.agentic import Project, DictToolHarness, ReActStrategy
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine

# 1) Harness = the agent's environment (the tools it can call).
def add(a, b):
    return a + b

harness = DictToolHarness({"add": add}, max_steps=4)

# 2) Strategy = the agent's policy. ReAct's policy is state -> action dict.
def policy(state):
    if state.step == 0:
        return {"name": "add", "arguments": {"a": 2, "b": 3}, "thought": "add them"}
    return {"name": "finish", "arguments": {"answer": "5"}}

strategy = ReActStrategy(policy, max_steps=4)

# 3) Project threads strategy + harness through the lifecycle.
proj = Project(strategy=strategy, harness=harness)

log = proj.infer("what is 2+3?")                 # one episode -> light-tier EventLog
report = proj.evaluate([{"task": "what is 2+3?", "expected": "5"}])   # answer_match
ametrics = proj.evaluate_agentic(["what is 2+3?"])   # real programmatic trajectory metrics

# GPU-free FULL-tier rollouts via the deterministic DemoRolloutEngine.
proj2 = Project()
logs = proj2.collect_rollout(DemoRolloutEngine(), ["what is 2+3?"], tools=[], max_steps=2)
rows = proj2.sft_dataset()      # the SFT/distill rows the spine would train on
fails = proj2.heal()            # closed-loop failure detection over collected trajectories
```

Running this prints (abridged):

```
infer -> events: 8 tier: light
evaluate -> mean_score: 1.0
evaluate_agentic -> metrics: {'tac': 0.0, 'ter': 0.5, 'arr': 0.0, 'scsr': 1.0, 'rad': 0.0, 'lcf': 0.0}
collect_rollout -> tiers: ['full']
sft_dataset -> rows: 1
heal -> failures: 0
```

### What each stage reuses (wrap, don't reimplement)

- `infer` / `collect` — run `AgentStrategy` on `Harness` via `run_episode`, projecting to `EventLog`.
- `collect_rollout` — drives a real `RolloutEngine` through the existing `create_rollout_fn`
  (the *same* producer GRPO uses), yielding native `Trajectory` objects (the RL substrate,
  in `Project.native_trajectories`) plus `full`-tier `EventLog`s.
- `evaluate` — scores with `answer_match` (or any `scorer(log, expected) -> float`).
- `evaluate_agentic` — scores each trajectory with the existing `TrajectoryEvaluator`'s
  programmatic metrics (`tac, ter, arr, scsr, rad, lcf`) via `EventLog.to_eval_dict()` — no
  model, no network. Reference-dependent and judge/semantic metrics activate when their
  inputs (plans, golden trajectories, litellm) are supplied.
- `train(fmt='sft')` — builds SFT rows from the project's own `full`-tier trajectories
  (`as_dataset_rows`, whose `messages` schema matches the real SFT trainer) and delegates to
  a `trainer_factory` returning an object with `.train() -> dict`.
- `train(fmt='grpo')` — on-policy RL: wires `create_rollout_fn(rollout_engine=…)` as the real
  trainer's `rollout_func`; rollout runs inside the trainer with a real model. Requires a `rollout_engine`.
- `distill` — agentic distillation (behavior cloning, not weight-KD): SFT a small `student`
  on a teacher's `full`-tier trajectories; rides the same rails as `train(fmt='sft')`.
- `heal` — projects trajectories to audit records (`to_audit_records`) and scans them with the
  real closed-loop `FailureDetector` (`loop_collapse` / `tool_crash`).

`Project.train`/`distill` take a `trainer_factory` (not a live trainer), so the `Project`
object itself stays GPU- and dependency-free while delegating to the real runner.

## The three pillars

New capability layers, each additive, each speaking `EventLog`.

### 1. Agent design — `strategy.py`, `strategy_advanced.py`, `strategy_memory.py`, `strategy_tree.py`

The policy layer. Every design reduces to `init → propose → observe → is_done`, with model
access behind an injected `policy` callable (testable without a model).

- `AgentStrategy` (base), `AgentState`
- `ReActStrategy` — the reference (ReAct, arXiv:2210.03629); driven by `run_episode`
- `PlanExecuteStrategy` — Plan-and-Solve (arXiv:2305.04091)
- `ReflexionStrategy` + `run_reflexion` — Reflexion (arXiv:2303.11366): retry loop seeded with verbal self-reflections
- `MemoryReActStrategy` — ReAct that recalls from an injected `BaseMemory` on `init` and writes observations back on `observe` (wires the memory pillar into the loop)
- `TreeOfThoughtsStrategy` — Tree-of-Thoughts / LATS (arXiv:2305.10601, arXiv:2310.04406): injected `propose_candidates` + `score`; greedy beam per step, plus `search()` for best-first frontier expansion with real backtracking

### 2. Memory — `memory.py`, `memory_vector.py`, `memory_graph.py`

Pluggable memory subsystem. CRUD plus injectable, **trainable** `consolidate`/`forget`
policies and `snapshot`/`restore` for rollout reset/replay.

- `MemoryKind` (working / episodic / semantic / procedural / entity — CoALA, arXiv:2309.02427), `Scope`, `MemoryItem`
- `BaseMemory` (base)
- `InContextMemory` — list-backed ReAct scratchpad (recent-k recall)
- `VectorMemory` — semantic recall: `read(query, k)` ranks by cosine similarity to the query embedding (embedder injected — stdlib-only, no model required to test)
- `GraphMemory` — relational recall (Graphiti / A-MEM, arXiv:2501.13956, arXiv:2502.12110): typed directed edges, `link` / `neighbors` BFS traversal, and logical-clock temporal decay (`read(recency_weighted=True)`, `forget_older_than`)
- `TrajectoryStore` — episodic replay store of `full`-tier `EventLog`s; emits distillation datasets (`as_dataset`)

Trainable memory policies follow the Memory-R1 (arXiv:2508.19828) pattern. The three drivers
span the retrieval paradigms: recent-k (`InContextMemory`), semantic (`VectorMemory`), and
relational (`GraphMemory`).

### 3. Harness — `harness.py`, `harness_openenv.py`

The agent's environment / RL env. Gym-like `reset` / `step` / `action_space`, capability
flags with a DRIFT conformance bench, and `replay()`.

- `Harness` (base), `Observation`, `HarnessCapabilities`
- `DictToolHarness` — pure-Python tool dict; produces `light`-tier logs (observational)
- `OpenEnvHarness` — wraps the OpenEnv sandbox adapter
- `run_conformance` → `ConformanceReport` — checks a harness's declared capabilities against behavior
- `replay(harness, log)` — re-execute a log's `TOOL_CALL`s through a harness

### Self-healing — `heal_loop.py`

The closed-loop recovery layer built on top of `heal` detection: detect → classify →
generate → retrain, with the litellm-bound classify/generate stages injected so the loop is
testable model-free.

- `SelfHealLoop(classifier, generator, *, trainer_factory=None)` — `.run(failures)` / `.run_on(project)` drive the full loop and return a stage-by-stage summary
- `build_dataset(examples)` — maps `TrainingExample`s to `{prompt, chosen, rejected}` preference rows using the real closed-loop `derive_preference_from_completions`
- `as_sync_classifier` / `as_sync_generator` — adapt the async litellm-bound closed-loop stages into the loop's sync injection points

## Honest scope: GPU-free vs GPU

**GPU-free (runs anywhere, for real):** everything in the quickstart — `infer`, `collect`,
`evaluate`, `evaluate_agentic` (real programmatic metrics), `collect_rollout` +
`sft_dataset` + `heal` via the deterministic `DemoRolloutEngine` (`rollout_engines/demo_engine.py`).
The demo engine emits a canned reasoning + answer with fixed logprobs so a genuine `full`-tier
`Trajectory` flows through the real rollout machinery — no model.

**Needs GPU / litellm:**

- `train(fmt='grpo')` — on-policy RL with a real model behind the rollout engine.
- SFT/distill *execution* — the dataset assembly is GPU-free; the actual `.train()` runs on GPU via the wired trainer.
- Full self-heal (`classify → regenerate → retrain`) — `heal` **detection** is GPU-free; the regenerate/retrain steps need litellm and ride on top of that detection.

### Running the service

A FastAPI backend exposes the GPU-free live spine path (create project → collect_rollout →
evaluate → distill/SFT preview → heal, plus a live lifecycle event stream over REST +
WebSocket). GRPO training and full self-heal are reported as `wired-runs-on-gpu` rather than
executed in-process — the same honesty scope the library holds.

```bash
uvicorn agenttune.agentic.service.app:create_app --factory
```

## Public surface

Exported from `agenttune.agentic` (`__init__.py`):

- **Events:** `Event`, `EventKind`, `EventLog`
- **Harness:** `Harness`, `DictToolHarness`, `OpenEnvHarness`, `Observation`, `HarnessCapabilities`, `ConformanceReport`, `run_conformance`, `replay`
- **Strategy:** `AgentStrategy`, `AgentState`, `ReActStrategy`, `PlanExecuteStrategy`, `ReflexionStrategy`, `MemoryReActStrategy`, `TreeOfThoughtsStrategy`, `run_episode`, `run_reflexion`
- **Memory:** `BaseMemory`, `InContextMemory`, `VectorMemory`, `GraphMemory`, `TrajectoryStore`, `MemoryKind`, `Scope`, `MemoryItem`
- **Project:** `Project`, `LifecycleEvent`, `answer_match`, `agentic_metrics`
- **Self-heal:** `SelfHealLoop`, `build_dataset`, `as_sync_classifier`, `as_sync_generator`
- **Substrate:** `Trajectory`, `Step`, `TrajectoryDataset`, `ToolRegistry`, `BaseTool`, `ToolResult`, `create_rollout_engine`, `create_rollout_fn`

## References

- ReAct — arXiv:2210.03629
- Reflexion — arXiv:2303.11366
- Plan-and-Solve — arXiv:2305.04091
- Tree-of-Thoughts — arXiv:2305.10601
- LATS — arXiv:2310.04406
- CoALA (memory taxonomy) — arXiv:2309.02427
- Graphiti / Zep (temporal graph memory) — arXiv:2501.13956
- A-MEM (agentic memory) — arXiv:2502.12110
- Memory-R1 (trainable memory policies) — arXiv:2508.19828
