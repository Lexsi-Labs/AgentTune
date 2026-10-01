# Production compositions: wiring DECIDE and training together

`TrainerConfigBridge` (`src/agenttune/decide/trainer_config_bridge.py`) is the one class that
turns a YAML file into the full agentic training stack: rollout engine, tools, reward
functions, an LLM judge, an optional multi-agent graph, PEFT config, and a trainer, all from
one config. `examples/` has four scripts that show what gets built with it (and,
in one case, around it) in practice: a full Decide→Train→Deploy flywheel, an SQL agent
validated by DECIDE, DECIDE-based tool routing, and DECIDE pipelines used directly as reward
functions. This page reads all five files and shows the real wiring, condensed; follow the
repo-path links for the complete versions.

Related pages: [Python API: DECIDE Engine](../reference/decide-engine.md) covers
`GraphRunner`/`CollectRunner`/`EvalRunner` on their own; [Known Issues](../community/known-issues.md)
documents exactly which audit-log-parsing paths below are real against a genuine DECIDE log
and which aren't; this page points out where each script lands on that line as it comes up,
rather than repeating the whole list.

## `TrainerConfigBridge`: the seven build phases

The class docstring lays out the phases explicitly, mirroring what the constructor comment
calls the "DECIDE_PLAN build phases":

```
Phase 1  — load config + choose algorithm
Phase 2  — build rollout engine  (transformers / vllm / api)
Phase 3  — attach tools
Phase 4  — attach reward functions / LLM judge
Phase 5  — (optional) build multi-agent AgentTuneGraph
Phase 6  — apply PEFT / LoRA
Phase 7  — create trainer and return
```

Construction just loads and env-resolves the YAML (`${VAR}` references are substituted from
`os.environ` recursively over the whole config tree via `_resolve_envs`):

```python
class TrainerConfigBridge:
    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        self.config: Dict[str, Any] = _resolve_envs(raw)
```

### Phase 1: algorithm

```python
@property
def algorithm(self) -> str:
    return self.config.get("training", {}).get("algorithm", "grpo").lower()
```

Just a config lookup with a `"grpo"` default; the real algorithm dispatch (which of GRPO /
DPO / PPO / RLOO / BCO actually gets instantiated) happens later, inside
`core.backend_factory.create_agentic_trainer`, which the bridge imports at module load and
calls at the very end of `build_trainer`.

### Phase 2: rollout engine

```python
def build_rollout_engine(self, rollout_cfg: Optional[Dict] = None):
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine
    cfg = rollout_cfg or self.config.get("training", {}).get("rollout", {})
    backend = cfg.get("backend", "transformers")
    kwargs: Dict[str, Any] = {}
    if backend == "transformers":
        kwargs["model_path"] = cfg.get("model_path", self.config.get("training", {}).get("model", "Qwen/Qwen3-0.6B"))
    elif backend == "vllm":
        kwargs["model_path"] = cfg.get("model_path", self.config.get("training", {}).get("model"))
        if "gpu_memory_utilization" in cfg:
            kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
    elif backend == "api":
        kwargs["api_model"] = cfg.get("api_model", "")
        kwargs["api_base_url"] = cfg.get("api_base_url", "")
        kwargs["api_key"] = cfg.get("api_key", "")
    else:
        raise ValueError(f"Unknown rollout backend '{backend}'. Choose: transformers, vllm, api")
    return create_rollout_engine(backend=backend, **kwargs)
```

`build_rollout_fn` layers `build_tools()` on top of this, via
`agentic.rollout_engines.rollout_factory.create_rollout_fn`. The returned callable is
described as "fully standalone, usable outside any training loop for debugging, data
collection, or inference," and takes `max_steps`/`system_prompt` straight from the same
`rollout` config block.

### Phase 3: tools

```python
def build_tools(self) -> List[Callable]:
    tools_cfg = self.config.get("training", {}).get("tools", [])
    tools: List[Callable] = []
    for tool_cfg in tools_cfg:
        tool_type = tool_cfg.get("type", "builtin")
        if tool_type == "builtin_sql":
            ...
        elif tool_type == "openenv":
            tools.extend(self._build_openenv_tools(tool_cfg))
        elif tool_type == "custom":
            module_path = tool_cfg.get("module", "")
            class_name = tool_cfg.get("class", "")
            if module_path and class_name:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                instance = cls(**tool_cfg.get("init_kwargs", {}))
                tools.append(instance)
    return tools
```

Named types map to real builtins (`builtin_sql` → `SQLDatabaseTool`, `builtin_web_search` →
`WebSearchTool`, `builtin_code` → `RunPythonTool`, `builtin_file` → `ReadFileTool`,
`builtin_bash` → `RunBashTool`), `openenv` routes to `_build_openenv_tools`; see
[Tool isolation](tool-isolation.md#via-yaml-trainerconfigbridge) for that block in full,
including the `atexit`-registered cleanup, and `custom` dynamically imports any class by
`module.class` path. Each builtin appends `t.execute` (the bound method, not the tool object)
to the list; `custom` appends the instance itself.

### Phase 4: reward functions and judge

```python
def build_reward_funcs(self) -> List[Any]:
    reward_cfg = self.config.get("training", {}).get("reward_funcs", [])
    resolved: List[Any] = []
    for rf in reward_cfg:
        if isinstance(rf, str):
            if "." in rf:
                try:
                    mod = importlib.import_module(rf.rsplit(".", 1)[0])
                    resolved.append(getattr(mod, rf.rsplit(".", 1)[1]))
                except (ImportError, AttributeError):
                    resolved.append(rf)   # fall back to the bare name string
            else:
                resolved.append(rf)
        elif callable(rf):
            resolved.append(rf)
    return resolved
```

A reward entry can be a bare registry name (`"correctness_reward"`), a `module.function`
dotted path (imported and resolved to the actual callable, e.g. how you'd reach
`hybrid_prm_reward` from `agentic.rewards.builtin_rewards.hybrid_prm.hybrid_prm_reward`, see
[Trajectory metrics reference](trajectory-metrics.md#downstream-use-what-hybrid_prm_reward-actually-reads)),
or an already-callable Python object passed straight through. `build_judge` builds one
`LLMJudge` from a `judge:` config block, choosing between `model` (API) and `model_path`
(local `transformers`/`vllm` backend) based on whether `backend` is one of those two strings.

### Phase 5: multi-agent graph

```python
def build_multi_agent_graph(self):
    from agenttune.agentic.langgraph_orchestrator import AgentTuneGraph
    ma_cfg = self.config.get("training", {}).get("multi_agent", {})
    graph = AgentTuneGraph()
    for rollout_cfg in ma_cfg.get("rollouts", []):
        engine = self.build_rollout_engine(rollout_cfg)
        rollout_fn = self.build_rollout_fn(engine, rollout_cfg)
        graph.add_rollout(rollout_cfg["name"], rollout_fn)
    for judge_cfg in ma_cfg.get("judges", []):
        judge = LLMJudge(**{k: judge_cfg[k] for k in (...) if k in judge_cfg})
        graph.add_judge(judge_cfg["name"], judge, aggregation=judge_cfg.get("aggregation", "mean"))
    router_expr = ma_cfg.get("router")
    if router_expr:
        graph.set_router(eval(f"lambda prompts: {router_expr}"))  # noqa: S307 — user-controlled config
    ...
    return graph.compile_rollout(), graph.compile_reward()
```

`AgentTuneGraph` (imported from `agenttune.agentic.langgraph_orchestrator`, itself a
re-export shim over the real implementation in `agenttune.langgraph.langgraph`; see
[Known Issues](../community/known-issues.md) for why `agenttune.langgraph` is *not* the
module to import from directly) composes multiple rollout engines and judges, with an
optional router and final-aggregation strategy, then `compile_rollout()`/`compile_reward()`
into the two callables `build_trainer` needs.

**Worth flagging plainly**: `router_expr` is passed straight into Python's `eval()` to build
a `lambda`. The `# noqa: S307` comment (suppressing the linter's normal "don't eval untrusted
input" warning) is itself an acknowledgment that this executes arbitrary code from whatever
supplied the YAML. Treat a `multi_agent.router` value the same way you'd treat any other
config-as-code, fine for a config you wrote yourself, not something to accept from an
untrusted source.

### Phase 6: PEFT

```python
def get_peft_config(self) -> Optional[Dict[str, Any]]:
    return self.config.get("training", {}).get("peft_config")
```

Just a dict passthrough; genuinely needs no `peft` import at config-build time, matching
the [Python API: DECIDE Engine](../reference/decide-engine.md) reference page's rating of
`get_peft_config()` as pure-Python (**A**).

### Phase 7: the full trainer, and the audit-log dataset extraction it falls back to

```python
def build_trainer(self, train_dataset=None, tools=None, reward_funcs=None, extra_kwargs=None):
    training_cfg = self.config.get("training", {})
    if train_dataset is None:
        train_dataset = self._extract_dataset_from_audit()
    if train_dataset is None:
        raise ValueError("No training dataset provided and no decide_bridge configured. "
                          "Pass train_dataset= or add decide_bridge section to YAML.")
    ...
    kwargs = {"algorithm": self.algorithm, "model": training_cfg.get("model", "Qwen/Qwen3-1.7B"),
              "train_dataset": train_dataset, "output_dir": training_cfg.get("output_dir", "./output"),
              **training_cfg.get("hyperparams", {}), **(extra_kwargs or {})}
    kwargs["tools"] = tools if tools is not None else self.build_tools()
    kwargs["reward_funcs"] = reward_funcs if reward_funcs is not None else self.build_reward_funcs()
    peft_cfg = self.get_peft_config()
    if peft_cfg:
        kwargs["peft_config"] = peft_cfg
    return create_agentic_trainer(**kwargs)
```

If no dataset is passed explicitly, `build_trainer` calls `_extract_dataset_from_audit()`,
which reads a `decide_bridge:` config block (`audit_path`, `stage_id`) and dispatches on
`self.algorithm` through `decide.training_bridge.DecideToTrainerBridge`:

```python
def _extract_dataset_from_audit(self):
    bridge_cfg = self.config.get("decide_bridge", {})
    if not bridge_cfg.get("enabled"):
        return None
    bridge = DecideToTrainerBridge(bridge_cfg.get("audit_path", "./audit.jsonl"))
    if self.algorithm == "dpo":
        pairs = bridge.extract_dpo_pairs(bridge_cfg.get("stage_id", "output"))
        return HFDataset.from_list(pairs) if pairs else None
    elif self.algorithm == "bco":
        labels = bridge.extract_bco_labels(bridge_cfg.get("stage_id", "output"))
        return HFDataset.from_list(labels) if labels else None
    else:  # grpo / ppo / rloo
        ds = bridge.extract_trajectories(bridge_cfg.get("stage_id", "output"))
        return ds if len(ds) > 0 else None
```

**This is where it matters which algorithm you configure.** `DecideToTrainerBridge.extract_dpo_pairs`
is a one-line delegation straight to `AuditReader.extract_dpo_pairs`, the exact method
[Known Issues](../community/known-issues.md) documents as looking for a `human_feedback`
field `AuditWriter` never writes, so against a real DECIDE-generated `audit.jsonl` it returns
an empty list every time, and this phase silently falls through to `None` (raising the
`ValueError` above if no `train_dataset` was supplied another way). `extract_bco_labels` rides
the same `AuditReader`, same gap.

`extract_trajectories`, by contrast, is its **own** parsing logic in
`decide/training_bridge.py`; it reads `pipeline_id`/`stage_id`/`stage_type` directly off
each JSONL line, which is exactly the schema `AuditWriter` really writes (per Known Issues:
"it writes `pipeline_id`/`stage_id`, no snapshot at all"). So of the three dataset-extraction
paths this phase can take, **GRPO/PPO/RLOO's `extract_trajectories` works against a real
audit log; DPO's and BCO's don't**: for DPO/BCO training data, build the dataset yourself
(e.g. from human review decisions you have independently) and pass it as `train_dataset=`
rather than relying on `decide_bridge:` extraction.

### Standalone entry point

```python
def build_trainer_from_yaml(config_path: str, train_dataset=None, **kwargs):
    bridge = TrainerConfigBridge(config_path)
    return bridge.build_trainer(train_dataset=train_dataset, extra_kwargs=kwargs or None)
```

One-line factory wrapping the same `build_trainer` call, useful when you don't need any of
the intermediate `step_build_*` phases, which exist as thin wrappers
(`step_build_rollout_engine`, `step_build_rollout_fn`, `step_build_tools`,
`step_build_reward_funcs`, `step_build_judge`, `step_build_graph`, `step_get_peft_config`,
`step_build_trainer`) for
callers (e.g. a DECIDE `tool_call` stage invoking one phase at a time) that want to drive the
bridge phase-by-phase rather than all at once.

## `examples/bridge_quickstart.py`: the Decide → Train → Deploy flywheel

This script's own module docstring is upfront about what it is: **"NOTE: This is pseudo-code
showing the flow."** It's the clearest illustration of the intended end-to-end loop even
though several of its calls ride the exact audit-extraction gap described above, worth
reading as "the shape of the flywheel," not as a script to run unmodified against a real
audit log.

```python
async def generate_audit_data():
    runner = GraphRunner.from_template(template_id="bfsi/kyc_triage", config_path="./config.yaml")
    for application in customer_applications:
        state = await runner.run(input_text=application)   # writes to ./audit.jsonl automatically

def extract_training_data():
    bridge = DecideToTrainerBridge("./audit.jsonl")
    dpo_pairs = bridge.extract_dpo_pairs(stage_id="income_agent")      # empty against a real log — see above
    bco_labels = bridge.extract_bco_labels(output_stage_id="output")   # same gap
    trajectories = bridge.extract_trajectories(stage_id="risk_assessment")  # real, matches AuditWriter's schema
    return dpo_pairs, bco_labels, trajectories

def train_grpo_model():
    trainer = train_from_audit(audit_path="./audit.jsonl", stage_id="risk_assessment", algorithm="grpo",
                                model="Qwen/Qwen2.5-1.5B-Instruct", output_dir="./runs/grpo_risk_v1",
                                tools=[{"name": "sql_query"}, {"name": "web_search"}],
                                reward_funcs=[], num_epochs=3)
    return trainer.train()

def deploy_models():
    deploy_trained_model(trained_model_path="./runs/dpo_income_v1/checkpoint-final",
                          config_path="./config.yaml", backend="transformers", backup=True)

def deploy_models_ab_test():
    deploy_trained_model(trained_model_path="./runs/dpo_income_v1/checkpoint-final",
                          config_path="./config.yaml", backend="transformers",
                          stage_model_map={"income_agent": "./runs/dpo_income_v1/checkpoint-final",
                                           "fraud_check": "gpt-4o", "kyc_extract": "gpt-4-turbo"})
```

The pattern the script demonstrates, real parts and gaps both included:

1. **Generate**: run a real DECIDE template (`bfsi/kyc_triage`) over many inputs;
   `GraphRunner` writes `audit.jsonl` automatically per run.
2. **Extract**: pull DPO pairs, BCO labels, and RL trajectories out of that log. Only the
   RL-trajectory path (used by GRPO/PPO/RLOO) actually returns anything against a genuine
   log today; DPO/BCO need a dataset built another way.
3. **Train**: all 5 algorithms via `train_from_audit` (a convenience wrapper in
   `decide/training_bridge.py` around `TrainerConfigBridge`-equivalent construction), each
   pointed at a different pipeline stage.
4. **Deploy**: `ModelDeploymentBridge`/`deploy_trained_model` (`decide/model_deployment.py`)
   rewrites `config.yaml`'s model fields, either globally or per-stage (the A/B pattern:
   `stage_model_map` lets one stage run the freshly-trained checkpoint as a canary while
   others keep their original model as baseline) with automatic backup for
   `rollback_deployment` to undo.
5. **Monitor**: `ModelDeploymentBridge.get_deployment_status` / `rollback_deployment` to
   check or revert the swap.

Full script: [`examples/bridge_quickstart.py`](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/bridge_quickstart.py).

## `use_case_agentic_1_sql_agent_with_decide_validation.py`: DECIDE as a SQL validator

This one runs today as written, no audit-log parsing at all, just a real `GraphRunner`
against a real template, repurposed by mutating its stage config in place before running it:

```python
async def create_sql_validation_pipeline():
    runner = GraphRunner.from_template("generic/text_classify")
    for stage in runner.config.get("stages", []):
        if stage["id"] == "process":
            stage["prompt"] = """Validate this SQL query and give a verdict.
SQL Query: {input_text}
Check:
1. Does it have SELECT clause?
2. Is syntax valid?
3. Is it safe (no injection risks)?
Return JSON: {{"is_valid": true/false, "issues": "list of issues", "verdict": "VALID/INVALID"}}"""
            stage["output_schema"] = {"type": "object", "properties": {
                "is_valid": {"type": "boolean"}, "issues": {"type": "string"}, "verdict": {"type": "string"}}}
    return runner

def extract_reward_from_output(stage_outputs: dict) -> float:
    if "process" in stage_outputs:
        output = stage_outputs["process"]
        if isinstance(output, dict):
            return 1.0 if output.get("is_valid", False) else 0.0
    return 0.0

# usage:
state = await runner.run(agent_output)                        # agent_output = a candidate SQL string
reward = extract_reward_from_output(state.stage_outputs)       # 1.0 valid / 0.0 invalid
```

The pattern: take an existing generic template (`generic/text_classify`), overwrite its
`process` stage's prompt and `output_schema` in memory to turn it into a SQL validator, run it
per candidate query, and read a binary reward straight off `state.stage_outputs["process"]`.
Nothing here needs training infrastructure; this is DECIDE used purely as a structured
verifier a training loop elsewhere could call per rollout.

Full script: [`examples/use_case_agentic_1_sql_agent_with_decide_validation.py`](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/use_case_agentic_1_sql_agent_with_decide_validation.py).

## `use_case_agentic_2_multi_tool_routing_with_decide.py`: DECIDE as a tool router

Same shape, different repurposed prompt; this time asking DECIDE to recommend which tool an
agent should use for a task, then converting its confidence level into a reward:

```python
stage["prompt"] = """Analyze this task and recommend which tool to use.
Task: {input_text}
Available tools:
- Database: For queries, lookups, data extraction
- API: For external service calls, integrations
- Computation: For calculations, analytics, processing
Return JSON: {{"recommended_tool": "database/api/computation", "confidence": "high/medium/low", "reasoning": "why"}}"""

def extract_tool_reward(stage_outputs: dict) -> float:
    if "process" in stage_outputs:
        output = stage_outputs["process"]
        if isinstance(output, dict):
            confidence = output.get("confidence", "low").lower()
            return {"high": 1.0, "medium": 0.6, "low": 0.3}.get(confidence, 0.0)
    return 0.0
```

The routing decision itself (`recommended_tool`) is available on `state.stage_outputs`, but
the reward function this script builds only reads `confidence`, so what it actually trains
toward is "the router feels confident about its recommendation," not "the router picked the
objectively correct tool" (there's no ground-truth tool label being checked against here). A
real deployment would want to additionally score `recommended_tool` against a known-correct
answer where one exists, rather than confidence alone.

Full script: [`examples/use_case_agentic_2_multi_tool_routing_with_decide.py`](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/use_case_agentic_2_multi_tool_routing_with_decide.py).

## `use_case_agentic_3_agent_training_with_decide_rewards.py`: DECIDE pipelines as reward models

The most direct version of "DECIDE-based reward functions": a `generic/sentiment_analysis`
template repurposed to grade an agent's *response text* directly, combining two independent
DECIDE outputs (`quality`, `confidence`) into one scalar:

```python
stage["prompt"] = """Evaluate the quality of this agent response.
Response: {input_text}
Analyze:
- Relevance to question
- Accuracy and truthfulness
- Clarity and coherence
- Helpfulness
Return JSON: {{"quality": "positive/neutral/negative", "confidence": "high/medium/low", "issues": "any problems"}}"""

def extract_rl_reward(stage_outputs: dict) -> float:
    if "process" in stage_outputs:
        output = stage_outputs["process"]
        if isinstance(output, dict):
            quality_reward = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}.get(
                output.get("quality", "negative").lower(), 0.0)
            confidence_mult = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(
                output.get("confidence", "low").lower(), 0.4)
            return quality_reward * confidence_mult
    return 0.0

async def simulate_agent_training_step(runner, agent_output: str):
    state = await runner.run(agent_output)
    return {"verdict": state.verdict, "reward": extract_rl_reward(state.stage_outputs)}
```

The script's own module docstring states the underlying idea plainly: **"Key insight: DECIDE
pipelines become reward models for RL training!"**: `quality` sets a base reward
(`positive`/`neutral`/`negative` → `1.0`/`0.5`/`0.0`), `confidence` scales it down
(`high`/`medium`/`low` → `1.0`/`0.7`/`0.4`), and the product is a reward in `[0, 1]` a real
GRPO/PPO/RLOO/DPO trainer could consume as one term in a reward function; this script itself
stops short of actually calling a trainer, only demonstrating the scoring step in isolation
across a small batch of hand-written candidate responses.

Full script: [`examples/use_case_agentic_3_agent_training_with_decide_rewards.py`](https://github.com/Lexsi-Labs/AgentTune/blob/main/examples/use_case_agentic_3_agent_training_with_decide_rewards.py).

## The common thread

All three `use_case_agentic_*` scripts share one structural trick: load an existing,
already-registered DECIDE template (`generic/text_classify`, `generic/sentiment_analysis`) via
`GraphRunner.from_template`, mutate the loaded config's stage prompt/output-schema *in memory*
before ever calling `.run()`, and then read a scalar reward straight off
`state.stage_outputs[stage_id]`. None of the three need `TrainerConfigBridge` or a
`decide_bridge:` YAML block at all; that machinery matters once you're ready to actually
train (Phase 7 above), not for using DECIDE as a live, per-rollout scoring function. The
`bridge_quickstart.py` flywheel is where `TrainerConfigBridge`'s real counterpart
(`train_from_audit`/`DecideToTrainerBridge`) and the actual training/deployment loop show up
and where the DPO/BCO audit-extraction gap actually bites, if you try to run it as literally
as its own docstring warns you not to.

See the [Local Notebooks](../notebooks/local-notebook.md) index for notebooks covering
DECIDE templates on their own, and the reward/eval side these compositions feed into.
