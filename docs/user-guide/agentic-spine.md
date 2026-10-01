# User Guide: Agentic Spine

Everything on this page is pure Python, no GPU, no network, no model download. Every
snippet below was run against this repo's actual source to confirm its output; where
output is shown, it's real output, not an illustration. If you haven't read
[Basic Concepts](../getting-started/basic-concepts.md) yet, do that first. This page
assumes you know what `EventLog`, `Harness`, `AgentStrategy`, and `Project` are for and
goes straight to how each one actually works.

Scope: `agenttune.agentic`, the 5 agent strategies, the 4 memory backends, the two
harness types, `EventLog`, the builtin tool library, and `Project`. Training a strategy's
policy with real RL is [User Guide: RL Training](rl-training.md); this page is about the
*design* layer that RL trains, not the training itself.

## `EventLog`: the shared trajectory format

An `EventLog` (`agenttune/agentic/events.py`) is a list of typed `Event`s plus a `tier`:

```python
from agenttune.agentic import Event, EventKind, EventLog

log = EventLog(tier="light")           # tier: "light" (observational) | "full" (trainable)
log.append(Event(EventKind.TOOL_CALL, {"action": {"name": "search", "arguments": {"q": "x"}}}))
log.append(Event(EventKind.TOOL_RESULT, {"output": "result A"}))
len(log)      # 2, EventLog supports len() and iteration directly
```

`Event` itself carries four fields: `kind` (an `EventKind`), `payload` (a plain dict,
its shape depends on `kind`), `token_span` (`(start, end)`, full-tier only), `logprobs`
(full-tier only), and `scope` (`REWARD` events only: `"step"` | `"turn"` | `"episode"`).
The eight `EventKind` values are `TEXT`, `REASONING`, `TOOL_CALL`, `TOOL_RESULT`,
`OBSERVATION`, `TURN_COMPLETE`, `REWARD`, `MEMORY_OP`.

### Light vs. full tier

**Light tier** is what you get from `run_episode`/`DictToolHarness`/`OpenEnvHarness`:
observational, no token spans, cheap to produce, fine for evaluation and for the closed
loop's failure detection. **Full tier** is what you get from a real rollout
(`EventLog.from_trajectory`); it carries logprobs and token spans, and only full-tier
logs are trainable. Calling a full-tier-only method on a light-tier log raises a clear
`ValueError` rather than silently returning garbage:

```python
try:
    log.as_dataset_rows("sft")     # log above is tier="light"
except ValueError as e:
    print(e)
# as_dataset_rows() requires a full-tier EventLog (token spans/logprobs come only
# from the harness/rollout path); this log is tier='light'.
```

`masked_tokens()` has the same guard for the same reason.

### Building a full-tier log: `EventLog.from_trajectory`

The full-tier path in practice comes from a real rollout, but you can build one by hand
from the library's native `Trajectory`/`Step` dataclasses
(`agenttune/agentic/trajectory/dataset.py`), which is exactly what
`Project.add_trajectory(EventLog.from_trajectory(t))` expects:

```python
from agenttune.agentic.trajectory.dataset import Trajectory, Step
from agenttune.agentic.events import EventLog

steps = [
    Step(step_number=1, state="task: what is the price of ABC?",
         thought="I should look up the price",
         action={"name": "lookup", "arguments": {"sku": "ABC"}},
         observation="price: 19.99", reward=0.5),
    Step(step_number=2, state="price: 19.99", thought=None,
         action={"name": "finish", "arguments": {"answer": "19.99"}},
         observation="19.99", reward=1.0),
]
traj = Trajectory(task="what is the price of ABC?", steps=steps,
                   final_response="The price is 19.99", reward=1.0,
                   logprobs=[-0.1, -0.2, -0.3])
log = EventLog.from_trajectory(traj)
log.tier    # 'full'
len(log)    # 11
```

Each `Step` becomes (in order) a `REASONING` event if it has a `thought`, a `TOOL_CALL`,
a `TOOL_RESULT`, a `REWARD` (`scope="step"`) if `step.reward is not None`, and a
`TURN_COMPLETE`. After all steps, one final `TEXT` event carries `traj.final_response`
(with a `token_span` derived from `traj.logprobs`, here `(0, 3)`), followed by one
episode-scoped `REWARD` carrying `traj.reward`.

### `.rewards(scope)`

Pulls out every `REWARD` event's value at a given scope:

```python
log.rewards("step")       # [0.5, 1.0], the two per-step rewards above
log.rewards("episode")    # [1.0], the one episode-level reward
```

### `.as_dataset_rows("sft")`: the SFT/distillation row shape

This is the method both `Project.train(fmt="sft")` and `Project.distill(...)` call under
the hood. It turns a full-tier log into the exact `{messages, segment_weights,
loss_mask}` shape the real SFT trainer consumes: segment-aware loss weighting, not a
flat chat transcript:

```python
rows = log.as_dataset_rows("sft")
len(rows)                       # 1, one EventLog -> one training row
rows[0]["messages"]
```

Real output for the trajectory above:

```python
[
    {"role": "assistant", "content": "I should look up the price"},
    {"role": "assistant", "content": '<tool_call>{"name": "lookup", "arguments": {"sku": "ABC"}}</tool_call>'},
    {"role": "tool", "content": "price: 19.99"},
    {"role": "assistant", "content": '<tool_call>{"name": "finish", "arguments": {"answer": "19.99"}}</tool_call>'},
    {"role": "tool", "content": "19.99"},
    {"role": "assistant", "content": "The price is 19.99"},
]
# segment_weights: [1.0, 2.0, 0.0, 2.0, 0.0, 1.0]
# loss_mask:        [False, False, True, False, True, False]
```

The weighting is deliberate: reasoning/text segments get weight `1.0`, tool-call segments
get weight `2.0` (a syntax/argument mistake there is penalized harder; this is what the
source calls Segment-Aware Loss), and `tool`/environment-observation segments get weight
`0.0` (E1 masking: the model is never trained to predict what the *environment* said).
`loss_mask` is just `weight == 0.0`, precomputed so a trainer doesn't need to recompute it.
Tool-call actions are normalized to the bare `{"name", "arguments"}` shape before encoding.
If you feed it a raw rollout action wrapped as `{"tool_calls": [...]}` (OpenAI-function-
call style, what real rollouts actually record), it unwraps and re-serializes each call
into its own message rather than teaching the model a format its own inference-time parser
can't read back.

### `.to_eval_dict()`: bridging into the existing evaluator

Projects an `EventLog` into the `{tool_calls, tool_outputs}` shape the (pre-existing)
`TrajectoryEvaluator` consumes. This is the wrapper that lets `agentic_metrics()` reuse
the real evaluator's `_calculate_tac`/`_calculate_ter`/etc. methods without reimplementing
any of them:

```python
log.to_eval_dict()
# {'tool_calls': [{'name': 'lookup', 'arguments': {'sku': 'ABC'}},
#                 {'name': 'finish', 'arguments': {'answer': '19.99'}}],
#  'tool_outputs': ['price: 19.99', '19.99']}
```

### `.to_audit_records()`: bridging into the closed loop's failure detector

Projects into one audit record per `TOOL_CALL`, matching the schema the closed loop's
`FailureDetector` scans (`trajectory_id`, `stage_name`, `tool_name`, `stage_type`,
`state_snapshot`, `status`). `stage_name` folds in the call's arguments (not just the tool
name) so a legitimate repeated tool call with *different* arguments doesn't get miscounted
as a routing loop; only a truly identical repeated action trips `loop_collapse`:

```python
records = log.to_audit_records()
records[0]
# {'trajectory_id': '...', 'stage_name': 'lookup {"sku": "ABC"}', 'tool_name': 'lookup',
#  'stage_type': 'tool_call', 'state_snapshot': {'arguments': {'sku': 'ABC'}}, 'status': 'ok'}
```

A following `TOOL_RESULT` with an `error` in its payload flips the *preceding* record's
`status` to `"error"` and adds `error_details`. `Project.heal()` is this method plus the
real `FailureDetector`, wired together. See [Project](#project-the-lifecycle-object)
below.

### Other projections: `from_pipeline_state`, `from_eval_dict`

Two more constructors normalize non-spine sources into the same schema so the rest of the
spine doesn't need to know where a trajectory came from: `EventLog.from_pipeline_state(state)`
projects a DECIDE `PipelineState` into a light-tier log (one `OBSERVATION` for the input,
a `TOOL_CALL`/`TOOL_RESULT` pair per stage in `state.step_history`, a final `TEXT` for
`state.reason`, and an episode `REWARD` of `state.confidence / 10.0` if confidence was
set). `EventLog.from_eval_dict(d)` does the inverse of `to_eval_dict()`: given a bare
`{tool_calls, tool_outputs}` dict, it produces a light-tier log. Both exist so an eval
harness or a DECIDE run can be scored/audited through the exact same code path as a spine
episode, without writing a third adapter.

## `Harness`: the environment an agent acts in

`Harness` (`agenttune/agentic/harness.py`) is a Gym-style ABC: `reset(task) ->
Observation`, `step(action) -> (Observation, reward, done, info)`, `action_space() ->
list[dict]`. It also declares `HarnessCapabilities` (`supports_snapshot`,
`supports_streaming`, `supports_tool_boundary_interrupt`, `supports_stepwise_turns`,
`max_steps`) and ships a `run_conformance()` bench that catches a harness lying about its
own capabilities.

### `DictToolHarness`: the in-process reference implementation

Runs plain Python callables as tools, no sandbox. Construction takes a `tools: dict[str,
Callable]`, plus keyword-only `max_steps` and `done_tool` (default `"finish"`):

```python
from agenttune.agentic import DictToolHarness, ReActStrategy, run_episode

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def policy(state):
    if state.step == 0:
        return {"name": "add", "arguments": {"a": 4, "b": 5}}
    if state.step == 1:
        return {"name": "multiply", "arguments": {"a": 9, "b": 3}}
    return {"name": "finish", "arguments": {"answer": "27"}}

harness = DictToolHarness({"add": add, "multiply": multiply}, max_steps=5)
harness.action_space()
# [{'name': 'add', 'arguments': {}}, {'name': 'multiply', 'arguments': {}},
#  {'name': 'finish', 'arguments': {'answer': ''}}]

log = run_episode(ReActStrategy(policy, max_steps=5), harness, "what is (4+5)*3?")
len(log)   # 10
```

The real event sequence: an `OBSERVATION` for the task, then `TOOL_CALL`/`TOOL_RESULT`/
`TURN_COMPLETE` triples per step:

```
OBSERVATION  {'text': 'what is (4+5)*3?'}
TOOL_CALL    {'action': {'name': 'add', 'arguments': {'a': 4, 'b': 5}}}
TOOL_RESULT  {'output': 9}
TURN_COMPLETE {'step': 1}
TOOL_CALL    {'action': {'name': 'multiply', 'arguments': {'a': 9, 'b': 3}}}
TOOL_RESULT  {'output': 27}
TURN_COMPLETE {'step': 2}
TOOL_CALL    {'action': {'name': 'finish', 'arguments': {'answer': '27'}}}
TOOL_RESULT  {'output': '27'}
TURN_COMPLETE {'step': 3}
```

Notice the tool's real return value (`9`, an `int`) flows straight into the next step's
`TOOL_RESULT`, while `finish`'s answer is whatever string the policy put in
`arguments["answer"]`; the harness doesn't coerce or validate either. An unknown tool
name or a raised exception inside the tool call becomes a `TOOL_RESULT` of
`"error: ..."` with `reward=-1.0` for that step rather than crashing the harness. You can
see this happening live in a policy that mistypes a tool name.

### Snapshot / restore, and the conformance bench

`DictToolHarness` sets `supports_snapshot=True` and backs it with an in-process
`pickle` blob of `(event_log, task, steps)`, explicitly documented as trusted, in-process
only, never meant to cross a process boundary:

```python
harness = DictToolHarness({"add": lambda a, b: a + b}, max_steps=3)
harness.reset("t")
blob = harness.snapshot()
harness.step({"name": "add", "arguments": {"a": 1, "b": 2}})
len(harness.event_log)     # 4
harness.restore(blob)
len(harness.event_log)     # 1, back to just the reset OBSERVATION
```

This snapshot/restore pair is what `TreeOfThoughtsStrategy.search()`-style backtracking
would use to explore a branch and roll back. See
[Tree-of-Thoughts](#5-treeofthoughtsstrategy-search-and-propose) below.

`run_conformance(harness, task="conformance-probe")` actually exercises the capability
flags rather than trusting them: if `supports_snapshot=True`, it snapshots, takes one
probe step, restores, and checks the event log really reverted; if `supports_snapshot=False`,
it checks that calling `snapshot()` really raises. Either way you get a
`ConformanceReport(passed: bool, drift: list[str])`:

```python
from agenttune.agentic.harness import run_conformance

run_conformance(harness, "probe task")
# ConformanceReport(passed=True, drift=[])
```

A harness that claims `supports_snapshot=True` but whose `restore()` doesn't actually put
the event log back the way it was shows up here as `drift=["declared supports_snapshot
but restore() did not revert state"]`; this is a real regression check, not a smoke test.

### `replay(harness, log)`

Re-executes only the `TOOL_CALL` actions of an existing `EventLog` through a *fresh*
harness instance and returns the resulting log, useful for re-running a captured
trajectory against a harness with different tool implementations (e.g. a bugfixed tool):

```python
from agenttune.agentic.harness import replay

harness.reset("what is 1+1?")
harness.step({"name": "add", "arguments": {"a": 1, "b": 1}})
harness.step({"name": "finish", "arguments": {"answer": "2"}})
log = harness.to_eventlog()

fresh = DictToolHarness({"add": lambda a, b: a + b}, max_steps=3)
replayed = replay(fresh, log)
len(replayed)   # 7, same event count as the original run
```

It seeds the fresh harness's `reset()` with the first `OBSERVATION`'s text (the original
task), then steps through every recorded `TOOL_CALL` action in order.

### `OpenEnvHarness`: sandboxed tool execution

Wraps any object exposing `reset()`/`step(action)` (an OpenEnv `Environment`, or a fake
for testing) as a `Harness`. Nothing in this module imports `openenv` at the top level; the
environment is injected, so this class works with a plain fake even without `openenv`
actually running anything:

```python
from agenttune.agentic.harness_openenv import OpenEnvHarness

class FakeStepResult:
    def __init__(self, observation, reward, done):
        self.observation, self.reward, self.done = observation, reward, done

class FakeObs:
    def __init__(self, text):
        self.text = text

class FakeEnv:
    def __init__(self):
        self.n = 0
    def reset(self):
        self.n = 0
        return FakeObs("env ready")
    def step(self, action):
        self.n += 1
        done = self.n >= 2
        return FakeStepResult(FakeObs(f"did {action}"), 1.0 if done else 0.0, done)

env = FakeEnv()
h = OpenEnvHarness(env, action_space=[{"name": "noop", "arguments": {}}], max_steps=5)
h.reset("solve the sandboxed task")
# Observation(text='env ready', metadata={})
h.step({"name": "noop", "arguments": {}})
# (Observation(text="did {'name': 'noop', 'arguments': {}}", metadata={'done': False}), 0.0, False, {'steps': 1})
h.step({"name": "noop", "arguments": {}})
# (Observation(text="did {'name': 'noop', 'arguments': {}}", metadata={'done': True}), 1.0, True, {'steps': 2})
```

`_extract()` tolerantly pulls `(observation, reward, done)` off whatever `env.step()`
returns: a real OpenEnv `StepResult`-like object exposing those three attributes, or a
bare observation (falls back to `reward=0.0, done=False`). `action_adapter` (optional,
defaults to identity) is where you'd convert the Harness's dict action into the target
env's native `Action` type for a real OpenEnv environment. Note
`capabilities.supports_snapshot=False` here always, since a remote/real OpenEnv env has no
in-process snapshot, so `run_conformance` against it should show `passed=True` (correctly
declaring it *can't* snapshot), not drift. See the corresponding
[Local Notebook](../notebooks/local-notebook.md) for the real extra.

## `AgentStrategy`: the 5 policies

Every strategy in `agenttune.agentic` reduces to the same four-method interface
(`init(task, tools) -> AgentState`, `propose(state) -> action`,
`observe(state, observation) -> AgentState`, `is_done(state) -> bool`), and every one of
them keeps model access behind an injected callable (a `policy`, a `reflect`, a
`propose_candidates`/`score` pair), so every example below runs with a scripted function
standing in for a model. Swap the scripted function for a real `model.generate(...)` call
and nothing else changes; see the corresponding
[Local Notebook](../notebooks/local-notebook.md) for exactly that swap with a real `SmolLM2-360M`.

`AgentState` is a plain dataclass: `task: str`, `events: list[Event]`, `step: int = 0`,
`done: bool = False`, `scratch: dict` (a strategy's private working area, where
`ReActStrategy` keeps the tool set, where `PlanExecuteStrategy` keeps the plan and cursor,
where `TreeOfThoughtsStrategy` keeps the search frontier).

`run_episode(strategy, harness, task) -> EventLog` is the driver every strategy plugs into
unchanged: it calls `strategy.init`, then loops `propose -> harness.step -> observe` until
`is_done`, and returns `harness.to_eventlog()`.

### 1. `ReActStrategy`

The reference implementation: one action per step, model decides what and when to stop.
Constructor: `ReActStrategy(policy: Callable[[AgentState], dict], max_steps: int = 10)`.
See the [`DictToolHarness`](#dicttoolharness-the-in-process-reference-implementation)
example above; that's a full `ReActStrategy` run. `is_done` here is purely step-count-
driven (`state.step >= self.max_steps`); the harness's own `done` (e.g. from calling
`finish`) is what actually ends most episodes earlier, via `run_episode`'s `if done:
state.done = True`.

### 2. `PlanExecuteStrategy`

Plan-and-Solve (arXiv:2305.04091): the policy is called exactly **once**, in `init`, and
must return the whole plan as a list of action dicts. `propose` then just hands back
`plan[cursor]` each step; `observe` advances the cursor and marks done once the plan is
exhausted (or `max_steps` is hit). The plan lives in `state.scratch["plan"]` /
`state.scratch["cursor"]` if you want to inspect it mid-run.

```python
from agenttune.agentic import DictToolHarness, run_episode
from agenttune.agentic.strategy_advanced import PlanExecuteStrategy

def planner(state):
    # called once, returns the WHOLE plan up front, not one step at a time
    return [
        {"name": "add", "arguments": {"a": 2, "b": 3}},
        {"name": "finish", "arguments": {"answer": "5"}},
    ]

harness = DictToolHarness({"add": lambda a, b: a + b}, max_steps=5)
strategy = PlanExecuteStrategy(policy=planner, max_steps=5)
log = run_episode(strategy, harness, "what is 2+3?")
len(log)   # 7
```

If `policy(state)` returns an empty list, `init` sets `state.done = True` immediately;
`run_episode` still calls `propose` once even on a done state in the current
implementation loop structure, but `is_done` short-circuits the `while` before another
harness step happens, so an empty plan is a safe (if useless) no-op episode, not a crash.

### 3. `ReflexionStrategy` + `run_reflexion`

Reflexion (arXiv:2303.11366): retry a task up to `max_attempts` times, carrying a verbal
self-critique from each failed attempt into the next one's context. This is the one
strategy that needs its own driver, `run_reflexion`, instead of a bare `run_episode`,
because retrying on failure is a multi-episode loop that `run_episode` doesn't do.

Constructor: `ReflexionStrategy(policy, reflect, max_attempts=3, max_steps=10,
reward_threshold=1.0)`. `policy` is the actor (same shape as ReAct's); `reflect(log) ->
str` turns a failed trajectory into one short critique string. Reflections accumulate on
`strategy.memory` (a plain `list[str]`) and get injected into *every* subsequent `init`,
both as `OBSERVATION` events (`f"Reflection: {r}"`) and under
`state.scratch["reflections"]`.

```python
from agenttune.agentic import DictToolHarness
from agenttune.agentic.strategy_advanced import ReflexionStrategy, run_reflexion

attempts = {"n": 0}

def policy(state):
    attempts["n"] += 1
    # deliberately wrong on attempt 1, right on attempt 2+, stands in for a model
    # that (eventually) corrects itself given feedback
    return {"name": "finish", "arguments": {"answer": "41" if attempts["n"] == 1 else "42"}}

def reflect(log):
    return "The previous attempt answered 41; the correct answer is 42, retry with 42."

def reward_fn(log):
    for e in log:
        if e.kind.value == "tool_result" and e.payload.get("output") == "42":
            return 1.0
    return 0.0

strategy = ReflexionStrategy(policy=policy, reflect=reflect, max_attempts=3, max_steps=2)
harness = DictToolHarness({}, max_steps=2)
attempts_log = run_reflexion(strategy, harness, "what is 6x7?", reward_fn)

len(attempts_log)                       # 2, stopped early once reward hit the threshold
[reward_fn(l) for l in attempts_log]    # [0.0, 1.0]
strategy.memory                          # ['The previous attempt answered 41; the correct answer is 42, retry with 42.']
```

`run_reflexion` itself is short: run an episode, score it with `reward_fn`, and if the
score is below `reward_threshold` append `reflect(log)` to `strategy.memory` before
looping again; it stops the moment an attempt clears the threshold, or after
`max_attempts`. Note the scripted `policy` above doesn't actually *read*
`state.scratch["reflections"]` to decide its answer; a real model-backed policy would. The
point of this toy example is to show exactly what Reflexion's bookkeeping does
mechanically, not to fake a model correcting itself.

### 4. `MemoryReActStrategy`

ReAct with a persistent, injected `BaseMemory` wired across the whole episode. `init`
**reads** the `k` most relevant items for the task (seeded both as `OBSERVATION` events,
`f"Recalled: {item.content}"`, and under `state.scratch["recalled"]`), and every
`observe` **writes** the step's observation into memory as a new episodic `MemoryItem`,
emitting a `MEMORY_OP` event so the write itself is visible in the trajectory.

Constructor: `MemoryReActStrategy(policy, memory, *, scope=None, recall_k=5,
recall_kind=None, max_steps=10)`.

```python
from agenttune.agentic import DictToolHarness, run_episode
from agenttune.agentic.strategy_memory import MemoryReActStrategy
from agenttune.agentic.memory import InContextMemory, MemoryItem, MemoryKind

mem = InContextMemory()
mem.write(MemoryItem(content="the user prefers metric units", kind=MemoryKind.SEMANTIC))

def policy(state):
    if state.step == 0:
        return {"name": "convert", "arguments": {"miles": 10}}
    return {"name": "finish", "arguments": {"answer": "16.09 km"}}

harness = DictToolHarness({"convert": lambda miles: f"{miles * 1.60934:.2f} km"}, max_steps=4)
strategy = MemoryReActStrategy(policy=policy, memory=mem, recall_k=3, max_steps=4)
log = run_episode(strategy, harness, "convert 10 miles to km")

len(mem._items)    # 3, the seeded SEMANTIC fact, plus one EPISODIC write per observe() call
[i.kind for i in mem._items]   # [MemoryKind.SEMANTIC, MemoryKind.EPISODIC, MemoryKind.EPISODIC]
```

Two `EPISODIC` writes land because `observe()` fires once per step (the tool-call step and
the finish step both count); memory here is genuinely accumulating across the episode,
not just being read once at the start. `scope` defaults to a fresh `Scope()`
(`agent_id="default", namespace="default"`) if you don't pass one. Reuse the same `Scope`
across multiple episodes/agents to share a memory pool, or give each agent its own to keep
them private.

### 5. `TreeOfThoughtsStrategy`: `.search()` and `.propose()`

Tree-of-Thoughts / LATS (arXiv:2305.10601, arXiv:2310.04406). This strategy is different
from the other four in one important way: it has **two** distinct ways to run.

Constructor: `TreeOfThoughtsStrategy(propose_candidates, score, beam_width=1,
max_steps=10, transition=None, is_goal=None, max_depth=10, max_nodes=256)`. Two callables
are required: `propose_candidates(state) -> list[dict]` (several candidate actions) and
`score(state, candidate) -> float` (a value for one candidate), and two more are optional
and only used by `.search()` (`transition`, `is_goal`).

**Via `run_episode` (`propose`/`observe`): effectively greedy.** Every step, all
candidates get scored, the top `beam_width` are kept in `state.scratch["frontier"]` (for
inspection), the full scored expansion is appended to `state.scratch["tree"]`, and the
**single top-scored candidate** is what actually fires as this step's action. A wider
`beam_width` only affects what's remembered for inspection, not which action is taken:

```python
from agenttune.agentic import run_episode
from agenttune.agentic.strategy_tree import TreeOfThoughtsStrategy
from agenttune.agentic.harness import DictToolHarness

def propose_candidates(state):
    return [
        {"name": "op", "arguments": {"kind": "multiply"}},
        {"name": "op", "arguments": {"kind": "add"}},
        {"name": "op", "arguments": {"kind": "subtract"}},
    ]

def score(state, candidate):
    kind = candidate["arguments"]["kind"]
    return 1.0 if kind in state.task else 0.0     # crude: reward the op the task names

def op_tool(kind):
    return {"multiply": 42, "add": 13, "subtract": -1}[kind]

harness = DictToolHarness({"op": op_tool}, max_steps=2)
strategy = TreeOfThoughtsStrategy(propose_candidates, score, beam_width=3, max_steps=2)
log = run_episode(strategy, harness, "please multiply the two numbers")
```

Real output: `multiply` wins both steps because it's the only candidate the crude `score`
gives a nonzero value, and `run_episode` never asks the strategy to call `finish` here (it
stops purely on `max_steps=2`):

```
OBSERVATION  {'text': 'please multiply the two numbers'}
TOOL_CALL    {'action': {'name': 'op', 'arguments': {'kind': 'multiply'}}}
TOOL_RESULT  {'output': 42}
TURN_COMPLETE {'step': 1}
TOOL_CALL    {'action': {'name': 'op', 'arguments': {'kind': 'multiply'}}}
TOOL_RESULT  {'output': 42}
TURN_COMPLETE {'step': 2}
```

**Via `.search(task, tools=None) -> dict`: a real best-first search, WITH backtracking.**
This is a separate, self-contained method that does *not* go through `run_episode` at all.
It keeps a full priority frontier (a heap) of every scored branch ever expanded, not just
the current best, so when the greedy-best path dead-ends or scores worse than an earlier
sibling, the next `heappop` naturally backtracks to that sibling instead of getting stuck.
It's bounded by `max_depth` (path length) and `max_nodes` (expansion budget), and returns
`{"found": bool, "path": [...], "goal_state": AgentState | None, "visited": int, "tree":
[...]}`.

Here greedy-first (always preferring the biggest single step) would try `+2, +2, +2, ...`
and never land exactly on a total of `3`; the search finds a 2-step path instead:

```python
import copy
from agenttune.agentic.strategy_tree import TreeOfThoughtsStrategy

def propose_candidates(state):
    if state.step >= 3:
        return []
    return [
        {"name": "move", "arguments": {"delta": 2}},
        {"name": "move", "arguments": {"delta": 1}},
        {"name": "move", "arguments": {"delta": -1}},
    ]

def score(state, candidate):
    return float(candidate["arguments"]["delta"])   # greedy always prefers +2

def transition(state, candidate):
    nxt = copy.deepcopy(state)
    nxt.scratch["total"] = state.scratch.get("total", 0) + candidate["arguments"]["delta"]
    nxt.scratch["last_action"] = candidate
    nxt.step = state.step + 1
    return nxt

def is_goal(state):
    return state.scratch.get("total") == 3

strategy = TreeOfThoughtsStrategy(propose_candidates, score, beam_width=1, max_steps=5,
                                  transition=transition, is_goal=is_goal,
                                  max_depth=3, max_nodes=100)
result = strategy.search("reach total 3")
result["found"]     # True
result["path"]       # [{'name': 'move', 'arguments': {'delta': 1}}, {'name': 'move', 'arguments': {'delta': 2}}]
result["visited"]    # 5
```

Note the custom `transition`/`is_goal`. `.search()`'s *default* `transition` (used if you
don't supply one) only tracks `scratch["last_action"]` and doesn't know your domain's state
update rule, and the default `is_goal` just checks `last_action.name == "finish"`. For
anything beyond the trivial "did we call finish" goal test, you supply both, as above.

## Memory: the 4 pluggable backends

`BaseMemory` (`agenttune/agentic/memory.py`) is the shared ABC: `write(item, *, scope=None)
-> str` (item id), `read(query=None, *, k=5, scope=None, kind=None) -> list[MemoryItem]`,
`update(id, patch)`, `delete(id)`, plus injectable no-op-by-default `consolidate(policy=...)`
/ `forget(policy=...)` hooks (a policy is `Callable[[list[MemoryItem]], list[MemoryItem]]`
returning the items to keep, trainable per Memory-R1, arXiv:2508.19828), and
`snapshot()`/`restore(state)` for rollout reset/replay. `MemoryItem` is `content: Any`,
`kind: MemoryKind = EPISODIC`, `scope: Scope`, `id: str` (auto-`uuid4`), `metadata: dict`.
`MemoryKind` has 5 values (CoALA taxonomy, arXiv:2309.02427): `WORKING`, `EPISODIC`,
`SEMANTIC`, `PROCEDURAL`, `ENTITY`. `Scope` is a frozen dataclass (`agent_id="default"`,
`namespace="default"`); two `Scope()` instances with the same fields compare equal, so
you can filter by scope without threading object identity around.

### `InContextMemory`: the ReAct scratchpad

List-backed, most-recent-`k` recall, no scoring. The default choice; the module docstring
notes raw episodes are a safer default than lossy consolidation (arXiv:2605.12978) unless
you inject a real `consolidate`/`forget` policy.

```python
from agenttune.agentic.memory import InContextMemory, MemoryItem, MemoryKind

m = InContextMemory()
for i in range(6):
    m.write(MemoryItem(content=f"event {i}", kind=MemoryKind.EPISODIC))

[i.content for i in m.read(k=3)]      # ['event 3', 'event 4', 'event 5'], most-recent-3

def keep_last_2(items):
    return items[-2:]

m.consolidate(policy=keep_last_2)
[i.content for i in m.read(k=10)]     # ['event 4', 'event 5']

snap = m.snapshot()                   # deep copy of current items
m.write(MemoryItem(content="event 6"))
m.restore(snap)
[i.content for i in m.read(k=10)]     # ['event 4', 'event 5'], the write after snapshot is gone
```

`consolidate`/`forget` are no-ops unless you pass a `policy=`; nothing happens
automatically. `snapshot()`/`restore()` deep-copy, so mutating items after a snapshot never
corrupts the saved state.

### `TrajectoryStore`: episodic replay + distillation bridge

Stores full-tier `EventLog`s (not arbitrary content: `write()` raises `TypeError` if
`item.content` isn't an `EventLog`), and adds one extra method beyond the `BaseMemory`
contract: `as_dataset(fmt="sft") -> list[dict]`, which emits SFT rows from every *full-tier*
stored trajectory (light-tier ones are silently skipped, not errored); this is the
distillation dataset path in miniature.

```python
from agenttune.agentic.memory import TrajectoryStore, MemoryItem
from agenttune.agentic.trajectory.dataset import Trajectory, Step
from agenttune.agentic.events import EventLog

steps = [Step(step_number=1, state="t", thought="think",
              action={"name": "finish", "arguments": {"answer": "ok"}},
              observation="ok", reward=1.0)]
traj = Trajectory(task="t", steps=steps, final_response="ok", reward=1.0, logprobs=[-0.1])
log = EventLog.from_trajectory(traj)   # tier='full'

store = TrajectoryStore()
store.write(MemoryItem(content=log))
len(store.read(k=5))          # 1
len(store.as_dataset("sft"))  # 1, one SFT row from the one stored full-tier trajectory
```

### `VectorMemory`: semantic retrieval

Unlike the recent-`k` drivers above, `read(query, k=...)` ranks by cosine similarity
between the query's embedding and each stored item's embedding, genuine semantic recall,
not string/keyword matching. The embedding function is **injected**
(`VectorMemory(embed=...)`), so this stays stdlib-only (cosine similarity is hand-rolled,
no numpy) and is fully testable with a tiny deterministic fake embedder:

```python
from agenttune.agentic.memory import MemoryItem
from agenttune.agentic.memory_vector import VectorMemory

VOCAB = ["stock", "market", "earnings", "frog", "amazon", "rainforest",
         "interest", "rate", "bank", "monetary", "policy"]

def embed(text):
    # toy embedder: a word-count vector over a fixed vocabulary, no model, no GPU
    text = str(text).lower()
    return [text.count(w) for w in VOCAB]

vm = VectorMemory(embed=embed)
vm.write(MemoryItem(content="The stock market rallied today on strong earnings."))
vm.write(MemoryItem(content="A new species of frog was discovered in the Amazon rainforest."))
vm.write(MemoryItem(content="The central bank raised interest rates due to monetary policy."))

top = vm.read("interest rates and monetary policy", k=3)
[i.content for i in top]
# ['The central bank raised interest rates due to monetary policy.',
#  'The stock market rallied today on strong earnings.',
#  'A new species of frog was discovered in the Amazon rainforest.']
```

The bank/rates sentence ranks first purely because its word-count vector is closest to the
query's. Swap `embed` for `sentence_transformers.SentenceTransformer(...).encode` and
nothing else in this snippet changes (see the corresponding
[Local Notebook](../notebooks/local-notebook.md)
for that swap with a real MiniLM model). `write`/`update` maintain a parallel `id ->
embedding` map (`_vecs`) so a content update re-embeds automatically; `consolidate`/
`forget` prune `_vecs` for any dropped item so the map never drifts out of sync with
`_items`. Passing `query=None` falls back to plain recent-`k`, same as `InContextMemory`.

### `GraphMemory`: relational recall with temporal decay

Grounded in Zep/Graphiti (arXiv:2501.13956) and A-MEM (arXiv:2502.12110): items are nodes,
connected by typed directed edges, so retrieval can be *relational* ("what's connected to
X") rather than flat recent-k or similarity-ranked. Stdlib-only (plain adjacency dicts,
no networkx).

**Relationships**: `link(src_id, dst_id, relation)` adds a directed edge (both nodes must
already exist; raises `KeyError` otherwise); `neighbors(id, k=1)` BFS-walks up to `k` hops
along outgoing edges (`k` here is traversal depth, unlike `read`'s `k` which is a result
count). `read(query, ...)` matches query text against node content (`_match`: case-
insensitive substring match), then returns the matched node's 1-hop neighborhood:

```python
from agenttune.agentic.memory import MemoryItem
from agenttune.agentic.memory_graph import GraphMemory

g = GraphMemory()
alice = g.write(MemoryItem(content="Alice"))
bob = g.write(MemoryItem(content="Bob"))
carol = g.write(MemoryItem(content="Carol"))
g.link(alice, bob, "manages")
g.link(alice, carol, "mentors")

[i.content for i in g.read("Alice", k=5)]     # ['Bob', 'Carol'], Alice's direct neighbors
[i.content for i in g.neighbors(alice, k=2)]  # ['Bob', 'Carol'], same here (no 2nd-hop edges exist)
```

**Temporal decay**: every node is stamped with a *logical* timestamp at write time, a
monotonic counter by default, or an injected `clock: Callable[[], Any]` (any value that
never decreases) for determinism. `read(query, ..., recency_weighted=True)` sorts a
matched node's neighbors by recency instead of insertion order, and `forget_older_than(age)`
drops every node whose logical age (`now - ts`) exceeds `age`, pruning dangling edges in
both directions:

```python
clock = {"t": 0}
def tick():
    clock["t"] += 1
    return clock["t"]

g = GraphMemory(clock=tick)
topic = g.write(MemoryItem(content="pricing policy"))
note_a = g.write(MemoryItem(content="pricing note from January"))
note_b = g.write(MemoryItem(content="pricing note from June"))
note_c = g.write(MemoryItem(content="pricing note from this week"))
g.link(topic, note_a, "has_note")
g.link(topic, note_b, "has_note")
g.link(topic, note_c, "has_note")

[i.content for i in g.read("pricing policy", k=5)]
# ['pricing note from January', 'pricing note from June', 'pricing note from this week']  -- insertion order

[i.content for i in g.read("pricing policy", k=5, recency_weighted=True)]
# ['pricing note from this week', 'pricing note from June', 'pricing note from January']  -- newest first
```

```python
g2 = GraphMemory(clock=tick)
old = g2.write(MemoryItem(content="old fact"))
newer = g2.write(MemoryItem(content="newer fact"))
newest = g2.write(MemoryItem(content="newest fact"))
[i.content for i in g2.read(k=10)]        # ['old fact', 'newer fact', 'newest fact']
g2.forget_older_than(1)
[i.content for i in g2.read(k=10)]        # ['newer fact', 'newest fact'], 'old fact' aged out
```

`recency_weighted` only changes ordering *within a matched node's neighbor results*; a
bare `read(k=5)` with no `query` still returns plain recent-`k` regardless of the flag,
because the `query is None` branch returns before the recency sort ever runs.

## The tool library

`BaseTool` (`agenttune/agentic/tools/base.py`) is the contract every builtin and custom
tool implements: a `name`/`description`, an `execute(**kwargs) -> ToolResult`, and an
optional `_parameters()` override (JSON Schema for the tool's arguments, feeds
`to_schema()`'s OpenAI-compatible function-calling schema). `ToolResult` is
`success: bool`, `output: Any`, `error: Optional[str]`, `metadata: dict`.

`ToolRegistry` (`tools/registry.py`) is a class-level registry: `ToolRegistry.get(name)`
auto-registers every builtin on first call and looks the tool up by name;
`register_custom(tool)` adds your own. **One gotcha worth flagging up front**: calling
`ToolRegistry.get()` for *any* tool, even a pure-stdlib one like `read_file`, triggers
`auto_register_builtins()`, which unconditionally imports every builtin module at once,
including the `langchain_community`-dependent ones (Slack/GitHub/Playwright/SQL/web-search).
Without `langchain_community` installed, even fetching `read_file` through the registry
raises `ImportError`. See
[Known Issues](../community/known-issues.md#looks-like-it-works-doesnt-or-gives-a-quietly-wrong-answer)
for the full note. The workaround used throughout this page is simply importing the tool
classes directly by module path instead of going through the registry.

### The zero-dependency tools

Five builtin tools need nothing beyond the standard library:
`agenttune.agentic.tools.builtin.file_tools` (`ReadFileTool`, `WriteFileTool`,
`ListDirTool`) and `code_tools` (`RunPythonTool`, `RunBashTool`), plus `search_tools`
(`GrepTool`). Every one returns a `ToolResult`, not a bare value:

```python
from agenttune.agentic.tools.builtin.file_tools import ReadFileTool, WriteFileTool, ListDirTool
from agenttune.agentic.tools.builtin.code_tools import RunPythonTool, RunBashTool
from agenttune.agentic.tools.builtin.search_tools import GrepTool
import tempfile, os

tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "note.txt")

WriteFileTool().execute(path=path, content="hello agenttune\nsecond line with TODO marker\n")
# ToolResult(success=True, output='Written to /tmp/.../note.txt', error=None, metadata={})

ReadFileTool().execute(path=path)
# ToolResult(success=True, output='hello agenttune\nsecond line with TODO marker\n', error=None, metadata={})

ListDirTool().execute(path=tmpdir)
# ToolResult(success=True, output=['note.txt'], error=None, metadata={})

GrepTool().execute(pattern="TODO", path=tmpdir)
# ToolResult(success=True, output='/tmp/.../note.txt:second line with TODO marker\n', error=None, metadata={})

RunPythonTool().execute(code="print(2+2)")
# ToolResult(success=True, output='4\n', error=None, metadata={})

RunBashTool().execute(command="echo hi && exit 0")
# ToolResult(success=True, output='hi\n', error=None, metadata={})
```

`RunPythonTool`/`RunBashTool` both run under `subprocess` with a `timeout` (default `30`s,
raises via `ToolResult(success=False, error="Timeout exceeded")` rather than hanging), and
combine stdout+stderr into `output`. `GrepTool` shells out to the real `grep` binary
(`recursive=True` by default); it's a thin subprocess wrapper, not a Python regex engine,
so it inherits real `grep`'s behavior (and quoting caveats) as-is.

`ReadFileTool.to_schema()` shows the OpenAI-compatible function schema every tool exposes,
generated from `_parameters()`:

```python
ReadFileTool().to_schema()
# {'type': 'function', 'function': {'name': 'read_file',
#   'description': 'Read the contents of a file at a given path.',
#   'parameters': {'type': 'object',
#     'properties': {'path': {'type': 'string', 'description': 'File path to read'},
#                    'encoding': {'type': 'string', 'default': 'utf-8'}},
#     'required': ['path']}}}
```

### Wiring `BaseTool` instances into a `DictToolHarness`

`DictToolHarness` expects a `dict[str, Callable]` of plain functions, not `BaseTool`
instances; a tool's `execute()` returns a `ToolResult`, not a bare value, so bridging the
two is one small wrapper: unwrap `.output` on success, raise on failure (the harness
already catches exceptions from a tool call and turns them into an error observation with
`reward=-1.0`, so raising here is the right thing to do rather than swallowing it):

```python
from agenttune.agentic import DictToolHarness, ReActStrategy, run_episode
from agenttune.agentic.tools.builtin.file_tools import ReadFileTool, WriteFileTool
from agenttune.agentic.tools.builtin.code_tools import RunPythonTool
import tempfile, os

tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "data.txt")

def call(tool):
    def _run(**kwargs):
        result = tool.execute(**kwargs)
        if not result.success:
            raise RuntimeError(result.error)
        return result.output
    return _run

tools = {"read_file": call(ReadFileTool()), "write_file": call(WriteFileTool()),
         "run_python": call(RunPythonTool())}

def policy(state):
    if state.step == 0:
        return {"name": "write_file", "arguments": {"path": path, "content": "10\n20\n30\n"}}
    if state.step == 1:
        return {"name": "read_file", "arguments": {"path": path}}
    return {"name": "finish", "arguments": {"answer": "done"}}

harness = DictToolHarness(tools, max_steps=4)
log = run_episode(ReActStrategy(policy, max_steps=4), harness, f"write numbers to {path} then read them back")
```

Real event trace: the `write_file` tool call actually creates the file, and the following
`read_file` call reads back exactly what was written:

```
TOOL_CALL    {'action': {'name': 'write_file', 'arguments': {'path': '...', 'content': '10\n20\n30\n'}}}
TOOL_RESULT  {'output': 'Written to .../data.txt'}
TOOL_CALL    {'action': {'name': 'read_file', 'arguments': {'path': '...'}}}
TOOL_RESULT  {'output': '10\n20\n30\n'}
TOOL_CALL    {'action': {'name': 'finish', 'arguments': {'answer': 'done'}}}
TOOL_RESULT  {'output': 'done'}
```

For the rest of the tool library (`HttpGetTool`/`HttpPostTool`, `WebSearchTool`,
`SlackTool`, `GitHubTool`, `PlaywrightTool`, `SQLDatabaseTool`), what each needs
(credentials, packages, browser binaries) and which ones have zero test coverage. See
[User Guide: Tool Library](tool-library.md) and
[Reference: Tools](../reference/tools-reference.md).

## `Project`: the lifecycle object

`Project` (`agenttune/agentic/project.py`) is the thing you actually instantiate day to
day: it threads one agent artifact through `build → collect → evaluate → train → distill →
heal`, all via `EventLog`, and records every stage transition as a `LifecycleEvent(stage,
kind, data)` you can inspect with `.events()`.

```python
Project(*, strategy: AgentStrategy | None = None, harness: Harness | None = None,
        workflow=None, data=None)
```

All constructor args are optional keyword-only; `Project()` alone is valid, and a stage
that needs a piece it doesn't have (e.g. `infer` needs both `strategy` and `harness`)
raises a clear `ValueError` rather than an `AttributeError` deep in some other method.
`workflow` is reserved for DECIDE integration and unused by the core spine today; `data` is
free-form project metadata the spine never reads.

### A richer worked example: 2 tools, full lifecycle

The [Quick Start](../getting-started/quickstart.md) uses one toy `add` tool to keep the
very first example minimal. Here's the same shape with two tools chained together, to show
`Project` isn't limited to a single-call toy case:

```python
from agenttune.agentic import Project, DictToolHarness, ReActStrategy

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def policy(state):
    if state.step == 0:
        return {"name": "add", "arguments": {"a": 4, "b": 5}}
    if state.step == 1:
        return {"name": "multiply", "arguments": {"a": 9, "b": 3}}
    return {"name": "finish", "arguments": {"answer": "27"}}

harness = DictToolHarness({"add": add, "multiply": multiply}, max_steps=5)
strategy = ReActStrategy(policy, max_steps=5)
proj = Project(strategy=strategy, harness=harness)

log = proj.infer("what is (4+5)*3?")                                  # one episode -> EventLog
proj.evaluate([{"task": "what is (4+5)*3?", "expected": "27"}])
# {'n': 1, 'mean_score': 1.0, 'scores': [1.0]}
```

`.infer(task)` requires both `strategy` and `harness` to be set. It's a thin wrapper
around `run_episode` that also appends the trajectory to `proj.trajectories` and emits
`("infer", "started")` / `("infer", "episode")` lifecycle events. `.collect(tasks)` just
calls `.infer` over a list and is the light-tier (no-model-required) way to gather many
episodes; for trainable full-tier trajectories from a *real* rollout engine, use
`.collect_rollout(engine, tasks, ...)` instead, which wraps `create_rollout_fn` (the same
producer GRPO uses) and projects each native `Trajectory` into a full-tier `EventLog` via
`EventLog.from_trajectory` automatically. See [RL Training](rl-training.md) for that path
end to end (it needs a real model).

`.evaluate(dataset, scorer=answer_match)` runs `scorer(self.infer(item["task"]),
item.get("expected"))` for every row and returns
`{"n", "mean_score", "scores"}`. The default `answer_match` scorer is deliberately dumb:
substring match of `str(expected)` against any `TOOL_RESULT`/`TEXT` payload. Swap in your
own `scorer` for anything more precise.

### `.evaluate_agentic(tasks)`: real programmatic metrics

Scores each task's trajectory with the pre-existing `TrajectoryEvaluator`'s programmatic
metrics (`tac`/`ter`/`arr`/`scsr`/`rad`/`lcf`, all in `[0, 1]`) via `EventLog.to_eval_dict()`
. Nothing here is reimplemented; it's a real wrapper around a real evaluator:

```python
proj.evaluate_agentic(["what is (4+5)*3?"])
# {'n': 1, 'metrics': {'tac': ..., 'ter': ..., 'arr': ..., 'scsr': ..., 'rad': ..., 'lcf': ...},
#  'per_trajectory': [{...}]}
```

This method (and the standalone `agentic_metrics(log, evaluator=None)` function it calls)
imports `agenttune.eval.agentic.trajectory_eval.TrajectoryEvaluator` lazily; that module
pulls in `datasets`, so this specific call needs that optional dependency installed even
though the rest of the spine on this page doesn't. `AgenticMetrics` and `AgenticEvalReport`
are `TypedDict`s documenting the exact keys you get back. Treat them as a frozen v1.0
contract, per the source's own comment.

### `.train(...)`: SFT or on-policy GRPO through the spine

```python
proj.train(trainer_factory=None, *, fmt="sft", rollout_engine=None, tools=None,
           max_steps=8, reward_fn=None, system_prompt=None, **trainer_kwargs)
```

`trainer_factory` is required; it must return an object exposing `.train() -> dict` (e.g.
the real `TRLSFTTrainer` / `TrlAgenticGrpo`), which is how `Project` stays GPU/dependency-
free itself while delegating the actual heavy lifting.

- **`fmt="sft"` (default)**: builds the dataset from `proj`'s **own** full-tier
  trajectories via `EventLog.as_dataset_rows("sft")` (the exact row shape verified above)
  and calls `trainer_factory(train_dataset=rows, **trainer_kwargs)`. Raises a clear
  `ValueError` up front if there are no full-tier trajectories yet (a light-tier
  `DictToolHarness` episode from `.infer()` is *not* trainable on its own; you need
  `.add_trajectory(EventLog.from_trajectory(...))` or `.collect_rollout(...)` first). This
  is also the distillation dataset path; `Project.distill` rides the same rails.
- **`fmt="grpo"`**: on-policy, requires `rollout_engine` (raises `ValueError` if omitted),
  and wires the real `create_rollout_fn(rollout_engine=...)` as the trainer's
  `rollout_func`, so rollouts happen live inside the real trainer against a real model, not
  from a precollected buffer.

Running either path for real needs a real model and (for GRPO) a GPU. See
[User Guide: RL Training](rl-training.md) for a full worked example with a real
`trainer_factory`; the shape above is what to expect, verified directly against this
method's source.

### `.distill(student, ...)`: teacher trajectories to student SFT

```python
proj.distill(student, *, trainer_factory=None, teacher_engine=None, tasks=None,
             fmt="sft", **trainer_kwargs)
```

Agentic distillation here means behavior cloning on a teacher's own captured
trajectories (SFT), not weight-level knowledge distillation. If you pass `teacher_engine` +
`tasks`, it collects teacher rollouts first via `.collect_rollout`; otherwise it distills
from whatever full-tier trajectories are already on `proj` (via `.collect_rollout(...)` or
`.add_trajectory(...)` beforehand). Either way it needs at least one full-tier trajectory;
same `ValueError` guard as `.train`. See [User Guide: Distillation](distillation.md) for
the full walkthrough with a real teacher/student pair.

### `.heal(...)`: real failure detection on the spine's own trajectories

```python
from agenttune.agentic import Project, DictToolHarness, ReActStrategy

def policy(state):
    return {"name": "finish", "arguments": {"answer": "5"}}

harness = DictToolHarness({}, max_steps=2)
proj = Project(strategy=ReActStrategy(policy, max_steps=2), harness=harness)
proj.infer("what is 2+3?")
proj.infer("what is 4+1?")

failures = proj.heal()
# [], these two clean, non-looping trajectories don't trip loop_collapse or tool_crash
```

`Project.heal()` builds its own audit records directly from `EventLog.to_audit_records()`,
independently of DECIDE's own `FailureDetector.scan_audit_log`/`AuditWriter` path: it writes
every trajectory's records to a temp JSONL, hands that to the real `FailureDetector` (or one
you pass via `detector=`), and cleans the temp file up in a `finally` block regardless of
outcome. This is a genuinely different code path from DECIDE's audit trail; the spine's own
schema always matched `FailureDetector`'s expectations, and (as of the current release)
DECIDE's `AuditWriter`/`scan_audit_log` schema now agree too, so both self-healing paths
detect failures correctly against their respective real logs. See
[Self-Healing](self-healing.md). Returns the `Failure` objects found (empty here because
nothing in this toy example loops or crashes) and emits `("heal", "detected")` with the
count and the distinct `failure_type`s seen.

### The rest of the surface: `events()`, `trajectories`, `sft_dataset()`, `add_trajectory()`, `native_trajectories`

```python
proj.events()             # list[LifecycleEvent], every stage transition so far, in order
proj.trajectories         # list[EventLog], everything infer/collect/collect_rollout produced
proj.native_trajectories  # list[Trajectory], the RL substrate, only populated by collect_rollout
proj.sft_dataset(fmt="sft")   # GPU-free preview: the exact SFT rows .train(fmt="sft") would build
proj.add_trajectory(log)      # register an externally-built full-tier EventLog for training/distillation
```

`add_trajectory` is the escape hatch for feeding a hand-built `Trajectory` (like the
`EventLog.from_trajectory` example earlier on this page) into `.train`/`.distill`/
`.sft_dataset` without running a real rollout at all:

```python
from agenttune.agentic import Project
from agenttune.agentic.trajectory.dataset import Trajectory, Step
from agenttune.agentic.events import EventLog

steps = [Step(step_number=1, state="t", thought="think",
              action={"name": "finish", "arguments": {"answer": "ok"}},
              observation="ok", reward=1.0)]
traj = Trajectory(task="t", steps=steps, final_response="ok", reward=1.0, logprobs=[-0.1])

proj = Project()                      # no strategy/harness needed for this path
proj.add_trajectory(EventLog.from_trajectory(traj))
len(proj.sft_dataset())               # 1
proj.native_trajectories              # [], add_trajectory registers the EventLog only, not a native Trajectory
```

`native_trajectories` only gets populated by `.collect_rollout(...)`, which retains both the
native `Trajectory` (the RL substrate, `to_trl_format()`) and its `EventLog` projection;
`add_trajectory` bypasses that and registers the `EventLog` directly, so
`native_trajectories` stays empty even though `.trajectories` and `.sft_dataset()` see the
new full-tier log.

## Where to go next

- **Train the policy behind any of these strategies with real RL**:
  [User Guide: RL Training](rl-training.md).
- **Compress a strong teacher design into a cheap student**:
  [User Guide: Distillation](distillation.md).
- **Run tool calls in an isolated sandbox instead of in-process**: the
  [`OpenEnvHarness`](#openenvharness-sandboxed-tool-execution) section above, and the
  [Local Notebooks](../notebooks/local-notebook.md) index.
- **Score a trajectory with an LLM judge, not just programmatic metrics**:
  [User Guide: Evaluation](evaluation.md).
- **Connect the spine to DECIDE's production audit trail**:
  [User Guide: Self-Healing](self-healing.md) and
  [Concepts: DECIDE & the closed loop](../concepts/decide-and-closed-loop.md).
- **Full exact signatures for everything on this page**:
  [Reference: Core API](../reference/core-api.md),
  [Reference: Tools](../reference/tools-reference.md),
  [Reference: Memory](../reference/memory-reference.md).
