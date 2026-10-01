# EventLog: the shared trajectory schema

`agenttune.agentic.events` (`src/agenttune/agentic/events.py`): one dataclass file, pure
stdlib (`json`, `uuid`, `dataclasses`, `enum`), no imports from the rest of the repo. Every
other page on this site that mentions "trajectory" is ultimately talking about this file.
See [Python API: Agentic Spine](../user-guide/agentic-spine.md) for the one-page
inventory; this page reads the module method-by-method.

The module docstring calls it "Phase 1 of the spine" and describes the intent precisely:

> Additive layer: projects the existing full-tier `agentic.trajectory.Trajectory` and
> light-tier DECIDE/eval structures into one `EventLog`. Nothing existing is modified.

That "additive" framing matters. `EventLog` does not replace `Trajectory`, `PipelineState`,
or the `{tool_calls, tool_outputs}` eval dict. Those three formats still exist, are still
produced by their own code paths, and are still consumed directly in places. `EventLog` is a
fourth format that any of the other three can be **projected into** (via `from_*`
classmethods) or **projected out of** (via `to_eval_dict()` / `to_audit_records()`), so code
written once against `EventLog` (a strategy, an evaluator, the self-healing detector) works
against all three sources without rewriting any of them.

## `EventKind`: 8 kinds, verified against the enum

```python
class EventKind(Enum):
    TEXT = "text"                    # model-emitted text
    REASONING = "reasoning"          # model-emitted reasoning/thought
    TOOL_CALL = "tool_call"          # a tool/action invocation
    TOOL_RESULT = "tool_result"      # the tool's returned output
    OBSERVATION = "observation"      # environment/state fed back to the agent
    TURN_COMPLETE = "turn_complete"  # end of one agent turn
    REWARD = "reward"                # a reward signal (see scope)
    MEMORY_OP = "memory_op"          # ADD/UPDATE/DELETE/NOOP on memory
```

These are exactly the eight values defined in `events.py`, no more, no fewer. A few notes
from reading the rest of the module against this list:

- **`MEMORY_OP` is declared but never emitted anywhere in this file.** None of the three
  `from_*` projections append a `MEMORY_OP` event, and none of the three
  export/`as_dataset_rows` methods branch on it. It exists in the enum for memory-backed
  strategies (`MemoryReActStrategy` et al., see
  [Python API: Agentic Spine](../user-guide/agentic-spine.md))
  to append later; nothing in `events.py` itself produces one.
- **`REWARD` is the only kind with a meaningful `scope`.** The `Event.scope` field is
  documented as "REWARD only: step|turn|episode", and every reward-emitting site in this file
  sets one: `from_trajectory` emits a per-step reward with `scope="step"` and one final
  episode reward with `scope="episode"`; `from_pipeline_state` emits a single
  `scope="episode"` reward from the pipeline's confidence.
- **`TEXT` and `REASONING` are treated identically by `as_dataset_rows`** (see below); both
  become an `assistant` message with weight `1.0`. The distinction exists in the schema (a
  reasoning/thought event vs. a final response) but collapses at SFT-row time.

## The `Event` and `EventLog` dataclasses

```python
@dataclass
class Event:
    kind: EventKind
    payload: dict
    token_span: Optional[tuple[int, int]] = None   # policy-emitted span; full tier only
    logprobs: Optional[list[float]] = None          # full tier only
    scope: Optional[str] = None                     # REWARD only: step|turn|episode

@dataclass
class EventLog:
    events: list[Event] = field(default_factory=list)
    tier: str = "light"                     # "light" (observational) | "full" (maskable)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
```

`payload` is a bare `dict`; there is no per-kind payload schema enforced anywhere in this
file. What keys show up depends entirely on which code populated the event; the projections
below are the closest thing to a payload contract, and it is worth reading them literally
because downstream code (`as_dataset_rows`, `to_eval_dict`, `to_audit_records`) depends on
the exact key names (`"text"`, `"action"`, `"output"`, `"value"`, `"error"`) matching.

`EventLog` supports `append`, `__iter__`, `__len__`, and:

```python
def rewards(self, scope: str) -> list[float]:
    return [e.payload["value"] for e in self.events
            if e.kind is EventKind.REWARD and e.scope == scope]
```

Note this indexes `e.payload["value"]` with `[...]`, not `.get("value")`. A `REWARD` event
built without a `"value"` key raises `KeyError` here, on purpose, rather than silently
returning `None`. Every `REWARD` event this module itself creates does carry `"value"`.

## Two tiers: `light` vs `full`

The tier is a plain string field (`"light"` or `"full"`), defaulted to `"light"`. Nothing in
`EventLog` validates the string at construction time; the only place tier is actually
enforced is the private gate:

```python
def _require_full(self, op: str) -> None:
    if self.tier != "full":
        raise ValueError(
            f"{op} requires a full-tier EventLog (token spans/logprobs come only "
            f"from the harness/rollout path); this log is tier={self.tier!r}."
        )
```

Two methods call `_require_full` and therefore raise `ValueError` on a light-tier log:

| Method | What it needs from `full` tier |
|---|---|
| `masked_tokens()` | Returns `[e.token_span for e in self.events if e.token_span is not None]`, needs real `token_span` tuples, which only a rollout/harness path with logprob access ever sets. |
| `as_dataset_rows("sft")` | Builds SFT training rows. Gated because training on a light-tier (observational-only) log (one built from a DECIDE `PipelineState` or a bare eval dict) makes no sense; those logs never carry token spans or logprobs, only text. |

Everything else (`rewards()`, `to_eval_dict()`, `to_audit_records()`, iteration, `len()`)
works on either tier. This is the practical difference the two tiers create: `light` logs are
read by UI/eval/heal code paths that only need the *content* of a trajectory (what tool was
called, what came back, what the reward was); `full` logs are additionally the ones you can
actually train a model on, because they carry the token-level information a loss function
needs.

Which projection produces which tier is fixed by the classmethod, not user-selectable:

| Classmethod | Tier produced |
|---|---|
| `from_trajectory` | `full` |
| `from_pipeline_state` | `light` |
| `from_eval_dict` | `light` |

## `as_dataset_rows("sft")`: the exact segment-weighting scheme

This is the method that turns a `full`-tier `EventLog` into TRL-shaped SFT rows. Reading the
body precisely (not the acronym-level summary) is worth it because the weights encode a real
design decision (segment-aware loss masking), and the comment in the source states the
intent inline per branch:

```python
messages: list[dict] = []
segment_weights: list[float] = [] # 1.0 for reasoning, 2.0 for action, 0.0 for observation

for e in self.events:
    if e.kind in (EventKind.REASONING, EventKind.TEXT):
        messages.append({"role": "assistant", "content": e.payload.get("text", "")})
        segment_weights.append(1.0) # Normal loss for reasoning (thought)
    elif e.kind is EventKind.TOOL_CALL:
        bare_calls = self._unwrap_tool_calls(e.payload.get("action") or {})
        if not bare_calls:
            bare_calls = [e.payload.get("action") or {}]
        for call in bare_calls:
            messages.append({"role": "assistant", "content": f"<tool_call>{json.dumps(call)}</tool_call>"})
            segment_weights.append(2.0)  # Double penalty for messing up syntax/actions (Segment-Aware Loss)
    elif e.kind is EventKind.TOOL_RESULT:
        messages.append({"role": "tool", "content": str(e.payload.get("output", ""))})
        segment_weights.append(0.0) # E1 Masking: Do not train on environment observations
    elif e.kind is EventKind.OBSERVATION:
        messages.append({"role": "user", "content": str(e.payload.get("text", ""))})
        segment_weights.append(0.0)

return [{"messages": messages, "segment_weights": segment_weights, "loss_mask": [w == 0.0 for w in segment_weights]}]
```

The real weights, verified against the source, not approximated:

| Event kind | Chat role | Segment weight | Why |
|---|---|---|---|
| `REASONING` / `TEXT` | `assistant` | **1.0** | Normal loss: the model's own reasoning/final text is a standard training target. |
| `TOOL_CALL` | `assistant` | **2.0** | Double loss weight. The comment names this "Segment-Aware Loss": getting tool-call syntax/arguments right is weighted more heavily than getting free-text reasoning right, since a malformed tool call is a harder failure mode than imprecise prose. |
| `TOOL_RESULT` | `tool` | **0.0** | Masked out: "E1 Masking: Do not train on environment observations." The model should never be trained to *predict* what a tool returned. |
| `OBSERVATION` | `user` | **0.0** | Same masking, for environment-fed observation text. |

`loss_mask` is derived, not independent: `[w == 0.0 for w in segment_weights]`, i.e. `True`
exactly where the weight is zero (tool results and observations). A single call to
`as_dataset_rows("sft")` returns a **list containing one row**: the whole `EventLog` becomes
one training example, not one row per step.

Two details worth calling out that the raw weight table doesn't show:

- **Multi-call steps stay parseable.** A `TOOL_CALL` event's `payload["action"]` is unwrapped
  by `_unwrap_tool_calls` (a `@staticmethod`) before serialization. Real rollouts record
  `action = {"tool_calls": [openai-function-call, ...]}` (per the source comment, from
  `rollout_engines/rollout_factory.py`); this helper detects that wrapper, converts each
  OpenAI-style `{"type": "function", "function": {...}}` call to the bare
  `{"name": ..., "arguments": <dict>}` shape the SFT chat template expects, and, critically,
  parses any string-encoded `arguments` back into a JSON object with `json.loads`. Each
  unwrapped call becomes its **own** `assistant` message (`<tool_call>{...}</tool_call>`),
  so a step with three simultaneous tool calls produces three separate weight-2.0 messages,
  not one message with an unparseable list embedded in it.
- **This exists because the training-time parser is strict.** The comment in the source is
  explicit about why: `rollout_factory._extract_tool_calls()` does `json.loads()` on this
  tag's content at inference time and only accepts a dict carrying a top-level `"name"` key.
  Feeding it the raw OpenAI wrapper via an f-string would silently teach the model a format
  its own parser can't read back.

## `to_audit_records()`: why `Project.heal()` works and DECIDE's own writer doesn't

This is the method that makes `EventLog`-based self-healing actually functional, in contrast
with a real, documented gap in DECIDE's own closed loop. See
[Known Issues](../community/known-issues.md) for the full story: DECIDE's `AuditWriter`
writes `pipeline_id`/`stage_id` fields with no `state_snapshot`, but
`FailureDetector.scan_audit_log` requires every line to carry `trajectory_id`, `stage_name`,
and a dict `state_snapshot`, so pointed at a *real DECIDE-generated* `audit.jsonl`, the
detector finds zero failures, forever.

`to_audit_records()` sidesteps that gap entirely by emitting records in the schema the
detector actually expects, verified against the source:

```python
def to_audit_records(self) -> list[dict]:
    records: list[dict] = []
    for e in self.events:
        if e.kind is EventKind.TOOL_CALL:
            action = e.payload.get("action") or {}
            name = action.get("name") if isinstance(action, dict) else str(action)
            args = action.get("arguments", {}) if isinstance(action, dict) else {}
            name = name or "unknown"
            stage_name = name
            if args:
                stage_name = f"{name} {_json.dumps(args, sort_keys=True, default=str)}"
            records.append({
                "trajectory_id": self.id,
                "stage_name": stage_name,
                "tool_name": name,
                "stage_type": "tool_call",
                "state_snapshot": {"arguments": args},
                "status": e.payload.get("status", "ok"),
            })
        elif e.kind is EventKind.TOOL_RESULT and e.payload.get("error") and records:
            records[-1]["status"] = "error"
            records[-1]["error_details"] = str(e.payload.get("error"))
    return records
```

Every field `scan_audit_log` requires is present: `trajectory_id` is set from the log's own
`id` (a UUID generated at construction), `stage_name` is a real string, and `state_snapshot`
is always a dict (`{"arguments": args}`, even when `args` is empty). A following
`TOOL_RESULT` event carrying a truthy `"error"` in its payload retroactively marks the
**previous** record's status as `"error"` and attaches `error_details`; this is why the loop
checks `and records` (there must be a preceding tool-call record to mutate).

The `stage_name` construction is deliberate, not incidental, and the source comment explains
why: it folds the tool name **and** its arguments into one string
(`f"{name} {json.dumps(args, sort_keys=True, default=str)}"`), rather than using the bare tool
name alone. `FailureDetector`'s loop-collapse check counts repeated `stage_name`s. In a
DECIDE pipeline a "stage" is a fixed graph node, so any revisit is genuinely a routing loop.
A tool call is different: an agent can legitimately call the same tool twice with different
arguments (ordinary ReAct behavior) without that being a loop. Folding the arguments into
`stage_name` means only a truly identical repeated action, same tool, same arguments, ever
trips `loop_collapse`.

This is exactly the mechanism `Project.heal()` (`agentic/project.py`) rides on: it calls
`log.to_audit_records()` on each of its own collected trajectories, writes the combined
records to a temp JSONL, and hands that file straight to a real
`FailureDetector.scan_audit_log()`, no mock, no simplified re-implementation:

```python
def heal(self, *, detector=None, max_revisits: int = 3):
    detector = detector or FailureDetector(max_revisits=max_revisits)
    records = [r for log in self._trajectories for r in log.to_audit_records()]
    ...
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    failures = list(detector.scan_audit_log(path))
    ...
    return failures
```

So `EventLog.to_audit_records()` is the piece that makes `agenttune.agentic.Project.heal()`
detect real failures on real trajectories; the schema mismatch documented in
[Known Issues](../community/known-issues.md) is specific to DECIDE's own `AuditWriter`
output; it doesn't apply to spine-collected trajectories that go through this projection.

## `to_eval_dict()`: the wrap into the existing evaluator

```python
def to_eval_dict(self) -> dict:
    tool_calls = [e.payload.get("action") for e in self.events
                  if e.kind is EventKind.TOOL_CALL and "action" in e.payload]
    tool_outputs = [e.payload.get("output") for e in self.events
                    if e.kind is EventKind.TOOL_RESULT]
    return {"tool_calls": tool_calls, "tool_outputs": tool_outputs}
```

This produces exactly the `{tool_calls, tool_outputs}` shape `TrajectoryEvaluator` in
`eval/agentic/trajectory_eval.py` consumes directly (see
[Trajectory metrics reference](trajectory-metrics.md) for what it does with that dict). Note
the `TOOL_CALL` filter requires `"action" in e.payload`. An event of that kind with no
`"action"` key is silently skipped, not included as `None`; `TOOL_RESULT` has no equivalent
guard and always contributes (`e.payload.get("output")`, which can be `None`). This method is
the exact inverse of `from_eval_dict` below, and shares the same `payload["action"]` /
`payload["output"]` key convention with the eval-relevant slice of `from_trajectory`.

## The three `from_*` projections: the actual normalization mechanism

This is the mechanism referenced throughout this site as "one schema for RL trajectories,
DECIDE states, and eval dicts", concretely, three classmethods that each read a different
native format and build an `EventLog` out of it. None of the three source formats are
modified; each classmethod only reads from its input.

### `from_trajectory(traj)`: full-tier agentic `Trajectory` → `EventLog`

```python
@classmethod
def from_trajectory(cls, traj) -> "EventLog":
    log = cls(tier="full")
    for step in traj.steps:
        if step.thought:
            log.append(Event(EventKind.REASONING, {"text": step.thought}))
        log.append(Event(EventKind.TOOL_CALL, {"action": step.action}))
        log.append(Event(EventKind.TOOL_RESULT, {"output": step.observation}))
        if step.reward is not None:
            log.append(Event(EventKind.REWARD, {"value": step.reward}, scope="step"))
        log.append(Event(EventKind.TURN_COMPLETE, {"step": step.step_number}))
    logprobs = traj.logprobs if isinstance(traj.logprobs, list) and traj.logprobs else None
    span = (0, len(logprobs)) if logprobs else None
    log.append(Event(EventKind.TEXT, {"text": traj.final_response},
                     token_span=span, logprobs=logprobs))
    log.append(Event(EventKind.REWARD, {"value": traj.reward}, scope="episode"))
    return log
```

Source: `agentic.trajectory.Trajectory`, the object a rollout/harness path produces, with
`.steps` (each a `Step` carrying `.thought`, `.action`, `.observation`, `.reward`,
`.step_number`), a top-level `.final_response`, `.reward`, and optionally `.logprobs`.

Per step: an optional `REASONING` event (only if `step.thought` is truthy), then always a
`TOOL_CALL` (wrapping `step.action`), always a `TOOL_RESULT` (wrapping `step.observation`),
an optional per-step `REWARD` (`scope="step"`, only if `step.reward is not None`), and always
a `TURN_COMPLETE` carrying the step number. After all steps: one `TEXT` event for the final
response; this is the *only* place in the whole module a `token_span` gets set, and only
when `traj.logprobs` is a non-empty list, as `(0, len(logprobs))`, followed by one episode
`REWARD`. This is why `from_trajectory` is the only projection that produces a `full`-tier log
capable of `masked_tokens()`/`as_dataset_rows()`: it's the only source object that ever
carries logprobs at all.

### `from_pipeline_state(state)`: DECIDE `PipelineState` → light-tier `EventLog`

```python
@classmethod
def from_pipeline_state(cls, state) -> "EventLog":
    log = cls(tier="light")
    log.append(Event(EventKind.OBSERVATION, {"text": state.input_text}))
    for stage_id in state.step_history:
        log.append(Event(EventKind.TOOL_CALL, {"stage": stage_id}))
        log.append(Event(EventKind.TOOL_RESULT,
                         {"output": state.stage_outputs.get(stage_id)}))
    log.append(Event(EventKind.TEXT, {"text": state.reason or ""}))
    if state.confidence is not None:
        log.append(Event(EventKind.REWARD, {"value": state.confidence / 10.0},
                         scope="episode"))
    return log
```

Source: `decide.state.PipelineState`: `.input_text`, `.step_history` (the ordered list of
stage IDs the pipeline actually visited), `.stage_outputs` (a dict keyed by stage ID),
`.reason`, `.confidence`.

The initial input becomes an `OBSERVATION` event. Each visited *stage* (not a real tool
call) becomes a `TOOL_CALL`/`TOOL_RESULT` pair; note the `TOOL_CALL` payload here uses a
different key than `from_trajectory`'s (`{"stage": stage_id}`, not `{"action": ...}`), which
is why `to_eval_dict()`'s `"action" in e.payload` guard matters: a `PipelineState`-derived
log's `TOOL_CALL` events carry no `"action"` key and are correctly excluded from
`to_eval_dict()`'s `tool_calls` list rather than appearing as `None`. The pipeline's final
`reason` becomes a `TEXT` event. `confidence` (documented elsewhere as a 0–10 scale) is
divided by 10 into an episode-scope reward, but only if it's not `None`; a pipeline that
never set a confidence produces no `REWARD` event at all, and `log.rewards("episode")` on
such a log returns `[]`.

### `from_eval_dict(d)`: the `{tool_calls, tool_outputs}` shape → light-tier `EventLog`

```python
@classmethod
def from_eval_dict(cls, d: dict) -> "EventLog":
    log = cls(tier="light")
    for call in d.get("tool_calls", []):
        log.append(Event(EventKind.TOOL_CALL, {"action": call}))
    for out in d.get("tool_outputs", []):
        log.append(Event(EventKind.TOOL_RESULT, {"output": out}))
    return log
```

Source: a plain `dict` with `tool_calls`/`tool_outputs` lists, the same shape any eval
pipeline (or `to_eval_dict()`'s own output) produces. This is the simplest and most literal
projection: every call becomes a `TOOL_CALL` with `payload["action"]`, every output becomes a
`TOOL_RESULT` with `payload["output"]`, using the *same* key names `from_trajectory` uses,
which is exactly why `to_eval_dict()` round-trips cleanly through this projection (it's
described in the source as the "inverse of `from_eval_dict`"). Note the two lists are not
paired positionally by index inside this method: all calls are appended first, then all
outputs, so a log built this way loses the step-by-step call/result interleaving that
`from_trajectory` preserves; anything reading it purely as an unordered bag of calls and
outputs (as `to_eval_dict()` itself does, and as `TrajectoryEvaluator`'s metrics do, see
[Trajectory metrics reference](trajectory-metrics.md)) is unaffected by that.

## Putting the three together

The practical effect: `agentic_metrics()` in `agentic/project.py` computes real programmatic
trajectory metrics via `EventLog.to_eval_dict()` regardless of whether the `EventLog` came
from a live rollout (`from_trajectory`), a DECIDE pipeline run
(`from_pipeline_state`), or a pre-existing eval dataset (`from_eval_dict`); the metric code
in `trajectory_eval.py` never needs to know which. The same is true in reverse for
`to_audit_records()` feeding `FailureDetector`, and for `as_dataset_rows("sft")` feeding any
SFT trainer that accepts `{messages, segment_weights, loss_mask}` rows, as long as the
source `EventLog` is `full`-tier.

See the [Local Notebooks](../notebooks/local-notebook.md) index for `EventLog`
exercised end-to-end through a real `Project` lifecycle, and
[Trajectory metrics reference](trajectory-metrics.md) for what the evaluator actually
computes once it has a `to_eval_dict()`-shaped dict in hand.
