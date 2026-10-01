# Distillation

Agentic distillation here means **behavior cloning**, not weight-level knowledge
distillation: you run a strong (usually large, slow, expensive) teacher agent, capture
everything it did as full-tier `EventLog` trajectories, turn those trajectories into an
SFT dataset, and fine-tune a small, cheap student model on it. The student learns to
imitate the teacher's *behavior*: its tool-call patterns, its output format, its
decision-making, without ever seeing the teacher's weights.

Everything here rides on `Project` (see [Basic Concepts](../getting-started/basic-concepts.md)
if you haven't met it yet) and the same `EventLog` schema that every other stage of the
spine (training, evaluation, self-healing) shares. If you've read
[User Guide: Agentic Spine](agentic-spine.md), the mental model is: `distill()` is
`train(fmt='sft')` with one twist: the trajectories come from a *teacher*, and the
resulting weights land on a *different* (usually smaller) model.

!!! note "You supply your own `trainer_factory`"
    `Project.distill()` delegates the actual training to a `trainer_factory` you provide,
    by design, so `Project` itself stays GPU/dependency-free and doesn't hard-code one
    trainer implementation. Every example on this page wraps TRL's own `SFTTrainer`
    directly inside a small `trainer_factory`; the pattern shown below in §3 is the one
    to copy for your own student model/dataset.

## TL;DR: one call, `create_distill_trainer`

Everything below (§1–§4) explains what `Project.distill()` does and how to hand-write a
`trainer_factory` for it — useful if you want full control, or if you're debugging. If
you just want to distill a teacher into a student without writing that factory yourself,
`agenttune.agentic.create_distill_trainer` builds one for you (a LoRA-wrapped TRL
`SFTTrainer`) and wraps trajectory collection + `Project.distill()` behind a
`create_agentic_trainer`-style object — construct it with plain kwargs, then call
`.train()`:

```python
from agenttune import create_distill_trainer  # or: from agenttune.agentic import create_distill_trainer

trainer = create_distill_trainer(
    student="HuggingFaceTB/SmolLM2-360M-Instruct",
    demonstrations=[   # hand-authored teacher demos — see §1 Option A below
        ("Classify sentiment: 'best purchase ever'", "SENTIMENT=positive"),
        ("Classify sentiment: 'broke on day one'", "SENTIMENT=negative"),
    ],
    output_dir="./out_distill",
    num_train_epochs=25,
    learning_rate=3e-4,
)
result = trainer.train()
```

Or distill from a live teacher model instead (§1 Option B/C):

```python
trainer = create_distill_trainer(
    student="Qwen/Qwen2.5-0.5B-Instruct",
    teacher_engine=teacher_rollout_engine,
    tasks=["Read config.yaml and report default_model.", "..."],
    tools=[my_teacher_tool],       # optional — only used with teacher_engine=+tasks=
    reward_fn=my_reward_fn,        # optional — scores the teacher rollout, forwarded to collect_rollout
    system_prompt="You are a careful assistant. Use tools when the task requires it.",
    output_dir="./out_distill",
)
result = trainer.train()
```

What's overridable, in order of how much of the default behavior you're replacing:

- **LoRA/model-loading knobs** (`use_peft`, `lora_r`, `lora_alpha`, `lora_dropout`,
  `lora_bias`, `lora_targets`, `task_type`, `dtype`, `device_map`) tune the built-in
  factory without replacing it.
- **`**sft_kwargs`** (`num_train_epochs`, `learning_rate`, `per_device_train_batch_size`,
  `save_strategy`, `report_to`, `bf16`, ...) forward straight through to `SFTConfig`,
  exactly like the `trainer_kwargs` in §4 below.
- **`trainer_factory=`** replaces the built-in TRL/LoRA factory entirely with your own
  callable (`model=, train_dataset=, **kwargs -> object with .train()`), e.g. a
  full-fine-tune loop, a different backend, or a factory with its own before/after probe.
- **`tools=`/`reward_fn=`/`system_prompt=`/`max_steps_per_turn=`** (or the lower-level
  `rollout_kwargs=`) control the internal `collect_rollout` call when distilling from a
  live `teacher_engine`; pass any custom tool or reward callable, not just built-ins.

`trainer.sft_dataset()` previews the training rows GPU-free (same rows `.train()` would
feed the factory); `trainer.trajectories`/`trainer.events()` expose the underlying
`Project`'s state. The rest of this page is the manual version of exactly what this
wraps — read on if you want to write your own `trainer_factory`, understand the dataset
schema, or use `Project` directly.

## The shape of it

```
teacher agent  ──run──▶  EventLog (tier="full")  ──as_dataset_rows("sft")──▶  SFT rows
      │                                                                          │
      │ (many tasks)                                                            ▼
      ▼                                                               trainer_factory(...)
 collect_rollout() / add_trajectory()                                          │
                                                                                 ▼
                                                                        student model weights
```

Three things have to happen, in order: **collect** teacher trajectories, **assemble** them
into an SFT dataset (this part is automatic; `Project` does it for you), and **train** a
student on that dataset via a `trainer_factory` you write. The rest of this page walks
through each step with real, runnable code.

## 1. Collect teacher trajectories

`Project.distill()` needs full-tier `EventLog`s, the tier that carries token spans and
(when available) logprobs. Light-tier logs (e.g. from `DictToolHarness`-driven `Project.infer()`)
are observational only and will be rejected. There are three ways to get full-tier
trajectories onto a `Project`, ordered from simplest to most realistic.

### Option A: construct trajectories by hand (`add_trajectory`)

Useful when your "teacher" is a fixed demonstration set rather than a live agent, e.g.
you're teaching a specific output contract and already know the exact answer you want
the student to produce for each input:

```python
from agenttune.agentic import Project, EventLog, Event, EventKind

def teacher_demo(text: str, label: str) -> EventLog:
    """One full-tier teacher trajectory: a request and the demonstrated answer."""
    log = EventLog(tier="full")
    log.append(Event(EventKind.OBSERVATION, {"text": f"Classify sentiment: '{text}'"}))
    log.append(Event(EventKind.TEXT, {"text": f"SENTIMENT={label}"}))
    return log

teacher = Project()   # no strategy/harness needed for this path
demos = [
    ("I love this product, best purchase ever", "positive"),
    ("terrible, broke on day one", "negative"),
    ("it arrived on time, works as described", "neutral"),
]
for text, label in demos:
    teacher.add_trajectory(teacher_demo(text, label))
```

Note the `tier="full"` on the `EventLog` constructor; that's what makes it eligible for
`distill()`/`train()`. A log built without it defaults to `tier="light"` and will raise a
`ValueError` the moment something calls `as_dataset_rows()` on it.

`Project.distill()` just needs full-tier `EventLog`s sitting in `self._trajectories` by
the time you call it; `add_trajectory` is the direct way to put them there. You don't
need `strategy=`/`harness=` on the `Project` for this path at all; `add_trajectory` and
`distill` don't touch either.

### Option B: collect real rollouts from a teacher engine (`collect_rollout`)

If your teacher is an actual model (not a hand-written demo set), drive it with a real
`RolloutEngine` over a list of tasks:

```python
from agenttune.agentic import Project
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine

# backend="auto" picks vLLM if installed, else transformers. Use model_path for a
# HuggingFace repo id or local checkpoint. The engine loads the model+tokenizer for you.
teacher_engine = create_rollout_engine(
    backend="transformers",
    model_path="Qwen/Qwen2.5-7B-Instruct",   # the TEACHER, bigger/slower is fine here
)

teacher = Project()
tasks = [
    "Classify sentiment: 'the delivery was late and support ignored me'",
    "Classify sentiment: 'exceeded my expectations, five stars'",
    "Classify sentiment: 'does the job, no strong feelings'",
]
logs = teacher.collect_rollout(teacher_engine, tasks, max_steps=4)
print(f"{len(logs)} full-tier trajectories collected, tier={logs[0].tier!r}")
```

`collect_rollout` wraps the same `create_rollout_fn` producer that GRPO training uses
internally; one call yields both the native `Trajectory` objects (kept as the RL
substrate, in `teacher.native_trajectories`) and full-tier `EventLog` projections of each
(in `teacher.trajectories`, which is what `distill()`/`train()` read). This is the gate
that makes distillation "real" instead of hand-authored: the teacher's actual generations,
tool calls, and (if the engine exposes them) logprobs all flow through.

`create_rollout_engine`'s useful `backend` values: `"transformers"` (local HF model, any
GPU), `"vllm"` (if vLLM is installed, faster batched generation), `"api"` (route through
litellm to a hosted model via `api_provider`/`api_model`/`api_key`), or `"auto"` to let it
pick. Passing an existing `model=`/`tokenizer=` pair skips the `from_pretrained` load if
you already have one in memory.

### Option C: teacher engine + real tools (agentic teacher, not an abstract placeholder)

The two options above show a text-only teacher. A more realistic distillation target is
an agent that actually calls tools. You want the student to learn *when* to call a tool
and *how* to format the call, not just what text to emit. Wire real tools in via
`agenttune.agentic.tools`:

```python
from agenttune.agentic import Project
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine
from agenttune.agentic.tools.registry import ToolRegistry

teacher_engine = create_rollout_engine(backend="transformers",
                                       model_path="Qwen/Qwen2.5-7B-Instruct")

# ToolRegistry.get() returns real BaseTool instances (with .name / .to_schema()) —
# collect_rollout's `tools=` wants these, not bare strings or {"name": ...} dicts.
tools = [ToolRegistry.get("read_file"), ToolRegistry.get("run_python")]

teacher = Project()
tasks = [
    "Read config.yaml and tell me what the default_model is set to.",
    "Compute the 15th Fibonacci number and report just the number.",
]
logs = teacher.collect_rollout(
    teacher_engine, tasks,
    tools=tools,
    max_steps=6,
    system_prompt="You are a careful assistant. Use tools when the task requires it.",
)
```

!!! warning "`ToolRegistry.get()` can crash on an unrelated missing dependency"
    The first call to `ToolRegistry.get(...)` triggers `auto_register_builtins()`, which
    unconditionally imports **every** builtin tool module at once, including the
    `langchain_community`-backed ones (Slack/GitHub/Playwright/SQL/web-search). Without
    `langchain_community` installed, even fetching the pure-stdlib `read_file` tool raises
    `ImportError`. See
    [Known Issues](../community/known-issues.md#looks-like-it-works-doesnt-or-gives-a-quietly-wrong-answer)
    for the exact mechanism. Install `langchain_community`, or sidestep the registry
    entirely and hand `collect_rollout(tools=...)` your own `BaseTool` subclass instances,
    anything with `.name` and `.to_schema()` (see `agenttune.agentic.tools.base.BaseTool`)
    works, registry or not.

The resulting `EventLog`s now contain real `TOOL_CALL`/`TOOL_RESULT` event pairs from the
teacher's actual tool use, which `as_dataset_rows()` (next section) turns into
`<tool_call>{...}</tool_call>` training targets: the student is being taught the
teacher's *tool-use policy*, not just its prose.

## 2. What the SFT dataset actually looks like

Before spending any GPU time, you can preview exactly what `distill()`/`train()` would
feed the trainer:

```python
rows = teacher.sft_dataset()          # GPU-free preview — same rows distill()/train() build
print(len(rows), "rows")
print(rows[0].keys())                 # dict_keys(['messages', 'segment_weights', 'loss_mask'])
```

Each row comes from `EventLog.as_dataset_rows("sft")`
(`src/agenttune/agentic/events.py`), and the schema is real and stable; this is what any
`trainer_factory` you write receives as `train_dataset`:

```python
{
    "messages": [
        {"role": "user", "content": "..."},                              # OBSERVATION
        {"role": "assistant", "content": "..."},                         # TEXT / REASONING
        {"role": "assistant", "content": '<tool_call>{"name": ..., "arguments": {...}}</tool_call>'},
        {"role": "tool", "content": "..."},                              # TOOL_RESULT
    ],
    "segment_weights": [0.0, 1.0, 2.0, 0.0],
    "loss_mask": [True, False, False, True],       # True where segment_weight == 0.0
}
```

The weights aren't arbitrary: they implement segment-aware loss:

- **`1.0`** for reasoning/text segments (`REASONING`, `TEXT` events): normal loss, the
  model should learn to produce this kind of content.
- **`2.0`** for tool-call segments (`TOOL_CALL` events): double weight, because getting
  tool-call syntax/arguments wrong is a more serious failure mode than imperfect prose.
- **`0.0`** for environment-originated segments (`TOOL_RESULT`, `OBSERVATION` events):
  the student shouldn't be trained to predict what the *environment* said back to it;
  `loss_mask` marks exactly these positions as `True` (masked out) for you.

One subtlety worth knowing if you ever inspect a `TOOL_CALL` event's raw payload: real
rollouts store the action as the OpenAI-style wrapper `{"tool_calls": [...]}`, but the
inference-time parser (`rollout_factory._extract_tool_calls`) only accepts the bare
`{"name": ..., "arguments": {...}}` shape via `json.loads`. `as_dataset_rows()` unwraps
this for you (`EventLog._unwrap_tool_calls`) and emits one `<tool_call>` message per call
even for multi-call steps; you don't need to do anything here, but it explains why the
tag content is never the raw `{"tool_calls": [...]}` dict.

`as_dataset_rows()`, and therefore `sft_dataset()`, `train()`, and `distill()`, only
accepts `fmt="sft"` today; anything else raises `ValueError`.

## 3. Write a `trainer_factory`

A `trainer_factory` is any callable returning an object with `.train() -> result`. That's
the entire contract; `Project` never inspects the result beyond passing it back to you
and logging it on the lifecycle event stream (`teacher.events()`).

### Minimal version

Wraps TRL's `SFTTrainer` directly, no LoRA, no probing:

```python
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def my_trainer_factory(*, model, train_dataset, **kwargs):
    ds = Dataset.from_list(train_dataset)
    tokenizer = AutoTokenizer.from_pretrained(model)
    student = AutoModelForCausalLM.from_pretrained(model)

    return SFTTrainer(
        model=student,
        args=SFTConfig(output_dir="./distilled-student", max_steps=100),
        train_dataset=ds,
        processing_class=tokenizer,
    )
```

Note the parameter name: `distill()` calls `trainer_factory(model=student, train_dataset=rows, **trainer_kwargs)`;
your factory needs a `model` keyword, not a hard-coded model name. `SFTTrainer` itself
already exposes a real `.train()`, so there's no wrapper class needed here; just return
the `SFTTrainer` instance directly.

### Fuller version: LoRA, before/after probe, real training config

This is closer to what you'd actually run (adapted from the corresponding
[Local Notebook](../notebooks/local-notebook.md), which runs this for real on
a GPU):

```python
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

def generate(model, tok, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class RealSFTTrainerFactory:
    """A trainer_factory that also captures a before/after generation, so you can SEE
    the student's behavior change — not just trust a loss number."""

    def __init__(self, out_dir: str, probe_prompt: str):
        self.out_dir = out_dir
        self.probe_prompt = probe_prompt
        self.model = None
        self.tok = None
        self.before = None

    def __call__(self, *, model, train_dataset, **kwargs):
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16).to("cuda")
        self.before = generate(self.model, self.tok, self.probe_prompt)

        cfg = SFTConfig(
            output_dir=self.out_dir, num_train_epochs=25, per_device_train_batch_size=4,
            learning_rate=3e-4, logging_steps=5, save_strategy="no", report_to=[],
            bf16=True, max_length=256, completion_only_loss=True,
            gradient_checkpointing=False,
        )
        # LoRA keeps the base model's general language ability intact and only
        # learns the new behavior — cheaper and less prone to catastrophic forgetting
        # than full fine-tuning on a small demonstration set.
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        return SFTTrainer(model=self.model, args=cfg,
                          train_dataset=Dataset.from_list(train_dataset),
                          processing_class=self.tok, peft_config=lora)
```

Two things about `train_dataset` in both versions above: it's a plain Python `list[dict]`
of the `{"messages", "segment_weights", "loss_mask"}` rows from the previous section, and
`Dataset.from_list(...)` is doing the only real "adaptation" work: turning that list into
whatever columnar format TRL expects. If you swap in a different trainer (a custom loop,
`DPOTrainer` for some other stage of your pipeline), that conversion step is on you; the
row schema itself doesn't change.

`completion_only_loss=True` in the example above tells TRL to mask the loss to
assistant-turn tokens using the chat template's own generation markers, a coarser,
model-agnostic version of what `segment_weights`/`loss_mask` already express at the
message level. If your trainer doesn't support `completion_only_loss`, the mask you
computed from `segment_weights` (`weight == 0.0` → don't train on this message) is the
manual equivalent; wire it in wherever your trainer accepts a per-example loss mask.

## 4. Run the distillation

With trajectories collected (§1) and a `trainer_factory` written (§3):

```python
result = teacher.distill(
    student="Qwen/Qwen2.5-0.5B-Instruct",
    trainer_factory=RealSFTTrainerFactory(
        out_dir="./distilled-student",
        probe_prompt="Classify sentiment: 'the delivery was late and support ignored me'",
    ),
)
print(result.training_loss)   # whatever your trainer_factory's .train() returns
```

`Project.distill()`:

1. Collects every full-tier trajectory currently on `self._trajectories` (from §1's
   `add_trajectory`/`collect_rollout` calls) and builds the SFT dataset from them via
   `as_dataset_rows(fmt)`.
2. Calls `trainer_factory(model=student, train_dataset=rows, **trainer_kwargs)`.
3. Calls `.train()` on whatever that returns.
4. Emits `("distill", "started", ...)` and `("distill", "done", ...)` lifecycle events
   either way (see [§6](#6-inspecting-lifecycle-events) below).

If you call `distill()` with no trajectories collected, it raises a clear `ValueError`
rather than silently training on nothing:

```
Project.distill needs teacher trajectories. Provide teacher_engine+tasks, or
collect_rollout(...) / add_trajectory(...) full-tier teacher trajectories first.
```

### Alternative: collect and distill in one call

If you haven't collected teacher rollouts yet, pass `teacher_engine=` and `tasks=`
directly; `distill()` runs `collect_rollout` for you first, then proceeds exactly as
above:

```python
result = teacher.distill(
    student="Qwen/Qwen2.5-0.5B-Instruct",
    trainer_factory=my_trainer_factory,
    teacher_engine=teacher_rollout_engine,
    tasks=["task 1", "task 2", "..."],
)
```

To also pass `tools=`/`max_steps=`/`system_prompt=` etc. through to that internal
`collect_rollout` call, nest them under `rollout_kwargs`; everything else in
`**trainer_kwargs` goes straight to your `trainer_factory`, and `distill()` specifically
pops out `rollout_kwargs` before that happens:

```python
result = teacher.distill(
    student="Qwen/Qwen2.5-0.5B-Instruct",
    trainer_factory=my_trainer_factory,
    teacher_engine=teacher_rollout_engine,
    tasks=["Read config.yaml and report default_model.", "..."],
    rollout_kwargs={"tools": tools, "max_steps": 6, "system_prompt": "..."},
)
```

Without `rollout_kwargs`, that internal `collect_rollout` call runs tool-free with the
default `max_steps=8`, fine for a text-only teacher, but if your teacher needs tools
(§1 Option C), you need `rollout_kwargs` to get them there.

## 5. `distill()` vs `train(fmt='sft')`

Both ride the exact same rails: same dataset assembly (`as_dataset_rows`), same
`trainer_factory` contract, same full-tier-trajectory requirement. The difference is just
*whose* trajectories and *whose* weights:

| | `train(fmt='sft')` | `distill(student, ...)` |
|---|---|---|
| Trajectories used | the `Project`'s own (from `collect_rollout`/`add_trajectory`) | same mechanism, but conventionally the *teacher*'s |
| `trainer_factory` called with | `train_dataset=rows` | `model=student, train_dataset=rows` |
| Typical use | fine-tune your own agent on its own (good) trajectories | compress a teacher's behavior into a smaller/cheaper model |

`Project.train()` also supports `fmt='grpo'` for on-policy RL; that path is unrelated to
distillation (it wires a `rollout_engine` in as the trainer's `rollout_func` rather than
building a static dataset) and is out of scope for this page.

## 6. Inspecting lifecycle events

Every stage of the spine (`infer`, `collect`, `collect_rollout`, `evaluate`, `train`,
`distill`, `heal`) appends a `LifecycleEvent(stage, kind, data)` to `Project._events`.
After a run:

```python
for ev in teacher.events():
    print(ev.stage, ev.kind, ev.data)
# collect_rollout started {'n': 2}
# collect_rollout done    {'n': 2}
# distill   started {'student': 'Qwen/Qwen2.5-0.5B-Instruct', 'n_rows': 6, 'n_teacher_trajectories': 2}
# distill   done    {'student': 'Qwen/Qwen2.5-0.5B-Instruct', 'result': ...}
```

Useful for a training dashboard, a test assertion, or just confirming that a `distill()`
call actually saw the trajectories you thought it did (`n_teacher_trajectories`) before
you wait on a multi-hour GPU job.

## Testing the wiring without a GPU or a model

Every piece up to "call `.train()`" is pure Python and can be exercised with zero GPU and
zero API key, using `DemoRolloutEngine` (`agenttune.agentic.rollout_engines.demo_engine`)
as the teacher and a trivial `trainer_factory` stub:

```python
from agenttune.agentic import Project
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine

class _FakeTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def train(self):
        return {"train_loss": 0.0, "n_rows": len(self.kwargs["train_dataset"])}

teacher = Project()
teacher.collect_rollout(DemoRolloutEngine(), ["task 1", "task 2", "task 3"])
result = teacher.distill(student="fake-student-id", trainer_factory=_FakeTrainer)
print(result)   # {'train_loss': 0.0, 'n_rows': 3}
for ev in teacher.events():
    print(ev.stage, ev.kind, ev.data)
```

`DemoRolloutEngine.generate()` returns a fixed reasoning+answer pair with hard-coded
logprobs, no model, no tokenizer, no network, but it still goes through the real
`create_rollout_fn`/`Trajectory` machinery, so this genuinely exercises the same
`collect_rollout → sft_dataset → distill → trainer_factory(...)` path you'd run for real,
just with a canned generation and a trainer that does nothing. Useful for CI, for
verifying your own `trainer_factory`'s signature is being called the way you expect
before you point it at a real model, or for confirming row counts match the number of
trajectories you collected.

## `native_trajectories`: the RL substrate underneath

`collect_rollout` retains the native `agentic.Trajectory` objects it produces (not just
their `EventLog` projections) on `teacher.native_trajectories`. These are the same objects
GRPO training consumes via `Trajectory.to_trl_format()`. Distillation and on-policy RL
training share the exact same rollout-collection code path (`create_rollout_fn`), they
just consume its output differently downstream (`EventLog.as_dataset_rows("sft")` for
distillation vs. the GRPO on-policy batch for RL). If you're building a pipeline that does
both, say, distill a student first, then RL-tune it, the trajectories from the teacher
rollout aren't reusable for the student's own RL training (they're the *teacher's*
behavior, not the student's), but knowing both paths pull from the same producer explains
why `collect_rollout`'s signature (`tools=`, `max_steps=`, `reward_fn=`, `system_prompt=`)
looks like a GRPO rollout config; it is one.

## Common pitfalls

- **Light-tier logs can't be trained on.** `Project.infer()`/`Project.collect()` (driven
  by a `Harness`, not a `RolloutEngine`) produce light-tier `EventLog`s: observational
  only. `distill()`/`train()` filter to `tier == "full"` and raise if none remain. Use
  `collect_rollout()` (a real `RolloutEngine`) or `add_trajectory(EventLog(tier="full"))`
  for anything you intend to train on.
- **Forgetting `trainer_factory`.** Both `train()` and `distill()` raise a `ValueError`
  immediately if `trainer_factory` is `None`; there's no silent no-op path.
- **Passing tools as bare dicts.** `collect_rollout(tools=...)` (and `distill`'s
  `rollout_kwargs={"tools": ...}`) want real objects with `.name`/`.to_schema()`, a
  `BaseTool` instance from `ToolRegistry.get(...)` or your own subclass. A bare
  `{"name": "my_tool"}` dict does not satisfy that interface.
- **Reaching for `agenttune.core.sft`.** It's broken (see the warning at the top of this
  page); every real example on this site wraps TRL's trainers directly instead.

## See it run for real

The corresponding [Local Notebook](../notebooks/local-notebook.md) runs this entire path for
real on a GPU: a teacher demonstrates an unusual output contract (`SENTIMENT=<label>`
instead of prose), `Project.distill` assembles the SFT dataset from full-tier
trajectories, and a real `trainer_factory` LoRA-fine-tunes
`HuggingFaceTB/SmolLM2-360M-Instruct` with TRL's `SFTTrainer`. It generates from the
student before and after training on the same probe prompt to show the weights actually
changed, not just that a loss number went down.

## Related reading

- [Basic Concepts](../getting-started/basic-concepts.md): `EventLog`, `Harness`,
  `Project`, and the rest of the spine's vocabulary.
- [Python API: Agentic Spine](agentic-spine.md): the standalone
  API-level reference for `Project` and everything it wraps.
- [Known Issues](../community/known-issues.md): the tool-registry import crash and
  other gaps worth knowing about.
