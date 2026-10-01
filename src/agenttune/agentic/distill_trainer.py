"""
create_distill_trainer — one-call agentic distillation.

Every distillation example/notebook in this repo (``examples/agentic_distillation_real.py``,
``docs/user-guide/distillation.md``, ``temp/distill.ipynb``) hand-writes the same
``trainer_factory`` class — load the student, build an ``SFTConfig``, optionally wrap it
in a LoRA ``peft_config``, return a TRL ``SFTTrainer`` — before it ever reaches
``Project.distill()``. ``create_distill_trainer`` builds that factory for you (still
overridable via `trainer_factory=`) and wraps trajectory collection + `Project.distill()`
behind a `create_agentic_trainer`-style object: construct it with plain kwargs, then
call `.train()`.

    trainer = create_distill_trainer(
        student="HuggingFaceTB/SmolLM2-360M-Instruct",
        demonstrations=[
            ("Classify sentiment: 'best purchase ever'", "SENTIMENT=positive"),
            ("Classify sentiment: 'broke on day one'", "SENTIMENT=negative"),
        ],
        output_dir="./out_distill",
        num_train_epochs=25,
        learning_rate=3e-4,
    )
    result = trainer.train()

Or, distilling from a live teacher model instead of hand-authored demonstrations:

    trainer = create_distill_trainer(
        student="Qwen/Qwen2.5-0.5B-Instruct",
        teacher_engine=create_rollout_engine(backend="transformers", model_path="Qwen/Qwen2.5-7B-Instruct"),
        tasks=["Read config.yaml and report default_model.", "..."],
        rollout_kwargs={"tools": [my_tool], "max_steps": 6},
        output_dir="./out_distill",
    )
    trainer.train()
"""

from __future__ import annotations

from typing import Any

from .events import Event, EventKind, EventLog
from .project import Project

_DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _demo_trajectory(prompt: str, response: str) -> EventLog:
    """One full-tier hand-authored teacher trajectory: a request and its demonstrated
    answer — the pattern documented in docs/user-guide/distillation.md §1 Option A."""
    log = EventLog(tier="full")
    log.append(Event(EventKind.OBSERVATION, {"text": prompt}))
    log.append(Event(EventKind.TEXT, {"text": response}))
    return log


def _generate(model, tok, prompt: str, max_new_tokens: int = 32) -> str:
    import torch

    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def _merge_consecutive_same_role(messages: list[dict]) -> list[dict]:
    """Merge adjacent same-role messages into one, joined by a blank line.

    EventLog.as_dataset_rows emits one "assistant" message per REASONING/TEXT
    event and another per TOOL_CALL -- a "thought, then act" step legitimately
    produces two assistant turns back to back with nothing in between. Combined
    with fold_tool_messages_into_user (which can leave an existing "user" turn
    adjacent to a freshly-folded one), a strict-alternation template (e.g.
    CohereLabs/tiny-aya-fire's) rejects the sequence outright. A no-op when
    roles already alternate, so it's safe to apply unconditionally in the
    fallback path.
    """
    merged: list[dict] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg.get("role"):
            merged[-1] = {
                **merged[-1],
                "content": f"{merged[-1].get('content', '')}\n\n{msg.get('content', '')}",
            }
        else:
            merged.append(dict(msg))
    return merged


class _DefaultSFTTrainerFactory:
    """The built-in `trainer_factory`: loads `model`, optionally wraps it in LoRA, and
    returns a TRL `SFTTrainer`. Everything in `**kwargs` at call time (num_train_epochs,
    learning_rate, output_dir, ...) goes straight to `SFTConfig` — the same shape
    `Project.distill(..., **trainer_kwargs)` already forwards.

    Mirrors `RealSFTTrainerFactory` from docs/user-guide/distillation.md, just
    parameterized instead of hand-written per call site.
    """

    def __init__(
        self,
        *,
        use_peft: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_targets: list[str] | None = None,
        task_type: str = "CAUSAL_LM",
        dtype: Any = None,
        device_map: Any = "auto",
        probe_prompt: str | None = None,
        trust_remote_code: bool = False,
    ):
        self.use_peft = use_peft
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.lora_targets = lora_targets
        self.task_type = task_type
        self.dtype = dtype
        self.device_map = device_map
        self.probe_prompt = probe_prompt
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self.sft_trainer = None
        self.before = None

    def __call__(self, *, model, train_dataset, **kwargs):
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = self.dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )

        if self.probe_prompt:
            self.before = _generate(self.model, self.tokenizer, self.probe_prompt)

        cfg_kwargs = dict(kwargs)
        cfg_kwargs.setdefault("output_dir", "./out_distill")
        cfg = SFTConfig(**cfg_kwargs)

        peft_config = None
        if self.use_peft:
            from peft import LoraConfig

            peft_kwargs = {
                "r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "bias": self.lora_bias,
                "task_type": self.task_type,
            }
            peft_kwargs["target_modules"] = self.lora_targets or _DEFAULT_LORA_TARGETS
            peft_config = LoraConfig(**peft_kwargs)

        # TRL's SFTTrainer tokenizes "messages" through the student's raw chat
        # template with no repair ladder at all (unlike AgentTune's own
        # rollout-generation code) -- a template that hard-requires strict
        # user/assistant alternation (e.g. CohereLabs/tiny-aya-fire's) rejects a
        # "tool"-role message a real teacher trajectory can carry (TOOL_RESULT
        # events become role="tool") or two "assistant" turns in a row (a
        # thought-then-act step, with nothing in between). Preflight-render each
        # row and repair (the same tool-message fold GRPO/RLOO's rollout path
        # already falls back to, plus merging any remaining same-role run)
        # BEFORE constructing SFTTrainer -- constructing it twice on a caught
        # failure would re-apply `peft_config` to the already-PEFT-wrapped
        # `self.model` from the first attempt, corrupting it with stacked
        # adapters, so this must be decided up front, not via try/except retry.
        rows = train_dataset
        needs_repair = False
        for row in rows:
            try:
                self.tokenizer.apply_chat_template(row["messages"], tokenize=False)
            except Exception:
                needs_repair = True
                break
        if needs_repair:
            from agenttune.agentic.rollout_engines.rollout_factory import (
                fold_tool_messages_into_user,
            )

            rows = [
                {
                    **row,
                    "messages": _merge_consecutive_same_role(
                        fold_tool_messages_into_user(row["messages"])
                    ),
                }
                for row in train_dataset
            ]

        self.sft_trainer = SFTTrainer(
            model=self.model,
            args=cfg,
            train_dataset=Dataset.from_list(rows),
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )
        return self.sft_trainer


class DistillTrainer:
    """Returned by `create_distill_trainer`. Holds everything needed to run
    `Project.distill(...)`, deferred until `.train()` is called — so construction
    stays cheap (no GPU/model load) even when a live `teacher_engine` rollout is
    involved, matching `create_agentic_trainer`'s "build, then `.train()`" shape."""

    def __init__(
        self,
        *,
        project: Project,
        student: str,
        trainer_factory: Any,
        fmt: str,
        teacher_engine: Any,
        tasks: list[str] | None,
        rollout_kwargs: dict | None,
        trainer_kwargs: dict,
    ):
        self.project = project
        self.student = student
        self.trainer_factory = trainer_factory
        self.fmt = fmt
        self.teacher_engine = teacher_engine
        self.tasks = tasks
        self.rollout_kwargs = rollout_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.result = None

    @property
    def trajectories(self) -> list[EventLog]:
        return self.project.trajectories

    def sft_dataset(self) -> list[dict]:
        """GPU-free preview of the rows `.train()` would feed the trainer_factory."""
        return self.project.sft_dataset(self.fmt)

    def events(self):
        return self.project.events()

    def train(self):
        kwargs = dict(self.trainer_kwargs)
        if self.teacher_engine is not None:
            kwargs["teacher_engine"] = self.teacher_engine
            kwargs["tasks"] = self.tasks
            if self.rollout_kwargs:
                kwargs["rollout_kwargs"] = self.rollout_kwargs
        self.result = self.project.distill(
            self.student, trainer_factory=self.trainer_factory, fmt=self.fmt, **kwargs
        )
        return self.result


def create_distill_trainer(
    student: str,
    *,
    demonstrations: list[tuple[str, str]] | list[dict] | None = None,
    trajectories: list[EventLog] | None = None,
    teacher_engine: Any = None,
    tasks: list[str] | None = None,
    rollout_kwargs: dict | None = None,
    tools: list[Any] | None = None,
    reward_fn: Any = None,
    system_prompt: str | None = None,
    max_steps_per_turn: int = 8,
    fmt: str = "sft",
    trainer_factory: Any = None,
    use_peft: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
    lora_targets: list[str] | None = None,
    task_type: str = "CAUSAL_LM",
    dtype: Any = None,
    device_map: Any = "auto",
    probe_prompt: str | None = None,
    trust_remote_code: bool = False,
    output_dir: str = "./out_distill",
    **sft_kwargs: Any,
) -> DistillTrainer:
    """
    Build an agentic-distillation trainer in one call.

    Wraps `Project` + `Project.distill()`: collects teacher trajectories (one of
    `demonstrations=`, `trajectories=`, or `teacher_engine=`+`tasks=`) and hands them,
    plus a built-in TRL `SFTTrainer` `trainer_factory` (LoRA-wrapped by default), to
    `Project.distill()` when you call `.train()`.

    Parameters
    ----------
    student : HF repo id or local path for the student model.
    demonstrations : list[(prompt, response)] | list[{"prompt"/"question", "response"/"answer"}]
        Hand-authored teacher demonstrations (docs' "Option A") — added eagerly as
        full-tier trajectories at construction time. No GPU needed.
    trajectories : pre-built full-tier `EventLog`s (e.g. from your own
        `collect_rollout`/`Project` elsewhere) — added eagerly, as-is.
    teacher_engine / tasks : a live `RolloutEngine` + task list (docs' "Option B/C") —
        rollout is collected lazily, inside `.train()`, not at construction time.
    rollout_kwargs : forwarded to the internal `collect_rollout` call when
        `teacher_engine`+`tasks` are used — put `tools=`/`max_steps=`/`system_prompt=`
        for an agentic (tool-using) teacher here, or use the top-level `tools=`/
        `system_prompt=`/`max_steps_per_turn=` shortcuts below instead.
    tools / reward_fn / system_prompt / max_steps_per_turn : convenience aliases merged
        into `rollout_kwargs` (`tools`->`tools` — any custom `BaseTool` instance(s),
        `reward_fn`->`reward_fn`, `system_prompt`->`system_prompt`,
        `max_steps_per_turn`->`max_steps`). Use these for an agentic (tool-using)
        teacher; pass your own callable(s) here, nothing is restricted to built-ins.
    fmt : dataset format `as_dataset_rows` produces; only `"sft"` is supported today
        (same restriction as `Project.distill`).
    trainer_factory : override the built-in TRL SFT factory entirely with your own
        callable (`model=, train_dataset=, **kwargs -> object with .train()`).
    use_peft / lora_r / lora_alpha / lora_dropout / lora_bias / lora_targets / task_type :
        LoRA knobs for the built-in factory (ignored if `trainer_factory` is given).
    dtype / device_map : model-loading knobs for the built-in factory.
    probe_prompt : if given, the built-in factory generates from the student on this
        prompt before training (`trainer.trainer_factory.before`) so you can compare
        against a post-training generation yourself.
    output_dir : passed through to `SFTConfig` (also the default push/export path).
    **sft_kwargs : forwarded to `Project.distill(...)` -> the trainer_factory's
        `**kwargs` -> `SFTConfig(**kwargs)` (num_train_epochs, learning_rate,
        per_device_train_batch_size, save_strategy, report_to, bf16, ...).

    Returns
    -------
    DistillTrainer — call `.train()` to run collection (if deferred) + distillation.
    `.sft_dataset()` previews the training rows GPU-free; `.trajectories`/`.events()`
    expose the underlying `Project`'s state.
    """
    if demonstrations is None and trajectories is None and teacher_engine is None:
        raise ValueError(
            "create_distill_trainer needs one of `demonstrations=`, `trajectories=`, "
            "or `teacher_engine=`+`tasks=` to source teacher trajectories from."
        )
    if teacher_engine is not None and not tasks:
        raise ValueError("create_distill_trainer(teacher_engine=...) needs `tasks=` too.")

    project = Project()

    if demonstrations:
        for demo in demonstrations:
            if isinstance(demo, dict):
                prompt = demo.get("prompt", demo.get("question"))
                response = demo.get("response", demo.get("answer"))
            else:
                prompt, response = demo
            project.add_trajectory(_demo_trajectory(prompt, response))

    if trajectories:
        for log in trajectories:
            project.add_trajectory(log)

    merged_rollout_kwargs = dict(rollout_kwargs or {})
    if tools is not None:
        merged_rollout_kwargs.setdefault("tools", tools)
    if reward_fn is not None:
        merged_rollout_kwargs.setdefault("reward_fn", reward_fn)
    if system_prompt is not None:
        merged_rollout_kwargs.setdefault("system_prompt", system_prompt)
    merged_rollout_kwargs.setdefault("max_steps", max_steps_per_turn)

    factory = trainer_factory or _DefaultSFTTrainerFactory(
        use_peft=use_peft,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
        lora_targets=lora_targets,
        task_type=task_type,
        dtype=dtype,
        device_map=device_map,
        probe_prompt=probe_prompt,
        trust_remote_code=trust_remote_code,
    )

    trainer_kwargs = dict(sft_kwargs)
    trainer_kwargs["output_dir"] = output_dir

    return DistillTrainer(
        project=project,
        student=student,
        trainer_factory=factory,
        fmt=fmt,
        teacher_engine=teacher_engine,
        tasks=tasks,
        rollout_kwargs=merged_rollout_kwargs if teacher_engine is not None else None,
        trainer_kwargs=trainer_kwargs,
    )
