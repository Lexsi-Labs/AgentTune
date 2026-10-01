# Backend Selection

AgentTune's RL trainers are built on **TRL**. `create_agentic_trainer` uses TRL by
default, and that's the supported, tested (real GPU test coverage), documented path.

```python
from agenttune.core.backend_factory import create_agentic_trainer
trainer = create_agentic_trainer("grpo", model=..., ...)   # TRL by default
```

`AgenticAlgorithm` (the enum this function validates against) maps to TRL-backed
classes: `backends/trl/agentic/{grpo,dpo,ppo,rloo,bco}/agentic_*.py`. See
[User Guide: RL Training](../user-guide/rl-training.md).

## Unsloth (optional, unofficial)

`create_agentic_trainer` also accepts `backend="unsloth"`, or `backend="auto"` to use
Unsloth when it's importable and fall back to TRL otherwise. Unsloth is **not** a
declared dependency of AgentTune. Install it yourself if you want it:

```bash
pip install unsloth==2026.8.19 --no-deps
pip install "unsloth_zoo>=2026.8.13" --no-deps
```

`--no-deps` skips Unsloth's own dependency pins (torch/transformers/trl/peft/xformers),
so it won't upgrade or downgrade the versions already in your environment; it installs
against whatever compatible stack you already have. `unsloth_zoo` doesn't share
Unsloth's version numbers. The `>=2026.8.13` floor above is what `unsloth==2026.8.19`
itself declares as its minimum compatible `unsloth_zoo`, not a matching pin.

It isn't benchmarked. Treat it as a self-service option rather than a fully supported
backend, though `tests/unsloth/` covers all five algorithms with an opt-in test suite,
run separately from the main suite:

```bash
RUN_UNSLOTH_TESTS=1 pytest tests/unsloth -q
```

TRL stays unaffected as long as Unsloth is never imported: `create_agentic_trainer`
only imports it lazily, inside the code path you actually requested.

```python
trainer = create_agentic_trainer("grpo", backend="unsloth", model=..., ...)
trainer = create_agentic_trainer("grpo", backend="auto", model=..., ...)  # unsloth if available, else TRL
```

**Once Unsloth is imported, that guarantee no longer holds for the rest of the process.**
Unsloth patches `transformers`' model classes at the class level, not per instance. Any
`backend="unsloth"`/`"auto"` call that actually uses Unsloth patches those classes
globally, so every later plain (non-Unsloth) model load in the same process breaks with
an `AttributeError` on the model
(`'Qwen2RotaryEmbedding' object has no attribute 'extend_rope_embedding'`,
`'Qwen2Attention' object has no attribute 'apply_qkv'`, and similar), even for code that
never asked for Unsloth. That's why `tests/unsloth/` runs as its own `pytest`
invocation instead of joining the main suite. If your own process mixes Unsloth
training with plain TRL or transformers usage, keep them in separate processes, or set
`PURE_TRL_MODE=1` to guarantee `import unsloth` never runs in the process that needs to
stay plain.

## API-based backends (no local GPU at all)

Both the rollout engine (`create_rollout_engine(backend="api", ...)`) and the LLM judge
(`LLMJudge(model=..., api_key=...)`) support calling a hosted model via `litellm` instead
of loading weights locally, useful for judging/scoring without a GPU, or for building a
DECIDE pipeline that never touches local hardware. See
[User Guide: RL Training](../user-guide/rl-training.md#rollout-engines-standalone) for the exact
parameters.
