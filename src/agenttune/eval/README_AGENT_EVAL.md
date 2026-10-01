# AgentTune Evaluation Guide

How to run baseline and base model evals before and after training, across all use cases.

---

## Bug fix — what changed

**Problem:** `run_eval()` with any backend (including the default `"auto"`) crashed with:

```
TypeError: VLLMGeneration.__init__() got an unexpected keyword argument 'chat_template'
```

**Root cause:** `backend="auto"` inside `create_rollout_engine()` resolves to `"vllm"` (the fast training backend). `VLLMRolloutEngine` then calls `VLLMGeneration.__init__()` with four kwargs that were removed in newer TRL versions: `chat_template`, `chat_template_kwargs`, `tools`, `rollout_func`.

**Fix:** One line in `_build_rollout()`:

```python
# Before (broken):
engine = create_rollout_engine(backend=backend, model_path=model_path)

# After (fixed):
resolved = backend if backend == "api" else "transformers"
engine = create_rollout_engine(backend=resolved, model_path=model_path)
```

`"auto"` and `"vllm"` now both use `"transformers"` for eval. Only `"api"` passes through. This is correct — `VLLMRolloutEngine` requires a full training accelerator setup (colocate mode, distributed process group) which isn't appropriate for standalone eval. `TransformersRolloutEngine` loads the model directly with `AutoModelForCausalLM` and runs `model.generate()` — no setup needed.

---

## Setup

```bash
pip install -e .                    # AgentTune
pip install datasets transformers   # HuggingFace stack
```

---

## Baseline eval — base model (before training)

Run this **before** you train. Use the same val dataset you'll use after training.

### Email search

```python
from agenttune.eval import run_eval
from your_script import search_inbox, read_email, list_senders, val_dataset

report = run_eval(
    model_path="Qwen/Qwen3-0.6B",   # base model, not your trained checkpoint
    use_case="email_search",
    dataset=val_dataset,
    tools=[search_inbox, read_email, list_senders],
    max_samples=50,
)
report.save("./baselines")
```

### File ingestion

```python
from agenttune.eval import run_eval
from your_script import list_dir, read_file, run_python, write_file, val_dataset_raw
from datasets import Dataset

rows = []
for ex in val_dataset_raw:
    rows.append({
        "prompt": [
            {"role": "system", "content": f"You are a data analyst. Directory: {ex['sample_dir']}"},
            {"role": "user",   "content": ex["prompt"][1]["content"]},
        ],
        "answer": ex["answer"],
    })
val_dataset = Dataset.from_list(rows)

report = run_eval(
    model_path="Qwen/Qwen3-0.6B",
    use_case="file_ingestion",
    dataset=val_dataset,
    tools=[list_dir, read_file, run_python, write_file],
    max_samples=30,
)
report.save("./baselines")
```

### FinQA

```python
from agenttune.eval import run_eval
from your_script import get_template, get_explanation, get_tables, run_python
from agenttune.agentic.tools.builtin.finqa_tool import RLLMFinQATool
from datasets import Dataset

finqa_tool = RLLMFinQATool()
finqa_tool.load(split="test", max_samples=50)

rows = []
for eid in finqa_tool.ids():
    question, answer = finqa_tool.qa(eid)
    rows.append({
        "prompt": [{"role": "user", "content": f"Example ID: {eid}\nQuestion: {question}"}],
        "answer": answer,
    })
val_dataset = Dataset.from_list(rows)

report = run_eval(
    model_path="Qwen/Qwen3-0.6B",
    use_case="finqa",
    dataset=val_dataset,
    tools=[get_template, get_explanation, get_tables, run_python],
    max_samples=50,
)
report.save("./baselines")
```

---

## Post-training eval — trained model

Same code, just swap `model_path` to your checkpoint:

```python
report = run_eval(
    model_path="./output/email_search_agent",
    use_case="email_search",
    dataset=val_dataset,
    tools=[search_inbox, read_email, list_senders],
)
report.save("./results")
```

---

## Comparing baseline vs trained

```python
import json
from pathlib import Path

def load_report(path):
    return json.loads(Path(path).read_text())

baseline = load_report("./baselines/email_search_20250101_120000.json")
trained  = load_report("./results/email_search_20250101_140000.json")

print(f"{'Metric':<28} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
print("-" * 62)
for metric in baseline["means"]:
    b = baseline["means"].get(metric, 0)
    t = trained["means"].get(metric, 0)
    delta = t - b
    arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
    print(f"{metric:<28} {b:>10.3f} {t:>10.3f}  {arrow} {delta:+.3f}")

print()
print(f"Pass rate:  {baseline['pass_rate']*100:.1f}%  →  {trained['pass_rate']*100:.1f}%")
```

---

## Re-scoring saved completions (no GPU needed)

```python
completions = [
    "<answer>Q3 revenue was $4.2B</answer>",
    "<answer>The email was sent on March 12</answer>",
]

report = run_eval(
    model_path="./output/my_agent",
    use_case="email_search",
    dataset=val_dataset,
    completions=completions,
)
```

---

## API backend (no local GPU)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

report = run_eval(
    model_path="gpt-4o-mini",
    use_case="email_search",
    dataset=val_dataset,
    tools=[search_inbox, read_email],
    backend="api",
)
```

Supported providers via LiteLLM: `"gpt-4o-mini"`, `"claude-sonnet-4-5"`,
`"groq/llama-3.3-70b-versatile"`, `"openrouter/..."`, `"ollama/llama3"`, etc.

---

## PEFT / LoRA eval

Train with a LoRA adapter and eval the base + adapter together:

```python
from agenttune.eval import run_eval
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine, create_rollout_fn

# Load base + adapter manually, pass the engine in
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "./output/lora_checkpoint")
model.eval()

engine = create_rollout_engine(
    backend="transformers", model=model, tokenizer=tokenizer
)
rollout_fn = create_rollout_fn(
    rollout_engine=engine,
    tools=[search_inbox, read_email],
    max_steps=10,
)

# Pass pre-built rollout_fn by running _run_one manually, or just use
# completions= to score outputs you already have
```

---

## Custom / future tasks

No preset needed — pass metrics inline:

```python
from agenttune.eval import run_eval
from agenttune.eval.agent_eval import token_f1, tool_use, answer_format, metric

report = run_eval(
    model_path="Qwen/Qwen3-0.6B",
    use_case="my_new_task",
    dataset=my_val_dataset,
    tools=[my_tool_a, my_tool_b],
    metrics=[
        token_f1(),
        tool_use(),
        answer_format(),
        metric("has_json", lambda r: 1.0 if r.predicted.strip().startswith("{") else 0.0),
    ],
)
```

Or register a preset once and reuse it anywhere:

```python
from agenttune.eval.agent_eval import PRESETS, token_f1, tool_use, specific_tools

PRESETS["code_gen"] = [
    token_f1(),
    tool_use(),
    specific_tools(["run_python", "read_file"], "code_tools"),
]

run_eval(..., use_case="code_gen")
```

---

## Recommended workflow

```
1. Build val dataset  (same split, kept fixed throughout)
2. Baseline:   run_eval(model_path="Qwen/...", ...)   → save to ./baselines/
3. Train:      trainer.train()
4. Eval:       run_eval(model_path="./output/...", ...) → save to ./results/
5. Compare:    load both JSONs, diff the means dict
```

Keep `max_samples` the same between baseline and post-training runs.
50 samples is usually enough to see a clear trend.

---

## All `run_eval()` parameters

| Parameter | Default | What it does |
|---|---|---|
| `model_path` | required | HF model ID or path to trained checkpoint |
| `dataset` | required | HF Dataset or list of dicts with `prompt` + `answer` |
| `use_case` | `"generic"` | Preset: `email_search`, `file_ingestion`, `finqa`, `generic` |
| `tools` | `[]` | Tool functions the agent can call |
| `metrics` | from preset | Override with your own list of metric objects |
| `backend` | `"auto"` | `"auto"` / `"transformers"` use local model. `"api"` uses LiteLLM |
| `max_steps` | `10` | Max tool calls per sample |
| `max_samples` | `None` | Cap dataset size |
| `completions` | `None` | Pre-computed predictions — skips inference |
| `verbose` | `True` | Print per-sample results |
