# Evaluation

`agenttune.eval` is really **two unrelated evaluation systems** that happen to share a
package. Confusing one for the other is the most common way to waste time here, so read
this page as two halves rather than one API.

- **System A, the "Universal" framework.** `BaseEvaluator` / `RLEvaluator` plus a
  metrics library, exported from `agenttune.eval`. This is the one thing in `eval/` that
  training code actually calls: `core/sft/trainer_base.py` invokes `BaseEvaluator` when
  you opt in with `use_custom_evaluator=True`.
- **System B, the standalone tool-use evaluator.** `eval/agent_eval.py`'s `run_eval`.
  Genuinely useful, tool-use-scoring oriented, fully real, and **not exported from
  `agenttune.eval`, with zero callers anywhere else in the repo.** You have to import it
  by module path. Nobody in this codebase currently wires it into anything; that doesn't
  make it broken, just orphaned.

Everything on this page was run against the current source while writing it; every
snippet below is copy-pasteable and actually executes (the two that need a real model
download `sshleifer/tiny-gpt2`, a few hundred KB, to stay runnable on CPU with no auth).
For the condensed fact table (ratings, exact gaps) see
[Python API: Evaluation](evaluation.md); this page is the "how do
I actually call this" version with full working code.

## Quick orientation

| What you want | Use this | Needs |
|---|---|---|
| Score a SFT/generation model on ROUGE/BLEU/perplexity | `BaseEvaluator` | `torch` + a real model/tokenizer (CPU-OK) |
| Score an RL/DPO policy (KL, entropy, win-rate, …) | `RLEvaluator` | same, plus optionally a reference model |
| Score a tool-using agent's transcripts (email/finance/file use cases) | `eval.agent_eval.run_eval` | nothing extra if you pass `completions=` |
| Execute + test-check generated code in a sandbox | `eval.safe_executor.execute_code_safely` | nothing, pure stdlib |
| Score a hand-built agent trajectory (tool args, loops, redundancy) | `TrajectoryEvaluator`'s `_calculate_*` methods | `jsonschema` (already a dependency) |
| Run a standard benchmark (MMLU, GSM8K, HumanEval, …) | `LMEvalRunner` | the `lm-eval` package + a real model |

## 1. `BaseEvaluator`: the Universal framework

`BaseEvaluator` (`src/agenttune/eval/evaluator.py`) drives a generation loop over a
dataset and computes whichever `Metric` objects you give it. It auto-detects prompt and
target columns via the same column-heuristics table used by the SFT/GRPO data schemas
(`agenttune.data.schemas.TASK_SCHEMAS`), plus a handful of generic fallbacks
(`prompt`/`input`/`question`/`instruction` for inputs, `completion`/`response`/`answer`/
`target`/... for targets), so a plain `{"prompt": ..., "completion": ...}` dataset works
without any column-mapping.

### Worked example: ROUGE/BLEU against a real dataset

This uses `sshleifer/tiny-gpt2` (a ~1MB test model on the Hub) purely so the example runs
end-to-end on CPU with no GPU and no real model weights to babysit. Swap in your actual
fine-tuned checkpoint for real use.

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from agenttune.eval import BaseEvaluator
from agenttune.eval.metrics import RougeMetric, BleuMetric

model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# A real-shaped SFT-style dataset. Any of `prompt`/`input`/`question`/`instruction`
# and `completion`/`response`/`answer`/`target` are auto-detected.
dataset = Dataset.from_list([
    {"prompt": "The capital of France is", "completion": "Paris."},
    {"prompt": "Water boils at", "completion": "100 degrees Celsius."},
    {"prompt": "The sun rises in the", "completion": "east."},
])

evaluator = BaseEvaluator(
    metrics=[RougeMetric(), BleuMetric()],
    batch_size=2,
    device="cpu",
    use_cache=False,
    generation_kwargs={"max_new_tokens": 8},
)
results = evaluator.evaluate(model=model, tokenizer=tokenizer, dataset=dataset)
print(results)
# {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rouge_1': 0.0, 'rouge_2': 0.0,
#  'rouge_l': 0.0, 'rouge': 0.0, 'bleu': 0.0, 'total': 3}
```

The zeroes above are `tiny-gpt2` being a randomly-behaving test model, not a bug. Point
this at a real fine-tuned checkpoint and the scores move. Two things worth knowing before
you rely on this in a script:

- **`evaluate()` prints a wall of `DEBUG:` lines unconditionally.** It's not gated behind
  a log level, it's plain `print()` calls left in from development. Harmless, just noisy;
  redirect stdout if it bothers you.
- **`RougeMetric`/`BleuMetric` silently return all-zero scores if `rouge_score`/`nltk`
  aren't installed** (`pip install rouge_score nltk`), no exception, no warning banner,
  just a quiet 0.0 that looks like a bad model. See
  [Known Issues](../community/known-issues.md).

If you omit `metrics=`, `BaseEvaluator` picks defaults from `task_type`: `"code"` →
`PassAtKMetric`, `"math"` → `MathAccuracyMetric`, `"text"`/`"generation"` → Perplexity +
ROUGE + BLEU, anything else → Perplexity + Accuracy.

```python
evaluator = BaseEvaluator(task_type="math")       # -> [MathAccuracyMetric()]
evaluator = BaseEvaluator(task_type="code", k_list=[1, 10])  # -> [PassAtKMetric(k_list=[1, 10])]
```

`MathAccuracyMetric` and `PassAtKMetric` are both standalone-callable too; you don't need
the full generation loop to use them:

```python
from agenttune.eval.metrics import MathAccuracyMetric

metric = MathAccuracyMetric()
metric.compute(
    predictions=["The answer is \\boxed{42}", "I think it's 17"],
    references=["42", "17"],
)
```

`PassAtKMetric` (`src/agenttune/eval/metrics/code.py`) is worth calling out separately: it
doesn't do string comparison at all; it drives the real `execute_code_safely` sandbox
under the hood (see [§3](#3-sandboxed-code-execution) below) to actually *run* each
candidate against its test cases before computing Pass@K/Avg@K/Maj@K.

### Writing your own metric

Every metric (the ones above, `run_eval`'s tool-use metrics in §2, everything) is just
a `Metric` subclass with a `compute(predictions, references, **kwargs) -> Dict[str,
float]` method. `safe_compute()` (inherited, don't override it) wraps `compute()` in a
`try/except` so one bad batch can't crash the whole evaluation run: a failure becomes
`{"<name>_error": 0.0}` in the results dict instead of an exception. This is real,
verified end to end:

```python
from agenttune.eval.metrics.base import Metric

class ExactLengthMatch(Metric):
    """1.0 if prediction and reference have the same word count (+/- tolerance)."""

    def __init__(self, tolerance: int = 2):
        super().__init__("exact_length_match")
        self.tolerance = tolerance

    @property
    def requires_generation(self) -> bool:
        return True   # needs generated text, not just precomputed logits/losses

    def compute(self, predictions, references, **kwargs):
        hits = sum(
            1 for p, r in zip(predictions, references)
            if abs(len(p.split()) - len(r.split())) <= self.tolerance
        )
        return {"exact_length_match": hits / max(len(predictions), 1)}

metric = ExactLengthMatch()
metric.safe_compute(["a b c"], ["a b c d"])
# {'exact_length_match': 1.0}
```

Pass an instance of your subclass anywhere `metrics=[...]` is accepted;
`BaseEvaluator(metrics=[ExactLengthMatch(), RougeMetric()])` works exactly like it does
for the built-in metrics, since `evaluate()`/`evaluate_rl()` only ever call
`metric.safe_compute(...)` and read `metric.name`/`metric.requires_generation`; there's
no registry to update, no base-class method to satisfy beyond `compute()`.

### `RLEvaluator`: the RL/DPO subclass

`RLEvaluator(BaseEvaluator)` (`src/agenttune/eval/rl_evaluator.py`) adds a specialized
`evaluate_rl()` loop with KL-divergence-vs-a-reference-model, policy entropy, and
reward-model or **implicit-reward** (DPO, no reward model needed; it scores
`beta * (policy_logprob - reference_logprob)` on chosen vs. rejected) computation. If you
don't pass `metrics=`, it defaults to `[KLDivergenceMetric(), RewardAccuracyMetric(),
PolicyEntropyMetric()]` on top of whatever `BaseEvaluator` would have picked.

```python
from agenttune.eval import RLEvaluator

evaluator = RLEvaluator(batch_size=4, device="cpu")
results = evaluator.evaluate_rl(
    policy_model=policy,           # a real causal LM
    reference_model=reference,     # optional, enables KL + implicit-reward scoring
    tokenizer=tokenizer,
    dataset=dpo_dataset,            # needs 'chosen'/'rejected' columns for DPO metrics
)
```

DPO-specific metrics (`WinRateMetric`, `RewardMarginMetric`, `PreferenceAccuracyMetric`,
`LogRatioMetric`, `ImplicitRewardMetric`, `CalibrationMetric`) live in
`agenttune.eval.metrics.dpo` and are **not re-exported from `agenttune.eval.metrics`**;
import them from the submodule directly:

```python
from agenttune.eval.metrics.dpo import WinRateMetric, RewardMarginMetric

evaluator = RLEvaluator(metrics=[WinRateMetric(), RewardMarginMetric()], device="cpu")
evaluator.evaluate_rl(policy, reference, tokenizer, dpo_dataset)
```

They're also directly callable on precomputed `(chosen_score, rejected_score)` pairs, no
model or dataset needed:

```python
from agenttune.eval.metrics.dpo import WinRateMetric

WinRateMetric().compute(predictions=[(0.8, 0.3), (0.2, 0.5)], references=[])
# {'win_rate': 0.5, 'win_count': 1, 'total_pairs': 2}
```

### The two legacy/production entry points you'll see referenced elsewhere

Two more real, working pieces sit alongside `BaseEvaluator` for backward compatibility.
Neither is part of the "worked examples" above because both need a real model load
(GPU-oriented, not something a doc snippet should pretend to run), but they're worth
knowing about:

- **`eval/runner.py`'s `run_eval(config: EvalConfig, dataset_dict=None)`**: the
  production CLI-facing entry point: loads a model (supports plain HF and vLLM backends,
  plus LoRA), dispatches into `RLEvaluator`, and writes `eval_results.json`.
  `EvalConfig.metrics` here is a **list of strings**, not metric instances, e.g.
  `metrics=["rouge", "win_rate", "pass_at_k"]`, mapped to real `Metric` classes by
  `get_metrics_from_names()`, which is also where DPO metrics (`"win_rate"`,
  `"reward_margin"`, ...) actually get wired in for this entry point.
- **`eval/core.py`'s legacy `EvalRunner`/`EvalTask`/`EvalConfig`/`EvalRegistry`**: an
  older, still-functional framework where you register tasks/metrics/datasets yourself.
  Its `EvalConfig.metrics` is also a list of strings, but a different default list and a
  different registry than `eval/runner.py`'s. **Careful:** there's a *second*, unrelated
  class also named `EvalRegistry` in `eval/registry.py` that adapts the metric classes to
  an older calling convention; it is not the one exported from `agenttune.eval` (that's
  `core.py`'s). Don't confuse the two if you go digging in the source.

## 2. `run_eval`: the standalone tool-use agent evaluator

`eval/agent_eval.py` is a self-contained module built for scoring agents on three
real dataset shapes (email search over Enron-style inboxes, FinQA-style financial
templates, file-ingestion tasks) plus a `"generic"` fallback that works for anything. It
is **not part of System A**: different file, different author, different metric
classes, no shared code, and as noted above, nothing in this repo currently imports it
except you.

### The zero-GPU path: `completions=`

The single most useful thing about `run_eval` is that it has a path that needs **no
model, no GPU, no rollout engine at all**: pass pre-generated completion strings and it
skips model loading entirely, going straight to scoring:

```python
from datasets import Dataset
from agenttune.eval.agent_eval import run_eval

val_dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": "What is the confirmation number in the invoice email?"}],
     "answer": "251832", "message_ids": [["m-101"]]},
    {"prompt": [{"role": "user", "content": "Who sent the March report?"}],
     "answer": "Petrous LLC", "message_ids": [["m-202"]]},
])

report = run_eval(
    model_path="unused-with-completions",   # only used for the printed banner
    dataset=val_dataset,
    use_case="email_search",
    completions=[
        "The confirmation number is 251832.",
        "It was sent by Petrous LLC.",
    ],
)
print(report.pass_rate)   # 1.0
print(report.means)       # per-metric averages, e.g. {'token_f1': 0.65, 'universal_score': 0.8, ...}
report.save("./eval_reports")   # writes a timestamped JSON report
```

`report` is a real `Report` dataclass with `.pass_rate`, `.means`/`.stds` per metric, a
pretty `__str__()` box-drawing summary, `.failures(top_n=10)` (worst-scoring samples), and
`.save(directory)`.

### The model-driving path: `model_path=`

Without `completions=`, `run_eval` builds a real rollout function via
`agentic.rollout_engines.rollout_factory.create_rollout_engine`/`create_rollout_fn` and
actually runs the agent (tools included) against every row:

```python
from agenttune.eval.agent_eval import run_eval

report = run_eval(
    model_path="Qwen/Qwen3-0.6B",
    use_case="email_search",
    dataset=val_dataset,
    tools=[search_inbox, read_email, list_senders],   # plain callables, or BaseTool instances
    max_steps=10,
    max_samples=50,
)
```

This path needs the model to actually load and generate (GPU strongly recommended for
anything beyond a tiny model); it's real, but not something to run inside a doc example.

### Built-in use cases and their metrics

`PRESETS` (a plain dict in `agent_eval.py`) maps `use_case` to a metric list:

| `use_case` | Primary metric | Also computes |
|---|---|---|
| `"email_search"` | `enron_answer_similarity` | `token_f1`, `exact_match_normalized`, `substring_match`, `message_id_recall`, `answer_format`, `tool_use`, tool-presence checks, `universal_score` |
| `"finqa"` | `finqa_numeric_exact` | `numerical_recall`, `unfilled_slot_penalty`, `section_header_coverage`, `table_row_coverage`, `length_ratio_score`, ... |
| `"file_ingestion"` | `numerical_accuracy_file` | `token_f1`, `tool_chain_order`, `summary_written`, ... |
| `"generic"` | `universal_score` | `token_f1`, `exact_match_normalized`, `answer_format`, `tool_use` |

Every metric in that table is a standalone, real function you can call directly on any
`(prediction, gold)` pair without going through `run_eval` at all; they're built with
`_Metric(name, fn)` and just wrap a plain callable:

```python
from agenttune.eval.agent_eval import token_f1, universal_score, tool_use, Sample

s = Sample(idx=0, question="Who sent the report?", gold="Petrous LLC",
           predicted="<answer>It was Petrous LLC.</answer>", n_tools=2)

print(token_f1()(s))          # 1.0 — token overlap after stopword removal
print(universal_score()(s))   # blended prose/numeric/code score
print(tool_use()(s))           # 0.7 — tiered score for n_tools=2 (0->0.0, 1->0.4, 2->0.7, 3+->1.0)
```

`universal_score()` is the most broadly useful one: it auto-detects whether the gold
answer looks numeric, code-like, or prose, and blends the matching sub-scorer
(token-F1 always active; a 2%-tolerance numeric-recall check activated when the gold has
numbers; a keyword+token-overlap code check activated when the gold has ≥3 code
keywords), reasonable as a generic answer-quality metric for `use_case="generic"`.

You can also pass a fully custom metric list instead of a preset, or write your own with
`metric(name, fn)`:

```python
from agenttune.eval.agent_eval import metric, specific_tools

my_metrics = [
    specific_tools(["search_inbox"], "used_search"),
    metric("has_dollar_sign", lambda s: 1.0 if "$" in s.predicted else 0.0),
]
report = run_eval(model_path="...", dataset=val_dataset, metrics=my_metrics, completions=[...])
```

Its own bundled `README_AGENT_EVAL.md` has a broken import example and references a
`fuzzy_match` function that doesn't exist in the current source; don't copy examples
from that file; everything above was verified against the real module.

## 3. Sandboxed code execution

`eval/safe_executor.py`'s `execute_code_safely` runs generated code in a separate
process (via `multiprocessing`) with a timeout, a restricted builtins dict, and captured
stdout. Pure stdlib, no dependencies, real process isolation (not just a `try/except`).
It auto-detects three test-case formats:

```python
from agenttune.eval.safe_executor import execute_code_safely

# 1. Structured dict format — {"input": ..., "expected_output": ...}
result = execute_code_safely(
    code="def add(a, b): return a + b",
    test_cases=[
        {"input": [1, 2], "expected_output": 3},
        {"input": [10, 20], "expected_output": 30},
    ],
)
print(result.test_passed, result.test_results)
# True [{'test_id': 0, 'input': [1, 2], 'expected': 3, 'actual': 3, 'passed': True}, ...]

# 2. MBPP-style assertion strings — auto-detected by the leading "assert"
result = execute_code_safely(
    code="def is_even(n): return n % 2 == 0",
    test_cases=["assert is_even(4) == True", "assert is_even(3) == False"],
)
print(result.test_passed)   # True
```

The third supported format is HumanEval-style: a single string test payload containing a
`def check(candidate):` function, detected when `test_cases` is itself a `str` containing
that literal substring. `execute_code_safely` then `exec`s your generated function,
`exec`s the check string, and calls `check(your_function)`:

```python
# 3. HumanEval-style — test_cases is a single string with def check(candidate)
check_code = """
def check(candidate):
    assert candidate(2, 3) == 5
    assert candidate(-1, 1) == 0
"""
result = execute_code_safely(code="def add(a, b): return a + b", test_cases=check_code)
print(result.success, result.test_passed, result.test_results)
# True True [{'test_id': 0, 'format': 'humaneval_check', 'passed': True}]
```

`execute_code_safely` also strips markdown code fences automatically (`extract_code`)
before running, and `validate_code_syntax(code)` gives you a cheap syntax-only check
(`ast.parse`, no execution) if you want to reject malformed completions before spending a
process spawn on them:

```python
from agenttune.eval.safe_executor import validate_code_syntax

validate_code_syntax("def foo(): return 42")   # (True, '')
validate_code_syntax("def foo(: return 42")    # (False, 'Syntax error at line 1: invalid syntax')
```

Numeric comparisons in the structured-dict path use `math.isclose` (relative tolerance
`1e-5`), and list/tuple outputs are compared recursively, so `[1, 2.0000001]` vs.
`[1, 2]` still passes.

## 4. Programmatic agent-behavior scoring: `TrajectoryEvaluator`

`eval/agentic/trajectory_eval.py`'s `TrajectoryEvaluator` is the engine behind
`Project.evaluate_agentic()`. Its full `evaluate_batch()` needs an LLM judge (an API key,
real network calls) for the goal-completion/grounding/safety rubric, but eight of its
scoring methods are pure, deterministic heuristics over a trajectory dict and need
**no LLM, no API key, nothing**: call them directly:

```python
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

evaluator = TrajectoryEvaluator(tool_schemas={
    "search_inbox": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
})

trajectory = {
    "trajectory_id": "demo-1",
    "tool_calls": [
        {"name": "search_inbox", "arguments": {"query": "invoice"}},
        {"name": "search_inbox", "arguments": {"query": "invoice"}},   # duplicate call
        {"name": "read_email", "arguments": {"id": "abc"}},             # no schema registered
    ],
    "tool_outputs": ["3 results found", "3 results found", "error: not found"],
    "reasoning_trace": "I need to find the invoice email, then read it.",
    "initial_plan": ["search_inbox", "read_email"],
}

evaluator._calculate_tac(trajectory)    # 0.667 — Tool Argument Correctness (jsonschema-validated)
evaluator._calculate_arr(trajectory)    # 0.333 — API Redundancy Ratio (duplicate calls / total)
evaluator._calculate_scsr(trajectory)   # 0.0   — Self-Correction Success Rate (never recovered from the trailing error)
evaluator._calculate_rad(trajectory)    # ~0.27 — Reasoning-to-Action Density (char-count proxy)
evaluator._calculate_lcf(trajectory)    # 0     — Loop Collapse Frequency (needs 3+ identical consecutive calls)
evaluator._calculate_pas(trajectory)    # 1.0   — Plan Adherence Score (both planned steps show up in tool_calls)
```

`_calculate_pmed` (Path Minimum Edit Distance) and `_calculate_ase` (Action-State
Efficiency) both need a `golden_trajectory` key in the dict to compare against and return
`None` without one; `_calculate_ter` (Tool Efficacy Reward, unique tool outputs /
total outputs) needs only `tool_outputs`:

```python
trajectory_with_golden = {
    "trajectory_id": "demo-2",
    "tool_calls": [
        {"name": "search_inbox", "arguments": {"query": "invoice"}},
        {"name": "read_email", "arguments": {"id": "abc"}},
    ],
    "tool_outputs": ["3 results found", "invoice #251832"],
    # the "correct" tool sequence, same shape as tool_calls, for comparison
    "golden_trajectory": [
        {"name": "search_inbox", "arguments": {"query": "invoice"}},
        {"name": "read_email", "arguments": {"id": "abc"}},
    ],
}

evaluator._calculate_ter(trajectory_with_golden)    # 1.0  — every tool output was distinct
evaluator._calculate_pmed(trajectory_with_golden)   # 0    — zero edit distance from golden (exact match)
evaluator._calculate_ase(trajectory_with_golden)    # 1.0  — len(golden) / len(tool_calls), no wasted steps
```

`_calculate_bleu` (a lightweight unigram-overlap proxy for BLEU-1, needs `final_answer`
and `reference_answer` keys) is also pure and callable the same way. The one remaining
scoring method, `_calculate_semantic_similarity`, is `async` and genuinely needs an LLM
call (a BERTScore-style semantic-equivalence judge); that one really does need the
engine/API key, unlike the eight above.

Building a `TrajectoryEvaluator` instance does construct a default `APIEngine` (for the
LLM-judge half you're not using here); it doesn't make a network call at construction
time, only if you later call `evaluate_batch()`/`_evaluate_single()`, so instantiating it
purely to reach these heuristic methods is safe offline.

## 5. Standardized benchmarking: `LMEvalRunner`

`eval/lm_eval_integration.py` shells out to the real `lm_eval` CLI (`pip install
lm-eval`) and parses its JSON output back into `EvalResult` objects. This genuinely needs
the package installed and a real model; it is not something a doc snippet can safely
run, but the code below is the accurate, current call shape:

```python
from agenttune.eval import LMEvalConfig, LMEvalRunner, get_lm_eval_task

config = LMEvalConfig(
    model_name="Qwen/Qwen3-0.6B",
    batch_size=4,
    device="cuda",
    limit=100,             # cap samples per task for a quick smoke test
    output_dir="./lm_eval_results",
)
runner = LMEvalRunner(config)
results = runner.evaluate_tasks([get_lm_eval_task("gsm8k"), get_lm_eval_task("hellaswag")])
for r in results:
    print(r.task_name, r.metrics)
```

`get_available_lm_eval_tasks()` lists the predefined tasks:
`hellaswag`, `arc_challenge`, `arc_easy`, `mmlu`, `truthfulqa`, `gsm8k`, `human_eval`,
`mbpp`, `crows_pairs`, `winogender`. `run_standard_benchmark(model_name, tasks=None)` is a
one-call convenience wrapper defaulting to
`["hellaswag", "arc_challenge", "mmlu", "gsm8k", "human_eval"]`.

Under the hood it literally builds and runs an `lm_eval` subprocess
(`_build_command`/`_run_evaluation`), so whatever you'd pass to the CLI directly applies
here too; it inherits lm-eval's own model-loading semantics (HF `AutoModel` by default).
See the [Local Notebooks](../notebooks/local-notebook.md) index for a full run against a real model.

## Notes & gotchas

- **`agenttune.eval.cli` cannot be imported.** It imports a class named
  `EvaluationRegistry` from `.registry`; the real class there is `EvalRegistry`. This
  raises `ImportError` the instant anything imports the module, and it has no
  console-script entry point, so nothing else in the repo exercises it. Don't build on
  top of `eval/cli.py`.
- **`RougeMetric`/`BleuMetric` fail silent, not loud.** Missing `rouge_score`/`nltk` gives
  you a clean-looking `0.0`, not an exception. Always sanity-check a known-good pair
  scores non-zero before trusting a real eval run.
- **Two different `EvalRegistry` classes exist** (`eval/core.py`'s, exported from
  `agenttune.eval`; `eval/registry.py`'s, a separate adapter class), same name, different
  responsibilities, easy to grab the wrong one from an IDE autocomplete.
- **`run_eval` (agent_eval.py) has zero callers elsewhere in the repo.** That's not a
  correctness problem, every code path above was verified working, it just means you're
  the first caller, so don't assume any other part of the training pipeline is aware of
  its `Report` format.

For the full list of known gaps across the whole repo (not just eval), see
[Known Issues](../community/known-issues.md).
