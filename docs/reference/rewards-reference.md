# Python API: Rewards

`agenttune.agentic.rewards`: the reward-function catalog (`REWARD_REGISTRY`), the
combinator that turns several of them into one GRPO-compatible callable
(`combine_rewards`), the LLM-as-judge trajectory evaluator (`LLMJudge`), and a
deterministic rule-guard wrapper. Pure heuristic rewards need nothing; the judge/PRM/
distilled-model rewards need a model or API key. See
[Python API: Agentic Spine](../user-guide/agentic-spine.md) for how this fits
alongside strategies/harnesses/tools, and
[Known Issues](../community/known-issues.md) for the shadowing bug documented in detail
below.

## `REWARD_REGISTRY`: how it's actually built

```python
# agentic/rewards/builtin_rewards/__init__.py
from .sql import *
from .use_case import *
from .hybrid_prm import hybrid_prm_reward
from .distilled_judge import distilled_judge_reward
# ...plus explicit additions below

REWARD_REGISTRY: dict[str, Callable] = { ... }  # 26 entries, listed below
```

Two `import *` statements land in the **same module namespace**, in this order: `sql.py`
first, then `use_case.py`. Both of these facts matter and are verified against the source
below, not assumed:

!!! note "Shadowed names still resolve to the weaker version, but the stronger ones are now also registered"
    `sql.py` *and* `use_case.py` **both** define a `format_reward`; because `use_case.py` is
    imported second, `REWARD_REGISTRY["format_reward"]` still resolves to its version: `+0.1`
    if the completion wraps its answer in `<answer>...</answer>`. `finqa.py` also has its own,
    numerically-careful `placeholder_coverage_reward`/`answer_correctness_reward` (unit-aware
    parsing, tolerance-based comparison across *all* gold numbers), and `REWARD_REGISTRY`'s
    plain `"placeholder_coverage_reward"`/`"answer_correctness_reward"` keys still resolve to
    `use_case.py`'s weaker, substring-matching versions, unchanged, for backward
    compatibility. What's changed: the stronger versions are **no longer registry-invisible**:
    `sql.py`'s `format_reward` is now reachable as `"sql_format_reward"`, and all four of
    `finqa.py`'s functions are reachable under `"finqa_format_reward"`,
    `"finqa_sql_grounding_reward"`, `"finqa_calculator_grounding_reward"`,
    `"finqa_placeholder_coverage_reward"`, `"finqa_answer_correctness_reward"`. You can still
    import any of them directly by module path instead of by registry key:
    ```python
    from agenttune.agentic.rewards.builtin_rewards.finqa import (
        placeholder_coverage_reward, answer_correctness_reward,
        sql_grounding_reward, calculator_grounding_reward,
    )
    ```
    See [Known Issues](../community/known-issues.md) for the original write-up and the fix.

`use_case.py`'s `tool_chain_reward(prompts, completions, **kwargs)` (+0.1 per tool in
`list_dir`/`read_file`/`run_python`/`write_file` found in the completion text, capped at 0.4)
and `summary_written_reward(prompts, completions, sample_dir=None, **kwargs)` (+0.2 if a new
file appears in `sample_dir` beyond the file-ingestion tool's original 4) are likewise now
registered, under their own names.

## The catalog: all 26 registered functions

Every function returns `list[float]`, one score per completion. **Needs** is "nothing" for
all of these; they're regex/string heuristics over the completion text. The `completions`
argument is a list where each item is either a raw string or a list of chat-message dicts;
functions that read the "final answer" pick the last `role: assistant` message when given
the list form.

| Registry key | File | Signature | Scores |
|---|---|---|---|
| `correctness_reward` | `sql.py` | `(completions, answer=None, **kwargs)` | Parses `*yes*`/`*no*` from the completion; `-0.5` no match, `0.6` matches `answer`, `-1.0` wrong |
| `structure_reward` | `sql.py` | `(completions, **kwargs)` | Tool-call → tool-response → content shape: `0.1`/`0.05` if well-formed, `-0.15` if a call was made with no response, else `0.0` |
| `query_reward` | `sql.py` | `(completions, answer=None, **kwargs)` | SQL strategy heuristic: penalizes >3 queries, `LIMIT 1`, missing `WHERE`; rewards a `yes`/`no` `answer` that matches whether any rows were found |
| `reward_correct_answer` | `sql.py` | `(completions, answer=None, **kwargs)` | Numeric match against `answer` in the final assistant message: `1.5` exact (±1e-3), `0.5` within 1%, else `0.0` |
| `reward_tool_used` | `sql.py` | `(completions, **kwargs)` | `1.0` if any message has `role: tool`, else `0.0` |
| `reward_concise_answer` | `sql.py` | `(completions, **kwargs)` | `0.3` if the final assistant message is ≤80 words |
| `format_reward` | **`use_case.py`** (shadows `sql.py`'s; see above) | `(prompts, completions, **kwargs)` | `0.1` if `<answer>...</answer>` is present |
| `search_grounding_reward` | `use_case.py` | `(prompts, completions, tool_call_counts=None, **kwargs)` | Tiered on `tool_call_counts`: `0`→0.0, `1`→0.2, `2`→0.3, `3+`→0.4. Returns all `0.0` if `tool_call_counts` isn't passed |
| `message_id_citation_reward` | `use_case.py` | `(prompts, completions, **kwargs)` | `+0.2` if the completion contains an email address or the string `message_id` |
| `answer_format_reward` | `use_case.py` | `(prompts, completions, **kwargs)` | `+0.1` if `<answer>` content is ≥10 chars |
| `answer_correctness_reward` | **`use_case.py`** (weaker than `finqa.py`'s; see above) | `(prompts, completions, answer=None, **kwargs)` | Compares `<answer>` text to `answer`: `1.0` exact, `0.5` substring either way, else `0.0` |
| `placeholder_coverage_reward` | **`use_case.py`** (weaker than `finqa.py`'s; see above) | `(prompts, completions, answer=None, **kwargs)` | Fraction of gold numeric tokens found as a **substring** of the prediction; `1.0`/`0.5`/`0.2`/`0.0` at 80%/50%/20% coverage |
| `template_structure_reward` | `use_case.py` | `(prompts, completions, answer=None, **kwargs)` | `+0.3 ×` fraction of gold's bolded `**headers**` reproduced in the prediction |
| `computation_reward` | `use_case.py` | `(prompts, completions, tool_call_counts=None, **kwargs)` | `0.2` for 1 call, `0.3` for 2+, `0.0` for 0/absent; intended for a `run_python` count |
| `numerical_match_reward` | `use_case.py` | `(prompts, completions, answer=None, **kwargs)` | Compares the **first** number in `<answer>` to the first number in `answer`: `1.0` exact, `0.5` within 5%, else `0.0` |
| `exploration_reward` | `use_case.py` | `(prompts, completions, tool_call_counts=None, **kwargs)` | Breadth tiers on `tool_call_counts`: `0`→0.0, `1`→0.1, `2`→0.2, `3+`→0.3 |
| `hybrid_prm_reward` | `hybrid_prm.py` | `(prompts, completions, **kwargs)` | See dedicated section below |
| `distilled_judge_reward` | `distilled_judge.py` | `(prompts, completions, **kwargs)` | See dedicated section below |
| `tool_chain_reward` | `use_case.py` | `(prompts, completions, **kwargs)` | `+0.1` per tool name (`list_dir`/`read_file`/`run_python`/`write_file`) found in the completion text, capped at `0.4` |
| `summary_written_reward` | `use_case.py` | `(prompts, completions, sample_dir=None, **kwargs)` | `+0.2` if a new file appears in `sample_dir` beyond the file-ingestion tool's original 4 |
| `sql_format_reward` | `sql.py` | `(completions, **kwargs)` | `+0.2` for markdown structure (bold/headers/lists/code fences); `sql.py`'s own `format_reward`, under its non-shadowed key |
| `finqa_format_reward` | `finqa.py` | `(prompts, completions, **kwargs)` | `finqa.py`'s own, differently-scored `format_reward` |
| `finqa_sql_grounding_reward` | `finqa.py` | `(prompts, completions, **kwargs)` | See [above](#the-catalog-all-26-registered-functions) |
| `finqa_calculator_grounding_reward` | `finqa.py` | `(prompts, completions, **kwargs)` | See above |
| `finqa_placeholder_coverage_reward` | `finqa.py` | `(prompts, completions, answer=None, **kwargs)` | Unit-aware, tolerance-based version of `placeholder_coverage_reward` |
| `finqa_answer_correctness_reward` | `finqa.py` | `(prompts, completions, answer=None, **kwargs)` | Tolerance-based comparison across *all* gold numbers, not just an exact/substring match |

## `combine_rewards(reward_funcs, weights=None)`

```python
def combine_rewards(reward_funcs, weights=None) -> Callable
```

`reward_funcs` accepts a single callable, a single string name (looked up in
`REWARD_REGISTRY`, raising `ValueError` listing every available key if unknown), or a list
mixing both. `weights` defaults to uniform and is always renormalized to sum to 1 (raises
`ValueError` if weights don't sum positive, or if `len(weights) != len(reward_funcs)`).

The returned callable's real signature is:

```python
def _combined(completions, **kwargs) -> list[float]
```

`completions` is the only parameter that isn't swept into `**kwargs`; everything else
(`prompts`, `answer`, `tool_call_counts`, ...) must be supplied as keyword arguments by
whoever calls the combined function, exactly how TRL's trainers invoke `reward_funcs`. For
each wrapped function, in weight order:

1. First tries **`fn(completions=completions, **kwargs)`**: a fully keyword-style call.
   This works for both calling conventions actually in use across the registry: `sql.py`'s
   functions (which declare `completions` first) bind it directly by name; `use_case.py`'s
   functions (which declare `prompts` first) also bind correctly *by name*, as long as the
   caller's `**kwargs` included `prompts=...`; Python resolves keyword arguments by name
   regardless of declared parameter order.
2. On `TypeError` (typically: `prompts` was never supplied, and a registry function
   requires it), retries **positionally**: `fn(completions, **kwargs)`. This binds the
   `completions` list into whatever the function's *first* parameter is named, `prompts`
   for every `use_case.py`/`hybrid_prm`/`distilled_judge` function, which is almost always
   wrong and typically raises a second `TypeError` (now missing the real `completions` arg).
3. That second `TypeError` is **not caught**; it propagates out of `combine_rewards`'s
   result. This is deliberate: the source comment notes that a reward function failing under
   both calling conventions is a real bug, and silently scoring it `0.0` would train the
   policy against a dead signal, which is worse than a loud crash.

Practical takeaway: always drive a `combine_rewards(...)` result with `prompts=` included in
your call's keyword arguments if any wrapped function needs it (which is most of the
registry), exactly what TRL's `GRPOTrainer` does automatically, but a manual/notebook call
that only passes `completions=` will crash on any `use_case.py`-style function.

Scores are combined as `sum(weight_i * float(score_i or 0.0))` per completion; the returned
function's `__name__` is set to `"combined_reward(name1+name2+...)"` for logging.

```python
combine_rewards(["correctness_reward", "structure_reward"])
combine_rewards(["correctness_reward", my_custom_fn], weights=[2.0, 1.0])
combine_rewards("reward_tool_used")          # single string, still returns a callable
```

## `RuleGuardCombinator` (`judges/rule_guards.py`)

```python
class RuleGuardCombinator:
    def __init__(self, base_judge_fn: Callable): ...
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]: ...
```

Wraps any reward/judge function and clamps its score to `min(score, 0.2)` when the
completion shares no vocabulary with `kwargs["retrieved_chunks"]` (a `list[str]`), a
deterministic guard against an LLM judge rewarding an ungrounded hallucination. The overlap
check itself is intentionally crude: any chunk word longer than 5 characters found verbatim
in the completion counts as grounded. If `retrieved_chunks` isn't passed, the guard is a
no-op and scores pass through unchanged. Not in `REWARD_REGISTRY`; wrap directly:
`RuleGuardCombinator(my_llm_judge_reward_fn)`. See the corresponding
[Local Notebook](../notebooks/local-notebook.md).

## `LLMJudge` (`llm_judge.py`)

```python
LLMJudge(
    model: str = "gpt-4o-mini",
    backend: Optional[Literal["transformers", "vllm"]] = None,
    model_path: Optional[str] = None,
    rollout_engine: Optional[Any] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    absolute_rubric: str = ABSOLUTE_RUBRIC,
    relative_rubric: str = RELATIVE_RUBRIC,
    cache_size: int = 10_000,
    judge_max_tokens: int = 256,
    local_gen_kwargs: Optional[Dict] = None,
    vllm_kwargs: Optional[Dict] = None,
)
```

Backend is chosen by what you pass, checked in this order: `rollout_engine=` (reuses an
already-loaded engine, e.g. the policy's own, avoiding a second model load) → `backend=
"transformers"`/`"vllm"` with a required `model_path=` (local judge, loaded eagerly in
`__init__`) → otherwise an API judge, provider inferred from `model`: a `"claude*"` prefix →
Anthropic, a `"/"` in the name → OpenRouter, else → OpenAI. API key resolution checks
`api_key=` first, then `ANTHROPIC_API_KEY`/`OPENROUTER_API_KEY`/`OPENAI_API_KEY` and raises
`ValueError` if neither is set.

| Method | Signature | Notes |
|---|---|---|
| `.evaluate_trajectory(task, trajectory, criteria=None)` | `-> float` | Scores one trajectory `[0, 1]`. Caches by `md5(task + trajectory + criteria)`, up to `cache_size` entries. Swallows exceptions and returns `0` on failure (prints a traceback but does not raise) |
| `.evaluate_batch(task, trajectories, mode="absolute", criteria=None)` | `-> list[JudgeScore]` | `mode="relative"` sends all trajectories in one prompt (better signal for GRPO, enforces a strict ranking); `mode="absolute"` scores each independently via `.evaluate_trajectory` |
| `.async_evaluate_trajectory` / `.async_evaluate_batch` | same shapes, `async def` | `async_evaluate_batch`'s default `mode` is `"relative"` (note: differs from the sync version's default of `"absolute"`) |
| `.as_reward_fn(criteria=None)` | `-> Callable[[trajectory], float]` | **The reward-fn adapter.** Returns `_fn(trajectory) -> float` that calls `self.evaluate_trajectory(getattr(trajectory, "task", ""), trajectory, criteria)`. Note the calling convention is deliberately different from every `REWARD_REGISTRY` function above: it takes **one trajectory object**, not `(completions, **kwargs)` lists; it's built for `create_rollout_fn(reward_fn=judge.as_reward_fn())`, a different integration point, not for `combine_rewards` |
| `.clear_cache()` | `-> None` | Empties the score cache |

`JudgeScore` is `@dataclass(score: float, explanation: str = "", trajectory_id:
Optional[str] = None, raw_response: Optional[str] = None)`.

## `hybrid_prm_reward` (`hybrid_prm.py`)

```python
def hybrid_prm_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]
```

Needs `agenttune.eval.agentic.trajectory_eval.TrajectoryEvaluator` and
`agenttune.decide.closed_loop.training_example_generator._parse_tool_call`. Regex-extracts
every `<tool_call>...</tool_call>` block from each completion, reconstructs a minimal
pseudo-trajectory dict per completion, and runs
`TrajectoryEvaluator(model_name=...).evaluate_batch(...)` via `asyncio.run(...)`.

Config, read from `**kwargs` per call (no dedicated params; pass them through whatever
calls this reward, e.g. `combine_rewards`'s pass-through `**kwargs`):

| kwarg | Default | Effect |
|---|---|---|
| `use_llm_judge` | `False` | Enables the soft LLM-judge metrics (IASA, SCSR) in addition to the deterministic ones |
| `eval_model_name` | `"groq/llama-3.3-70b-versatile"` | Model used for the LLM-judge metrics; needs a Groq (or whatever provider) API key when `use_llm_judge=True` |
| `arr_penalty_weight` | `0.2` | Weight subtracted for the ARR (action-repetition?) penalty score |
| `lcf_penalty_weight` | `0.1` | Weight subtracted for the LCF penalty score, applied only when `lcf_score > 0` |

Score per completion is `tac_score - arr_penalty_weight * arr_score [- lcf_penalty_weight *
lcf_score]`, clamped to `[0, 1]`; when `use_llm_judge=True` it's further averaged with
`overall_judge_score`, `scsr_score`, and `iasa_score` (4-way mean). If constructing or
running the `TrajectoryEvaluator` raises anything, the error is logged and the **entire
batch** gets `0.0`, a fault-tolerant fallback, not a raised exception. Also logs batch-mean
metrics to Weights & Biases if `wandb.run` is active. See the corresponding
[Local Notebook](../notebooks/local-notebook.md).

## `DistilledJudge` / `distilled_judge_reward` (`distilled_judge.py`)

```python
class DistilledJudge:
    def __init__(self, model_path: str = "custom_reward_model"): ...
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]: ...

def distilled_judge_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]
```

`DistilledJudge.__init__` tries to build a `transformers` `text-classification` pipeline over
`model_path`; on `ImportError` (no `transformers`) or any load failure it logs a
warning/error and leaves `self.pipeline = None`; **it never raises**. `__call__` returns
`[0.0] * len(prompts)` whenever `self.pipeline` is `None`, the "silently returns an
all-zero reward" behavior flagged in [Known Issues](../community/known-issues.md).
Otherwise it runs the pipeline on `prompt + completion` per pair (truncated to 2048 tokens)
and reads `result[0]["score"]`.

`distilled_judge_reward` is the registry wrapper. It reads the model path from
`kwargs.get("distilled_judge_path", "custom_reward_model")` and lazily builds **one**
`DistilledJudge`, cached as a function attribute
(`distilled_judge_reward._judge`) the first time it's ever called. That cache keys on
nothing but "has this attribute been set"; calling it once with one
`distilled_judge_path` and later with a different one **silently keeps reusing the first
judge and its original model_path**; the second path is never loaded.

## See also

- [Python API: Agentic Spine](../user-guide/agentic-spine.md) for the one-paragraph
  overview this page expands on.
- [Known Issues](../community/known-issues.md): the FinQA shadowing bug and the
  `DistilledJudge` silent-zero behavior, as originally documented.
