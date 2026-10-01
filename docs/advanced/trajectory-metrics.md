# Trajectory metrics reference: TAC, ARR, SCSR, RAD, LCF, PAS, PMED, ASE

`TrajectoryEvaluator` (`src/agenttune/eval/agentic/trajectory_eval.py`) computes eight
programmatic (no-LLM-call) metrics on an agent trajectory, plus one LLM-judged score and a
handful of simpler proxies (BLEU-1 overlap, an LLM-based semantic-similarity proxy, latency).
The acronyms show up all over the reward/eval stack: `AgenticEvalResult`
(`decide/closed_loop/contracts.py`), `hybrid_prm_reward`
(`agentic/rewards/builtin_rewards/hybrid_prm.py`), `EventLog.to_eval_dict()` (see
[EventLog schema](eventlog-schema.md)), without being defined anywhere. This page reads each
`_calculate_*` method's actual body and gives a small hand-built example for each.

All eight take the same input shape: a plain `trajectory: Dict[str, Any]`, the same dict
`EventLog.to_eval_dict()` produces (`{"tool_calls": [...], "tool_outputs": [...]}`), plus
whichever of `reasoning_trace` / `initial_plan` / `golden_trajectory` a particular metric
needs (none of those three are populated by `to_eval_dict()`, so PAS/PMED/ASE/RAD are only
meaningful on trajectory dicts assembled with that extra context; see the note under each).

## Quick reference

| Metric | What it needs beyond `tool_calls`/`tool_outputs` | Range | No-data return | Used by `hybrid_prm_reward`? |
|---|---|---|---|---|
| TAC | `self.tool_schemas` (constructor arg) | `[0, 1]` | `1.0` (no calls) | Yes, always |
| ARR | — | `[0, 1]` | `0.0` (no calls) | Yes, always |
| SCSR | — | `[0, 1]` | `1.0` (no errors) | Only if `use_llm_judge=True` |
| RAD | `reasoning_trace` | `[0, ∞)` | `1.0` (empty serialized calls) | Never |
| LCF | — | `{0, 1, 2, ...}` (raw count, not normalized) | `0` (<3 calls) | Yes, always |
| PAS | `initial_plan` | `[0, 1]` or `None` | `None` (no plan) | Never |
| PMED | `golden_trajectory` | `{0, 1, 2, ...}` (raw edit distance, not normalized) | `None` (no golden path) | Never |
| ASE | `golden_trajectory` | `[0, ∞)` or `None` | `None` (no golden path) | Never |

The `None`-vs-`0.0` distinction for PAS/PMED/ASE is deliberate in the source: it separates "not
applicable to this trajectory" (no plan/golden path was ever provided) from "applicable, and
scored zero" (a plan or golden path exists but the agent's actual calls didn't match it at
all).

## TAC: Tool Argument Correctness

```python
def _calculate_tac(self, trajectory: Dict[str, Any]) -> float:
    """Tool Argument Correctness (TAC) / Parameter Hallucination Rate.
    Validates tool arguments against self.tool_schemas using jsonschema."""
```

For every call in `tool_calls`: if it's a dict with a `name` that exists as a key in
`self.tool_schemas`, its `arguments` (parsed from a JSON string if needed) are validated
against that JSON Schema with `jsonschema.validate`; success increments `valid_args`, any
exception (including malformed JSON) does not. A call whose name isn't in `self.tool_schemas`
still increments `total_args` but never `valid_args`; **effectively invalid by default**. A
non-dict call (bare string) is the one exception: it counts as valid automatically ("assume
valid for fallback"). No tool calls → returns `1.0` (vacuously correct). The final score is
`valid_args / total_args`.

**This means TAC only measures anything if `tool_schemas` is actually supplied to the
`TrajectoryEvaluator` constructor.** With the default `tool_schemas={}`, every dict-shaped
tool call falls into the "unknown tool" branch and TAC silently degrades to "fraction of
tool calls that were bare strings, worth knowing before trusting a TAC number that looks
suspiciously low.

**Worked example:**

```python
schemas = {"search": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}
traj = {"tool_calls": [
    {"name": "search", "arguments": {"query": "hi"}},   # valid: has required "query"
    {"name": "search", "arguments": {"bad_arg": 1}},    # invalid: missing "query"
]}
ev = TrajectoryEvaluator(tool_schemas=schemas)
ev._calculate_tac(traj)   # -> 0.5   (1 valid_args / 2 total_args)
```

## ARR: API Redundancy Ratio

```python
def _calculate_arr(self, trajectory: Dict[str, Any]) -> float:
    """API Redundancy Ratio (ARR)"""
```

The fraction of tool calls that are exact duplicates of a call seen earlier in the same
trajectory. Each call is serialized (`json.dumps(call, sort_keys=True)` for dicts, `str(call)`
otherwise) and checked against a running `seen` set; a repeat increments `duplicates`. No
calls → `0.0`. Otherwise `duplicates / len(tool_calls)`. Note this is exact-match only; two
calls to the same tool with even slightly different arguments (e.g. a retry with a corrected
query) do **not** count as redundant.

**Worked example:**

```python
traj = {"tool_calls": [
    {"name": "search", "arguments": {"query": "a"}},
    {"name": "search", "arguments": {"query": "a"}},   # exact duplicate of call 1
    {"name": "search", "arguments": {"query": "b"}},
]}
ev._calculate_arr(traj)   # -> 0.333...   (1 duplicate / 3 calls)
```

## SCSR: Self-Correction Success Rate

```python
def _calculate_scsr(self, trajectory: Dict[str, Any]) -> float:
    """Self-Correction Success Rate (SCSR)"""
```

Scans `tool_outputs` for the literal (case-insensitive) substrings `"error"`, `"exception"`,
or `"invalid"`. For every output `i` (except the last) that contains one of those substrings,
that's a counted error; if the *next* output (`i+1`) contains none of them, that's a counted
recovery. Separately, and this is easy to miss on a quick read, the **last** output is
checked again on its own: if it still contains an error keyword, that's one more counted
error, with no possibility of a matching recovery (there's no output after it). No errors
encountered anywhere → `1.0` (nothing to recover from, treated as a perfect score, not
undefined). Otherwise `successful_recoveries / errors_encountered`.

Because the last element is inspected both as `out_next` inside the loop *and* separately
afterward, a trajectory that ends on an unresolved error effectively penalizes it twice: once
for failing to recover from the second-to-last output, and again as its own standalone error.

**Worked example:**

```python
traj = {"tool_outputs": ["Error: bad query", "still Error here"]}
# loop (range(len-1) = range(1)): i=0 -> out_i has "error" -> errors_encountered=1
#                                        out_next has "error" too -> not a recovery
# final check: last output ("still error here") has "error" -> errors_encountered=2
ev._calculate_scsr(traj)   # -> 0.0   (0 recoveries / 2 errors)

traj2 = {"tool_outputs": ["Error: invalid syntax", "42", "Result ok"]}
# i=0: "error" in out_i -> errors_encountered=1; out_next="42" has none -> recovery=1
# i=1: "42" has no error keyword -> skip
# final check: last output "result ok" has no error keyword -> no extra increment
ev._calculate_scsr(traj2)  # -> 1.0   (1 recovery / 1 error)
```

## RAD: Reasoning-to-Action Density

```python
def _calculate_rad(self, trajectory: Dict[str, Any]) -> float:
    """Reasoning-to-Action Density (RAD) using char counts as proxy"""
```

A crude but cheap proxy: character-length of `reasoning_trace` (joined with spaces first if
it's a list) divided by character-length of `json.dumps(tool_calls)`. Higher RAD means the
agent is writing a lot of reasoning relative to how much action it's taking; lower RAD means
it's mostly acting with little explanation. `reasoning_trace` is not populated by
`EventLog.to_eval_dict()`; this metric only means something on a trajectory dict you've
separately attached reasoning text to.

**Worked example:**

```python
traj = {"reasoning_trace": "check twice",                       # 11 characters
        "tool_calls": [{"name": "search"}]}
# json.dumps([{"name": "search"}]) == '[{"name": "search"}]'    # 20 characters
ev._calculate_rad(traj)   # -> 0.55   (11 / 20)
```

## LCF: Loop Collapse Frequency

```python
def _calculate_lcf(self, trajectory: Dict[str, Any]) -> int:
    """Loop Collapse Frequency (LCF)"""
```

Counts how many times the tool-call sequence collapses into **3 or more identical
consecutive calls**. Fewer than 3 total tool calls → `0` immediately (can't form a
3-repeat run). Otherwise it walks the sequence tracking a `consecutive_repeats` counter that
resets to `1` on any change and increments on an exact repeat (same dict-or-string
representation as the previous call); `lcf` is incremented **only at the moment
`consecutive_repeats` first reaches exactly `3`**; a run of 4, 5, or more identical calls
still only contributes `1` to `lcf`, not one per extra repeat past the third.

Note this is an `int`, not normalized to `[0, 1]` like the others; it's a raw count of
distinct loop-collapse events.

**Worked example:**

```python
A = {"name": "search", "arguments": {"query": "x"}}
B = {"name": "answer"}
traj = {"tool_calls": [A, A, A, B]}
# i=1: A==A -> consecutive_repeats=2
# i=2: A==A -> consecutive_repeats=3 -> lcf += 1
# i=3: B!=A -> reset consecutive_repeats=1
ev._calculate_lcf(traj)   # -> 1
```

## PAS: Plan Adherence Score

```python
def _calculate_pas(self, trajectory: Dict[str, Any]) -> float:
    """Plan Adherence Score (PAS)"""
```

Requires `initial_plan` (a list of plan-step strings); **no plan → returns `None`**, not
`0.0`, distinguishing "not applicable" from "zero adherence." Given a plan, tool names are
pulled from `tool_calls` (`call.get("name")` or `str(call)`), lower-cased. For each plan step,
if that step's (lower-cased) text is a **substring** of *any* tool name, it counts as matched.
Score is `matched / len(plan)`. An empty `tool_calls` with a non-empty plan returns `0.0`
directly (skips the substring loop).

**Worked example:**

```python
traj = {"initial_plan": ["search", "summarize", "translate"],
        "tool_calls": [{"name": "web_search"}, {"name": "summarizer"}]}
# "search" in "web_search"       -> matched (substring)
# "summarize" in "summarizer"    -> matched ("summarizer" contains "summarize")
# "translate" in <either name>   -> not matched
ev._calculate_pas(traj)   # -> 0.666...   (2 / 3)
```

## PMED: Path Minimum Edit Distance

```python
def _calculate_pmed(self, trajectory: Dict[str, Any]) -> int:
    """Path Minimum Edit Distance (PMED)"""
```

Requires `golden_trajectory` (a reference list of "correct" tool calls); no golden path →
`None`. Extracts tool-name sequences from both `tool_calls` (`seq1`) and `golden_trajectory`
(`seq2`), then runs a standard Levenshtein-distance dynamic-program over the two **name**
sequences (insertion/deletion/substitution all cost `1`). Returns the raw integer edit
distance, not normalized by sequence length, so it grows with trajectory length and isn't
directly comparable across trajectories of very different lengths.

**Worked example:**

```python
traj = {"tool_calls": [{"name": "search"}, {"name": "filter"}, {"name": "answer"}],
        "golden_trajectory": [{"name": "search"}, {"name": "answer"}]}
# seq1 = ["search", "filter", "answer"], seq2 = ["search", "answer"]
# seq2 is seq1 with "filter" deleted -> edit distance 1
ev._calculate_pmed(traj)   # -> 1
```

## ASE: Action-State Efficiency

```python
def _calculate_ase(self, trajectory: Dict[str, Any]) -> float:
    """Action-State Efficiency (ASE)"""
```

Also requires `golden_trajectory`; no golden path → `None`; empty `tool_calls` with a golden
path present → `0.0`. Otherwise simply `len(golden_trajectory) / len(tool_calls)`. Values
above `1.0` mean the agent reached the goal in *fewer* steps than the reference path (more
efficient than golden); values below `1.0` mean it took more steps (less efficient).

**Worked example:**

```python
traj = {"tool_calls": [{"name": "search"}, {"name": "filter"}, {"name": "answer"}],
        "golden_trajectory": [{"name": "search"}, {"name": "answer"}]}
ev._calculate_ase(traj)   # -> 0.666...   (2 golden steps / 3 actual steps -> less efficient)
```

## The other fields on `AgenticEvalResult`

Reading `decide/closed_loop/contracts.py`'s `AgenticEvalResult` dataclass alongside
`trajectory_eval.py` turns up a few more scores worth knowing about, even though they weren't
asked for above:

- **`ter_score`** (Tool Efficacy Reward): `len(set of unique tool_outputs) / len(tool_outputs)`,
  a novelty heuristic (`_calculate_ter`): `0.0` with no outputs, `1.0` if every output was
  distinct.
- **`bleu_score`**: a unigram-overlap proxy (`_calculate_bleu`) between `final_answer` and
  `reference_answer`: `|overlap tokens| / |reference tokens|`. Named BLEU but is not real BLEU
  (no n-gram precision, no brevity penalty).
- **`bert_score`**: despite the name, not BERTScore. `_calculate_semantic_similarity` makes a
  real LLM call asking it to rate semantic equivalence 0.0–1.0 and parses that out of a JSON
  response. Costs a model call per trajectory; returns `0.0` on any failure (bad JSON, API
  error).
- **`latency_ms`**: just `trajectory.get("latency_ms", 0.0)`, not computed.
- **`goal_completion_score`, `tool_sequence_validity`, `unnecessary_steps_penalty`,
  `error_recovery_score`, `overall_judge_score`, `iasa_score` (Intent-Action Semantic
  Alignment), `scr_score` (Sub-goal Completion Rate), `egs_score` (Evidence Grounding
  Score)**: these come from the LLM judge call inside `_evaluate_single`/`evaluate_batch`
  (`_build_judge_prompt`), not from any `_calculate_*` method. `scr_score` is only populated
  (else stays `None`) when the trajectory dict has a `subgoals` key. `rotation_id` records
  which of three paraphrased judge system-prompts was used for that call; see
  [Reward-hacking defenses](reward-hacking-defenses.md#judge-prompt-rotation) for what that
  rotation is actually defending against, and how `evaluate_batch`'s `overall_judge_score`
  gets passed through `RuleGuardCombinator` before being returned.

## Constructing a `TrajectoryEvaluator`

```python
class TrajectoryEvaluator:
    def __init__(self, engine: InferenceEngine = None,
                 model_name: str = "groq/llama-3.3-70b-versatile",
                 api_base: str = None, tool_schemas: Dict[str, Any] = None):
        self.engine = engine or APIEngine(model_name=model_name, api_base=api_base)
        self.model_name = model_name
        self.api_base = api_base
        self.tool_schemas = tool_schemas or {}
```

(`model_name`/`api_base` are stored as instance attributes as of this fix; an earlier version
only passed them to `APIEngine` and never kept them, which silently broke
`_calculate_semantic_similarity`'s own `litellm.acompletion(model=self.model_name, ...)` call
with an `AttributeError` swallowed by its blanket `except`, always returning `0.0` for
`bert_score` regardless of the actual answers. Fixed and verified: a real semantic-similarity
call now returns a genuine score instead of silently degrading to `0.0`.)

Two constructor args matter for the metrics on this page specifically: `tool_schemas` (see
TAC above, pass it or TAC degrades to near-meaningless) and, indirectly, `engine`. The eight
programmatic metrics never touch `self.engine` at all; they're pure functions of the
trajectory dict; only the LLM-judged fields (`goal_completion_score`, `overall_judge_score`,
etc.) and `bert_score` actually call out to a model. That split is why `hybrid_prm_reward`
can run `TrajectoryEvaluator(model_name="offline-deterministic-only")` when
`use_llm_judge=False` (see below); the model name is passed through to `APIEngine` but
never actually invoked if nothing calls `self.engine.generate_single`/`generate_batch`,
because none of `_calculate_tac`/`_calculate_arr`/etc. do.

## How `hybrid_prm_reward` gets from raw completions to a trajectory dict

Every metric on this page takes a pre-built `trajectory` dict as input, but a real RL rollout
only gives you raw prompt/completion strings. `hybrid_prm_reward`
(`agentic/rewards/builtin_rewards/hybrid_prm.py`) is the real code that bridges that gap, and
it's worth seeing before the metrics themselves, since it's the actual caller most of them go
through in training:

```python
from agenttune.decide.closed_loop.training_example_generator import _parse_tool_call

for prompt, comp in zip(prompts, completions):
    tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', comp, re.DOTALL)
    parsed_tools = []
    for block in tool_call_blocks:
        parsed = _parse_tool_call(f"<tool_call>{block}</tool_call>")
        if parsed:
            parsed_tools.append(parsed)
    if not parsed_tools:
        fallback = _parse_tool_call(comp)   # try the raw completion as a last resort
        if fallback:
            parsed_tools.append(fallback)
    trajectories.append({
        "trajectory_id": "rl_rollout", "prompt": prompt, "completion": comp,
        "tool_calls": parsed_tools, "error": None, "latency_ms": 100.0,
    })
```

Every `<tool_call>...</tool_call>` block in the raw completion text is regex-extracted and run
through `_parse_tool_call` (reused from `decide/closed_loop/training_example_generator.py`,
not reimplemented here) to build the `tool_calls` list every metric on this page actually
reads. Note `tool_outputs` is never populated by this reconstruction; it's absent from the
dict entirely, so any metric this page documents that depends on `tool_outputs` (SCSR, and
`ter_score`/BLEU indirectly) reads as its own no-data default (`1.0` for SCSR, since
`errors_encountered` stays `0` with an empty list) when scored through `hybrid_prm_reward`
specifically. `latency_ms` is hardcoded to `100.0` here too, not measured, so
`_calculate_latency` on a `hybrid_prm_reward`-built trajectory always returns `100.0`
regardless of the rollout's actual wall-clock time.

## Downstream use: what `hybrid_prm_reward` actually reads

`hybrid_prm_reward` (`agentic/rewards/builtin_rewards/hybrid_prm.py`) is the reward function
that turns `TrajectoryEvaluator` output into a GRPO/PPO/RLOO-compatible scalar. Reading its
body precisely, it does **not** use all nine scores on `AgenticEvalResult`, only a specific
subset:

```python
tac = getattr(r, "tac_score", 0.0)
arr = getattr(r, "arr_score", 0.0)
lcf = getattr(r, "lcf_score", 0)

score = tac - (arr * arr_penalty_weight)
if lcf > 0:
    score -= (lcf_penalty_weight * lcf)

if use_llm_judge:
    base = getattr(r, "overall_judge_score", 0.0)
    scsr = getattr(r, "scsr_score", 0.0)
    iasa = getattr(r, "iasa_score", 0.0)
    score = (score + base + scsr + iasa) / 4.0

final_reward = max(0.0, min(1.0, score))
```

| Metric | Used by `hybrid_prm_reward`? | How |
|---|---|---|
| TAC | **Yes, always** | Base score term: `score = tac - arr * arr_penalty_weight` |
| ARR | **Yes, always** | Penalty term, weighted by `arr_penalty_weight` (default `0.2`) |
| LCF | **Yes, always** | Penalty term, weighted by `lcf_penalty_weight` (default `0.1`), only applied if `lcf > 0` |
| SCSR | Only if `use_llm_judge=True` | Averaged into the final score with `overall_judge_score` and `iasa_score` |
| IASA | Only if `use_llm_judge=True` | Same averaging step |
| `overall_judge_score` | Only if `use_llm_judge=True` | Same averaging step |
| RAD, PAS, PMED, ASE | **Never** | Computed by `TrajectoryEvaluator` and stored on the returned `AgenticEvalResult`, but `hybrid_prm_reward` never reads them; they're available for a custom reward function to use, just not wired into this one. |
| TER, BLEU, `bert_score`, latency | **Never** | Same: computed and stored, not read here. |

So the "default False" `use_llm_judge` kwarg is the real gate: with it off,
`hybrid_prm_reward` is a **fully programmatic, no-LLM-call** reward built purely from
TAC/ARR/LCF, good for the hot GRPO loop where you can't afford a judge call per rollout. Turn
it on and it blends in the LLM judge's `overall_judge_score`, `scsr_score`, and `iasa_score`
via a plain average of four terms. Either way, the final reward is clamped to `[0.0, 1.0]`
before being returned, and any exception raised while reconstructing trajectories or calling
`TrajectoryEvaluator.evaluate_batch` degrades the whole batch to `[0.0] * len(prompts)` rather
than propagating and crashing training.

See the [Local Notebooks](../notebooks/local-notebook.md) index for `hybrid_prm_reward`
exercised against a real rollout, and for the broader reward/eval picture these metrics
feed into.
