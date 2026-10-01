# Reward-hacking defenses

Three real, independent mechanisms in this repo push back against an agent (or a training
run) learning to exploit its own reward signal rather than the underlying task: deterministic
score clamping on top of an LLM judge, relative-vs-absolute judge scoring, and judge
prompt rotation. None of them are theoretical; each is a small amount of real code, read in
full below. See the [Local Notebooks](../notebooks/local-notebook.md) index for these
exercised together, and [Trajectory metrics reference](trajectory-metrics.md) for the
programmatic metrics (TAC/ARR/LCF/etc.) these defenses sit alongside.

## Score clamping: `RuleGuardCombinator`

`agentic/rewards/judges/rule_guards.py` is a 48-line file with one class. The problem it
solves: an LLM judge is "enthusiastic" (the source's own word); it can be talked into a high
score by an answer that reads well but isn't actually grounded in anything the agent
retrieved. `RuleGuardCombinator` wraps a judge function and deterministically overrides its
score when a cheap, non-LLM check fails, so the judge's own soft failure mode can't leak
through as-is:

```python
class RuleGuardCombinator:
    """
    Wraps an existing reward function (like an LLM judge) and applies deterministic
    rules to clamp or override the score. This prevents the RM from inheriting exploits.
    """
    def __init__(self, base_judge_fn: Callable):
        self.base_judge_fn = base_judge_fn

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        base_scores = self.base_judge_fn(prompts, completions, **kwargs)

        final_scores = []
        for prompt, comp, score in zip(prompts, completions, base_scores):
            chunks = kwargs.get("retrieved_chunks", [])
            if chunks and not self._check_grounding_overlap(comp, chunks):
                score = min(score, 0.2)   # clamp to 0.2 maximum if ungrounded
            final_scores.append(score)

        return final_scores

    def _check_grounding_overlap(self, completion: str, chunks: List[str]) -> bool:
        if not chunks:
            return True   # pass if no chunks were provided
        comp_lower = completion.lower()
        for chunk in chunks:
            words = [w for w in chunk.lower().split() if len(w) > 5]
            for word in words:
                if word in comp_lower:
                    return True
        return False
```

Three things worth being precise about, reading this literally rather than by name:

- **The grounding check is a lower bound (`min(score, 0.2)`), not a replacement.** A judge
  that already scored below `0.2` for other reasons keeps that lower score. The guard only
  ever pulls a score *down*, never up, and only when it fires.
- **The overlap heuristic is deliberately crude**: any word longer than 5 characters from any
  retrieved chunk that appears anywhere in the completion counts as "grounded." The source
  comment is upfront about this: "In a real system, this would use a more sophisticated
  N-gram or semantic overlap." This is a cheap first line of defense, not a semantic
  grounding check.
- **The guard is a no-op if `retrieved_chunks` is never passed.** `chunks = kwargs.get(...)`
  defaults to `[]`, and `if chunks and not self._check_grounding_overlap(...)` short-circuits
  on the falsy empty list, no chunks means no clamping happens at all, silently. The
  combinator only does anything if the caller actually threads `retrieved_chunks` through
  `**kwargs`.

### Where this actually runs: `TrajectoryEvaluator.evaluate_batch`

`RuleGuardCombinator` isn't just defined; it's wired into the real batch-judging path in
`eval/agentic/trajectory_eval.py`, verified in `evaluate_batch`:

```python
base_overall = float(parsed.get("overall_score", 0.0))

guard = RuleGuardCombinator(lambda p, c, **kw: [base_overall])
chunks = traj.get("retrieved_chunks", [])
final_answer = traj.get("final_answer", content)
clamped_score = guard(["dummy_prompt"], [final_answer], retrieved_chunks=chunks)[0]
```

The "base judge function" passed in here is a one-line lambda that just returns the score the
LLM already gave. `RuleGuardCombinator` is being used purely for its clamping side effect on
a single already-computed score, not to combine multiple judges. `clamped_score` (not the raw
`base_overall`) is what ends up as `AgenticEvalResult.overall_judge_score`. This same call
site also writes both the raw and clamped score to a `judgments.jsonl` file (path from the
`JUDGMENTS_FILE` env var, defaulting under `data/`); the source labels this "D1 Hook", so
you can audit, after the fact, exactly which trajectories got clamped and by how much.

### The single-trajectory path does not clamp

Worth being precise about a real asymmetry in `trajectory_eval.py`: `TrajectoryEvaluator`
has two entry points for getting a judged score: `_evaluate_single` (used internally, one
trajectory at a time) and `evaluate_batch` (the one shown above). Only `evaluate_batch` builds
a `RuleGuardCombinator` and clamps. `_evaluate_single` parses the judge's JSON response and
assigns `overall_judge_score=float(parsed.get("overall_score", 0.0))` **directly**, with no
grounding check at all:

```python
async def _evaluate_single(self, trajectory, scoring_mode="absolute"):
    ...
    content = await self.engine.generate_single(messages, **kwargs)
    parsed = json.loads(content)
    return AgenticEvalResult(
        ...
        overall_judge_score=float(parsed.get("overall_score", 0.0)),
        ...
    )
```

If your code path calls `_evaluate_single` directly (or anything that routes through it rather
than through `evaluate_batch`), the grounding clamp simply isn't applied. The score-clamping
defense is real, but it is specifically a property of the batch path, not of
`TrajectoryEvaluator` as a whole. Prefer `evaluate_batch` (even for a batch of one) when the
grounding check matters.

## Relative vs. absolute judge scoring

Two genuinely different mechanisms in this repo both go by "relative scoring," and it's worth
keeping them apart because they solve different problems.

### 1. Relative *prompting*: `LLMJudge.evaluate_batch(mode=...)`

`agentic/rewards/llm_judge.py`'s `LLMJudge.evaluate_batch` takes a `mode: Literal["absolute",
"relative"]` (default `"absolute"`):

```python
def evaluate_batch(
    self,
    task: str,
    trajectories: List[Any],
    mode: Literal["absolute", "relative"] = "absolute",
    criteria: Optional[Dict[str, float]] = None,
) -> List[JudgeScore]:
    """
    mode="relative"  all shown to judge at once — better signal for GRPO
    mode="absolute"  each scored independently
    """
    if mode == "relative":
        system, user = _build_relative_prompt(
            task, trajectories, self.relative_rubric, self.system_prompt
        )
        try:
            return _parse_relative(self._call(_messages_from(system, user)), len(trajectories))
        except Exception:
            return [JudgeScore(score=0.5) for _ in trajectories]

    return [JudgeScore(score=self.evaluate_trajectory(task, t, criteria)) for t in trajectories]
```

In `"absolute"` mode, each trajectory gets its own independent judge call against a fixed
rubric (`ABSOLUTE_RUBRIC`); the judge never sees the other candidates. In `"relative"` mode,
`_build_relative_prompt` puts **every trajectory in the batch into one prompt**, tagged by
id, and asks the judge to score all of them together against `RELATIVE_RUBRIC`, whose text is
explicit about the comparison it wants:

```python
RELATIVE_RUBRIC = """
You are comparing multiple AI agent trajectories that were all given the same task.
Score each trajectory from 0.0 to 1.0 relative to the others:
  - A trajectory that accomplishes the goal MUST score higher than one that does not.
  - Prefer trajectories that are more efficient (fewer unnecessary steps).
  - If one is only slightly better, the score gap should be small.
  - You MAY give partial credit for progress towards the goal.
""".strip()
```

**Why this is harder to game than a fixed absolute rubric:** an absolute rubric is a single
static target; any completion that happens to pattern-match the rubric's surface criteria
(the right keywords, the right shape of answer, a confident tone) can score well regardless of
whether it actually solved the task, because the judge is never shown a better alternative to
compare against. A relative prompt forces the judge to look at several candidates for the
*same* task side by side and rank them against each other; a shallow, pattern-matching
response gets exposed the moment a genuinely better response sits right next to it in the
same prompt. This is exactly the point the docstring makes in passing, "better signal for
GRPO", since GRPO trains on the *relative* ordering of a group of rollouts for the same
prompt anyway, a relative judge call is a closer match to what the training signal actually
needs than nine independent absolute scores would be.

**Parsing robustness matters here specifically because relative mode is one judge call
scoring several trajectories at once.** If that single call's JSON response is malformed,
the failure mode affects the whole batch, not just one trajectory. `_parse_relative` handles
this explicitly:

```python
def _parse_relative(text: str, n: int) -> List[JudgeScore]:
    text = _strip_fences(text)
    try:
        obj     = json.loads(text)
        entries = obj.get("scores", obj) if isinstance(obj, dict) else obj
        if not isinstance(entries, list):
            raise ValueError
        result = [JudgeScore(score=max(0.0, min(1.0, float(e.get("score", 0.5)))),
                             explanation=str(e.get("explanation", "")),
                             trajectory_id=str(e.get("id", "")), raw_response=text)
                  for e in entries]
        if len(result) == n:
            return result
    except Exception:
        pass
    return [JudgeScore(score=0.5, raw_response=text) for _ in range(n)]
```

Every individual score is clamped to `[0.0, 1.0]` even on a well-formed response (`max(0.0,
min(1.0, ...))`), and the whole batch is rejected, falling back to a neutral `0.5` for every
trajectory, unless the parsed entry count exactly matches `n` (the number of trajectories
sent). A judge that scores 4 trajectories but only returns 3 entries doesn't get its partial
answer used; the mismatch itself is treated as a parse failure. Worth knowing: the fallback
score differs slightly depending on which call path hits it: `evaluate_batch`'s own
`try/except` around `_parse_relative` falls back to `JudgeScore(score=0.5)` per trajectory,
`async_evaluate_batch`'s equivalent path falls back to `0.5` as well for the relative branch,
but its **absolute**-mode `asyncio.gather` fallback (`return_exceptions=True`) uses `0.6` for
any trajectory whose coroutine raised. Neither number is a deep design decision; both are
just "assume moderately-okay" placeholders; but if you're auditing why a trajectory scored
exactly `0.5` or `0.6` with no rationale text, this is why.

### 2. Relative *normalization*: `TrajectoryEvaluator.evaluate_batch(scoring_mode=...)`

A second, distinct "relative" mode lives in `eval/agentic/trajectory_eval.py`'s own
`evaluate_batch`, and it does **not** change the judge prompt at all; every trajectory in the
batch still gets its own independent absolute judge call (each with its own rotated system
prompt, see below). What changes is a purely statistical post-processing step applied to the
scores *after* they come back:

```python
if scoring_mode == "relative" and len(results) > 1:
    # Group-Relative Scoring (RULER-style)
    scores = [r.overall_judge_score for r in results]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5 if variance > 0 else 1.0

    for r in results:
        r.overall_judge_score = (r.overall_judge_score - mean_score) / std_dev
```

This is a per-batch z-score normalization (mean 0, standard deviation 1) of the already-clamped
`overall_judge_score` values. The source comment calls it "Group-Relative Scoring
(RULER-style)." It's complementary to, not the same mechanism as,
`LLMJudge`'s relative *prompting*: this one keeps independent absolute judgments (each still
subject to `RuleGuardCombinator` clamping above) and only rescales them relative to their own
batch's mean/spread afterward, useful for a GRPO-style advantage signal where what matters is
how a rollout compares to its group, not its raw absolute score. `std_dev` falls back to
`1.0` when variance is `0` (e.g. a single-element batch, or every score identical), avoiding a
division by zero.

## Judge prompt rotation

`TrajectoryEvaluator._build_judge_prompt` (`eval/agentic/trajectory_eval.py`), not the RAG
reward stack, is where the real prompt-rotation mechanism lives. (Worth flagging honestly:
`rag/rewards/finder_rewards.py`'s module docstring lists "LLM-judge correctness, eval-time
only (`judge_eval.py`, prompt rotation)" among the RAG reward stack's excluded components, but
`rag/rewards/judge_eval.py` itself has no rotation logic at all; it builds one `LLMJudge`
with a single fixed `GROUNDEDNESS_RUBRIC` and calls `evaluate_trajectory` once per trajectory.
The actual rotation implementation the comment is pointing at lives on the agentic side, in
`trajectory_eval.py`, and is reused wherever `TrajectoryEvaluator` itself is the judge.)

```python
system_prompts = [
    # 0: Standard
    """You are an expert agent trajectory evaluator.
Score the provided agent trajectory on the following metrics (each 0.0 to 1.0):
...""",
    # 1: Paraphrase 1
    """You act as a harsh but fair judge of AI agent workflows.
Review the trajectory and score these specific criteria from 0.0 (fail) to 1.0 (perfect):
...""",
    # 2: Paraphrase 2
    """As a senior AI auditor, evaluate the agent's step-by-step execution.
Provide a score between 0.0 and 1.0 for each metric below:
..."""
]

rotation_id = random.randint(0, len(system_prompts) - 1)
system_prompt = system_prompts[rotation_id]
```

Three hand-written paraphrases of the same rubric (same six criteria: `goal_completion`,
`tool_sequence_validity`, `unnecessary_steps`, `error_recovery`, `intent_action_alignment`,
`evidence_grounding`, worded three different ways), and every single judge call, whether via
`_evaluate_single` or the batched `evaluate_batch`, picks one uniformly at random with
`random.randint` and records which one it used as `rotation_id`, which flows straight through
to `AgenticEvalResult.rotation_id`.

**Why this defends against reward hacking:** if a policy is trained against one fixed,
unchanging judge prompt, the strongest gradient it can find isn't necessarily "solve the task
better", it can just as easily be "produce output that happens to trigger *this exact
prompt's* scoring heuristics," a much narrower and more exploitable target. Randomly rotating
among differently-worded (but semantically equivalent) system prompts every call means a
policy has to satisfy the underlying six criteria robustly enough to score well regardless of
phrasing, rather than overfitting to one prompt's specific wording or ordering.

`rotation_id` isn't just generated and discarded; it's tracked end-to-end for exactly this
purpose. It's a real dataclass field on `AgenticEvalResult` (`decide/closed_loop/contracts.py`,
comment: `# For prompt rotation tracking`), and `JudgmentHook.log_judgment`
(`agentic/events/judgment_hook.py`) writes it alongside the judge's verdict into a persistent
`judgments.jsonl`, the same "D1 Hook" pattern `evaluate_batch` uses directly. Having the
rotation id on every logged judgment means you can later slice the log by which prompt
variant produced which score and check whether a policy's apparent improvement holds up
across all three phrasings or is concentrated on one, a real, checkable signal that the
policy is exploiting one specific prompt rather than genuinely improving.

## Judge-side robustness: caching and backend independence

Two more details from `LLMJudge` (`agentic/rewards/llm_judge.py`) round out the picture of
why these defenses are practical to run in a real training loop rather than just a one-off
eval.

**Caching is keyed on content, not identity.** `evaluate_trajectory`/`async_evaluate_trajectory`
both check a cache before making a call:

```python
def _cache_key(task: str, trajectory: Any, criteria: Dict) -> str:
    if hasattr(trajectory, "steps"):
        raw = task + str([(getattr(s, "action", ""), getattr(s, "observation", ""))
                          for s in trajectory.steps])
    elif isinstance(trajectory, list):
        raw = task + json.dumps(trajectory, sort_keys=True)
    else:
        raw = task + str(trajectory)
    return hashlib.md5((raw + json.dumps(criteria, sort_keys=True)).encode(),
                       usedforsecurity=False).hexdigest()
```

The key is built from the task string plus every step's `(action, observation)` pair (for a
real `Trajectory`) or the trajectory's own JSON-serialized content, not from an object id or
a trajectory id, so two *different* rollouts that happen to produce the exact same sequence
of actions and observations hit the same cache entry and skip a second judge call entirely.
`cache_size` (default `10_000`) caps how many entries `self._cache` (a plain dict) will hold;
the `evaluate_trajectory` write path is `if 0 < self.cache_size and len(self._cache) <
self.cache_size`, so the cache stops growing once full rather than evicting; it does not
implement LRU eviction, it just freezes once at capacity.

**Absolute mode supports weighted, named criteria**, not just a single scalar rubric:
`evaluate_trajectory(task, trajectory, criteria={"goal_completion": 0.5, "efficiency": 0.5})`
threads that dict into `_build_absolute_prompt`, which renders it into the prompt as a
`Criteria:` block (`f"  - {k} (weight {v:.2f})"` per line) alongside the rubric text; the
judge sees explicit weights, not just prose. `mode="relative"` batch calls don't take a
`criteria` argument at all; `_build_relative_prompt` only accepts the rubric text and the
list of trajectories, which is consistent with relative scoring being a holistic
side-by-side ranking rather than a per-criterion weighted rubric evaluation.

**The judge backend is swappable without touching any of the above.** `LLMJudge.__init__`
picks a `_provider` from `backend`/`model`/`rollout_engine`: a real API model string picks
`anthropic` (model starts with `"claude"`), `openrouter` (contains `"/"`), or plain `openai`;
`backend="transformers"`/`backend="vllm"` load a genuinely local model (`_TransformersLocalJudge`,
`_VLLMLocalJudge`, the vLLM one specifically documented as reusing TRL's `VLLMGeneration` in
`colocate` mode, sharing GPU memory with the policy rather than needing a second GPU); passing
an existing `rollout_engine` reuses whatever engine is already running the policy itself as
the judge. None of the clamping, rotation, or relative/absolute logic above cares which
provider actually answers the call. `_call`/`_acall` are the only two methods that branch on
`_provider`, and every defense mechanism sits entirely on top of their return value.

## Putting it together

A single call to `TrajectoryEvaluator.evaluate_batch` for a GRPO-style rollout batch, with
all three defenses active end-to-end: each trajectory gets scored against a randomly-rotated
judge system prompt (rotation tracked per result), the raw score is clamped via
`RuleGuardCombinator` if `retrieved_chunks` were supplied and grounding fails, and, if
`scoring_mode="relative"`, the whole batch of already-clamped scores is z-normalized against
its own mean and spread before being handed to training:

```python
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

evaluator = TrajectoryEvaluator(model_name="groq/llama-3.3-70b-versatile")

trajectories = [
    {"trajectory_id": "t1", "tool_calls": [...], "tool_outputs": [...],
     "final_answer": "...", "retrieved_chunks": ["...retrieved passage..."]},
    {"trajectory_id": "t2", "tool_calls": [...], "tool_outputs": [...],
     "final_answer": "...", "retrieved_chunks": ["...retrieved passage..."]},
    # ... rest of the GRPO group for this prompt
]

results = await evaluator.evaluate_batch(trajectories, scoring_mode="relative")
for r in results:
    print(r.trajectory_id, r.overall_judge_score, r.rotation_id)
    # overall_judge_score is: judged -> clamped by RuleGuardCombinator (if ungrounded) -> z-normalized
```

None of these three steps depend on each other: clamping fires per trajectory regardless of
`scoring_mode`, rotation happens on every judge call regardless of whether clamping or
normalization ever trigger, and `scoring_mode="relative"`'s normalization runs over whatever
scores came out of the (already-clamped) per-trajectory judge calls. Each can be exercised
independently, which is also how the corresponding
[Local Notebook](../notebooks/local-notebook.md) walks through them. If
your use case needs the *prompting*-level relative comparison instead (`LLMJudge.evaluate_batch(mode="relative")`,
which shows the judge all candidates in one prompt rather than z-normalizing independent
scores), reach for `LLMJudge` directly rather than `TrajectoryEvaluator`; the two relative
mechanisms documented above don't compose with each other; pick the one that matches whether
you want the judge itself to compare candidates, or want independent judgments rescaled
afterward.
