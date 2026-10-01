# Real examples (GPU required)

The case studies under [`examples/`](.) are GPU-free: they inject a
stand-in trainer and drive `run_episode` with a scripted policy, so they exercise the lifecycle
*plumbing* but never update a model's weights or let a real model make a decision. **These
examples do the opposite** — they load a real model and run it on the GPU through the real
library contract. Two surfaces:

**Training** — real weight updates through `Project.distill` / `Project.train` / TRL trainers:

```bash
python examples/agentic_distillation_real.py   # SFT / distillation
python examples/agentic_grpo_real.py            # on-policy GRPO (RL)
python examples/self_heal_dpo_real.py           # self-heal -> DPO (preference RL)
python examples/reward_model_real.py            # reward-model training (the RL scorer)
python examples/rloo_real.py                    # RLOO (on-policy RL, leave-one-out baseline)
python examples/bco_real.py                     # BCO (binary-classifier preference)
python examples/ppo_real.py                     # PPO (actor-critic RL with a reward model)
python examples/rag_training_real.py            # agentic-RAG GRPO: a real model calls the search tool mid-rollout
```

**Agentic inference** — a real model drives the agent loop / heal stages; nothing injected:

```bash
python examples/agentic_strategy_real.py        # real model IS the ReAct policy + agentic_metrics
python examples/self_heal_llm_real.py           # real LLM runs the classify + generate heal stages
```

**DECIDE closed loop** — the retrain → gate → deploy chain produces and deploys a real adapter:

```bash
python examples/closed_loop_real.py             # FullClosedLoop: real DPO retrain -> real gate -> real deploy
```

**Real components & transport** — the service and the pluggable pillars run for real, no GPU
needed (these close "documented-but-only-tested-in-process" gaps rather than model gaps):

```bash
python examples/service_real.py                 # the FastAPI service stood up as a real uvicorn server
python examples/vector_memory_real.py           # VectorMemory with a real sentence-transformer embedder
python examples/openenv_harness_real.py         # OpenEnvHarness against a real openenv Environment
```

The training and agentic-inference examples require a CUDA GPU and the models in the local HF
cache (`HuggingFaceTB/SmolLM2-360M-Instruct`; `self_heal_llm_real.py` and `rag_training_real.py`
use `Qwen/Qwen2.5-3B-Instruct` — as the heal LLM, and as a tool-calling RAG policy respectively).
The three real-component examples run on CPU:
`vector_memory_real.py` needs `sentence-transformers/all-MiniLM-L6-v2` cached.

## Agentic distillation, for real — `agentic_distillation_real.py`

A teacher's demonstrations become full-tier trajectories; `Project.distill` assembles the SFT
dataset from them and hands it to a **real** `trainer_factory` that fine-tunes the student with
TRL's `SFTTrainer` (LoRA, completion-only loss) on the GPU. To make the weight change visible,
the teacher demonstrates an output contract the base model does not follow — answer every request
with exactly `SENTIMENT=<label>` — and we generate before and after.

Output from an actual run (RTX PRO 6000, ~7s):

```
[gpu]      NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition  (cuda available: True)
[teacher]  16 full-tier trajectories -> 16 SFT rows (schema ['messages'])
...
{'loss': 3.6043, ... 'mean_token_accuracy': 0.474, 'epoch': 1.25}
{'loss': 0.2495, ... 'mean_token_accuracy': 0.940, 'epoch': 25.0}
{'train_runtime': 7.1, 'train_loss': 0.807, 'epoch': 25.0}
[distill]  TRL SFTTrainer finished: 7.1s, final train_loss=0.807
[probe]    prompt: Classify sentiment: 'the delivery was late and support ignored me'
[before]   base model  -> 'The sentiment of the given text is negative.'
[after]    distilled   -> 'SENTIMENT=negative'
[verdict]  learned the SENTIMENT= contract: True
```

The training loss falls 3.60 → 0.25 and token accuracy climbs 0.47 → 0.94; the base model answers
in a full sentence, the distilled student answers in the demonstrated contract. This is the same
`Project.distill(student, trainer_factory=...)` call the GPU-free `agentic_distillation.py`
makes — only the `trainer_factory` is real instead of injected.

## On-policy GRPO, for real — `agentic_grpo_real.py`

Drives agenttune's real `TrlAgenticGrpo` trainer (TRL's `GRPOTrainer`, LoRA) to fine-tune
SmolLM2-360M with reinforcement learning on the GPU. The reward pays out only for the correct
label in the exact `SENTIMENT=<label>` contract, so the policy has to actually classify — a
constant-label shortcut caps at ~1/3.

Output from an actual run (RTX PRO 6000):

```
[grpo]    90 logged steps  |  mean reward 0.031 -> 0.656
[probe]   Classify the sentiment ... Reply with exactly SENTIMENT=<label> ... Review: 'the delivery was late and support ignored me'
[before]  base policy samples -> ['The sentiment of the review is: negative.', ... , 'SENTIMENT=POSITIVE']
[after]   RL policy samples   -> ['SENTIMENT=negative', 'SENTIMENT=negative', 'SENTIMENT=negative', 'SENTIMENT=positive']
[verdict] reward improved: True  |  contract rate 25% -> 100%  |  correct-label rate 0% -> 75%
```

The base model answers inconsistently (prose, or a wrong-label `SENTIMENT=POSITIVE`); after GRPO
the policy answers in the contract and mostly with the correct label. Two honest caveats live in
the example's docstring: evaluate the *saved* adapter (the trainer's in-memory model generates
garbage), and reward correctness specifically or GRPO will hack a partial-credit format reward
into always answering "positive".

## Self-heal → DPO, for real — `self_heal_dpo_real.py`

Closes the self-heal loop. The self-healing case studies emit corrective preference data —
`{prompt, chosen, rejected}` where `chosen` is the fix and `rejected` is the failure — and stop.
This builds that data through the real spine (`build_dataset`) and runs a real Direct Preference
Optimization pass (TRL `DPOTrainer`, LoRA) on the GPU.

Output from an actual run (RTX PRO 6000):

```
[heal]    12 corrective preference rows (schema ['prompt', 'chosen', 'rejected']) — chosen=contract, rejected=prose
[dpo]     reward accuracy 0.50 -> 1.00  |  final reward margin 1.97
[before]  base policy -> 'The sentiment of the given text is negative.'
[after]   DPO policy  -> 'Negative'
[verdict] DPO ranks the corrective answer above the failure (accuracy -> 1.00, margin 1.97); greedy output moved off the prose form: True
```

DPO is a preference method, so its evidence is the reward accuracy/margin (the model now ranks the
corrective answer above the failure), not exact-string imitation — that's what SFT does above. The
generation shift (prose sentence → terse label) is corroborating, directional evidence.

## Reward-model training, for real — `reward_model_real.py`

GRPO/PPO optimise against a reward. Besides programmatic reward *functions* (`08_lifecycle_and_
rewards.py`), the other path is a learned *reward model* — a scalar scorer trained to rank a good
answer above a bad one. This trains one with TRL's `RewardTrainer` (LoRA) on the same
`{prompt, chosen, rejected}` data the heal loop emits, using `AutoModelForSequenceClassification`
with a single output.

Output from an actual run (RTX PRO 6000):

```
[data]    12 preference rows (schema ['prompt', 'chosen', 'rejected']) — chosen=contract, rejected=prose
[rm]      mean chosen-minus-rejected margin over 12 pairs +0.15 -> +4.14
[before]  reward(chosen)=+0.88  reward(rejected)=+0.70  chosen_wins=True
[after]   reward(chosen)=+2.14  reward(rejected)=-2.06  chosen_wins=True
[verdict] reward model ranks the contract answer above prose, margin +4.20 (was +0.18)
```

The score gap widens from ~0 to ~4: the untrained head barely separates the answers, the trained
model scores the correction well above the failure. (Ranking *accuracy* saturates — with a dozen
easy pairs a random head ranks them right by chance — so the margin is the honest signal.)

## RLOO, for real — `rloo_real.py`

The other on-policy RL family: like GRPO it samples several completions per prompt and reinforces
the higher-reward ones, but with a leave-one-out baseline instead of the group-normalised
advantage. Same task and reward as the GRPO example, so they're directly comparable — real TRL
`RLOOTrainer` (LoRA).

Output from an actual run (RTX PRO 6000):

```
[rloo]    90 logged steps  |  mean reward 0.031 -> 0.703
[before]  base policy samples -> ['The sentiment of the review is: negative.', ..., 'SENTIMENT=POSITIVE']
[after]   RL policy samples   -> ['SENTIMENT=negative', 'SENTIMENT=negative', 'SENTIMENT=negative', 'SENTIMENT=positive']
[verdict] reward improved: True  |  contract rate 25% -> 100%  |  correct-label rate 0% -> 75%
```

Reward 0.03 → 0.70, contract 25% → 100%, correct-label 0% → 75% — within noise of the GRPO run,
as expected for the same objective under a different advantage estimator. The same GRPO caveats
apply: evaluate the saved adapter, and reward correctness so the policy can't hack the format.

## BCO, for real — `bco_real.py`

Binary Classifier Optimization — learns from unpaired thumbs-up / thumbs-down labels on
individual completions (no `{chosen, rejected}` pairing needed), optimising a binary classifier
over the policy's log-ratios. Real `trl.experimental.bco.BCOTrainer` (LoRA). PPO and BCO live
under `trl.experimental` in TRL 1.6; the old top-level import raises AttributeError — the
experimental path is the working one.

Output from an actual run (RTX PRO 6000):

```
[data]    24 unpaired rows (12 desirable / 12 undesirable)
[bco]     implicit reward — desirable +1.86 vs undesirable -4.24 (margin +6.10)
[before]  base policy -> 'The sentiment of the given text is negative.'
[after]   BCO policy  -> 'SENTIMENT=negative'
[verdict] BCO scores desirable above undesirable: True; greedy output moved off prose: True
```

## PPO, for real — `ppo_real.py`

The classic actor-critic RLHF algorithm. Unlike GRPO/RLOO (which take a Python reward function),
PPO needs a real reward *model* — so the example first trains one (`RewardTrainer`), then runs the
full `trl.experimental.ppo.PPOTrainer` (LoRA) loop with a value model against it.

Honest scope: **PPO is the most unstable of these at this scale.** The example verifies the whole
pipeline executes end-to-end on the GPU — reward-model training, then policy/value/reward-model
optimization — but the reward over a short run is noisy and does not reliably climb (a hotter
learning rate drove it negative outright). It varies run to run, so no fixed numbers are quoted;
a representative run:

```
[ppo]     24 updates completed on GPU (RM-trained, actor-critic loop ran)
[reward]  reward-model score over updates: start ..., peak ..., end ...  (noisy — PPO is unstable at this scale)
[verdict] PPO pipeline ran end-to-end on GPU; ... a reliable gain needs many more episodes
```

This is deliberately not dressed up as a win: PPO is wired and runs, but converging it needs far
more episodes and tuning than a quick demo. The six other trainers above show clean learning;
PPO shows the pipeline running.

## Agentic-RAG GRPO, for real — `rag_training_real.py`

`RAGEnvironment`'s docstring advertises a one-liner — `create_agentic_trainer("grpo",
tools=env.tools, reward_funcs=[env.reward]).train()` — that, until now, **could not run**:

- `reward_funcs=[env.reward]` scored every completion a silent **0.0**. `env.reward` was a
  single-sample `reward(sample)`, but TRL calls reward functions with the batch signature
  `reward(completions=..., **kwargs) -> list[float]`; the mismatch raised a `TypeError` that
  `combine_rewards` swallowed to zero. Training would have run against a dead, all-zero reward.
- `tools=env.tools` went to TRL's native tool loop, which needs transformers ≥ 5.0 and can't
  build a schema from a `BaseTool` object — so on this stack it raised outright.

Two small library fixes close the gap: `RAGEnvironment.reward` is now **dual-mode** (accepts both
the single-sample form the tests use *and* the trainer's batch form, recovering `num_searches` /
`retrieved_chunk_ids` from the rollout trajectories); and the GRPO backend now **auto-builds a
tool-calling `rollout_func` from `tools`** — exactly as the RLOO backend already did — so `search`
is actually invoked during the rollout. (`combine_rewards` also stops silently zeroing a reward
that errors under both conventions — it now fails loud.)

This example runs the documented one-liner: a live `Qwen2.5-3B-Instruct` policy calls the
real `search` tool during the GRPO rollout, retrieves real BM25 passages, and is scored by the real
composite RAG reward (correctness + gold-chunk coverage + retrieval efficiency), with a real LoRA
GRPO step and the adapter saved and reloaded as a `PeftModel`. Before training it asserts the
**literal** `reward_funcs=[env.reward]` path — driven by `combine_rewards`, exactly as
`create_agentic_trainer` wires it — scores non-zero on a real rollout batch; `train()` itself wraps
`env.reward` in a thin recorder only to observe the tool loop from inside optimisation. During
`train()`, **16/16 sampled completions issued a real search and received a real, non-zero RAG
reward** (0.10–0.80 per prompt, matching the base measurement) — proof the tool loop fires inside
optimisation and the reward is no longer the swallowed zero.

**Honest scope (matches `ppo_real.py`):** the claim is that the *documented pipeline executes
end-to-end with a real, non-vacuous, search-driven reward*, not that it converges. On a small
corpus, a competent tool-caller and a deterministic BM25 retriever produce ~0 within-group reward
variance (every sampled completion issues the same query, retrieves the same chunk, answers the
same), so GRPO's advantage ≈ 0 and no weight movement is expected from a short run. The example
*measures and reports* that variance (`reward_std: 0.0`, `frac_reward_zero_std: 1.0`, `loss: 0.0`)
rather than cherry-picking a curve. The policy model must be tool-calling-capable — a tiny base
model that never emits a `search` call yields a degenerate, retrieval-free rollout (`SmolLM2-360M`
does exactly that; `Qwen2.5-3B` is why this one uses it). Needs `Qwen/Qwen2.5-3B-Instruct` cached.

---

## Docs → QA data-gen + difficulty curriculum, for real — `rag_datagen_real.py`

`agenttune.rag.datagen` advertises a P3 pipeline for making agentic-RAG training work on a user's
*own* corpus: `generate_qa_from_corpus(chunks, generator)` → `label_difficulty(qa, solver)` →
`balance_by_difficulty` / `sort_by_difficulty`. Both model-dependent callables (`generator`,
`solver`) were only ever exercised by **mocks** in `tests/rag/test_datagen.py`. This example runs
the whole pipeline with a **real `Qwen2.5-3B-Instruct`** on both ends and feeds the result into the
same `RAGEnvironment.reward` path that `rag_training_real.py` trains on.

The corpus is **deliberately built to straddle the knowledge boundary**, because that is what makes
the difficulty probe non-vacuous. `label_difficulty` probes the base model *with no document*: a
famous fact (Eiffel Tower, insulin) it answers unaided → *easy*; an **invented** fact (the "Kthonic
Protocol, ratified 3021") it cannot possibly know → *hard*. An all-famous corpus would label
everything *easy*, leaving `hard = 0` and an empty 1:1 balance — so both classes are seeded on
purpose. A verified run: a live Qwen reads each chunk to write a grounded Q&A (6 pairs; unparseable
generations are dropped, not faked), **6/6** gold chunks are retrieved in the real BM25 retriever's
top-3, and the blind base-model probe splits them cleanly **easy=3 / hard=3** (every famous fact
easy, every invented fact hard) → a real 3/3 balanced set and an easy→hard order. A generated *hard*
question then scores **non-zero through `combine_rewards([env.reward])`** on a real search rollout
(0.582, 4/4 searches), on a `Dataset` schema-identical to the one `create_agentic_trainer` consumes.

**Honest scope:** this proves the *mechanism runs for real over an LLM* — real generation, verified
grounding, a discriminating difficulty probe — and that its output feeds the real reward path. It
does **not** claim auto-generated QA *trains well*: "docs → synthetic QA" is an
explicit open bet (our own step, a parity requirement), and this example does not close it. Needs
`Qwen/Qwen2.5-3B-Instruct` cached.

---

## LangGraph orchestrator → real GRPOTrainer — `langgraph_orchestrator_real.py`

`AgentTuneGraph` composes rollout + LLM-judge nodes and documents `compile_rollout()` as drop-in
`GRPOTrainer`-compatible. The only wiring into a real trainer (`e2e_test_suite.py::T11`) couldn't run
here (hard-coded `/workspace` path, FP8 model) and scored training with a hand-written reward reading
`completion.get("final_reward", 0.5)` off *string* completions — so it always returned the constant
**0.5** and the graph's judges never scored the training completions. Root cause: `compile_rollout()`
exposed judge results only as *batch-level* scalars, never the per-completion vector a reward_func
needs. Two library fixes close it: the judge node keeps its per-trajectory scores and the rollout
exposes a per-completion `judge_rewards` vector; a new `compile_grpo_reward()` returns a
`reward_func(completions, **kwargs) -> list[float]` reading it.

The example runs the documented one-liner for real: a live `Qwen2.5-3B-Instruct` tool agent is the
rollout node, and **two divergent** `LLMJudge`s (a lenient answer-only rubric + a harsh
show-your-work rubric) are mean-aggregated per completion — so the multi-node composition is
observable, not a no-op (a preflight asserts the judges diverge). The composed judge reward is real
and non-constant (a `0.0`/`0.75`/`1.0` spread — verified `≥2 distinct` and `max>0` so the judge
genuinely parsed, not T11's `0.5` stub) and drives a real GRPO LoRA `train()` (one step even shows a
real non-zero `grad_norm`). **Honest scope:** a real, graph-composed judge reward drives a real
`train()`; convergence is not claimed, within-group variance reported as-measured. Needs
`Qwen/Qwen2.5-3B-Instruct` cached.

---

## Self-heal reward, for real — `self_heal_reward_real.py`

The self-heal generator (`decide.closed_loop.TrainingExampleGenerator`) turns a classified failure
into a `chosen`/`rejected` pair by *scoring* synthesized corrections with `TAC (Tool Argument
Correctness) + TER`. That reward never ran for real: `_process_single` built a **mock trajectory**
that shoved the raw completion string in as `tool_calls` (the code even says "In a real app, parse
comp_text as JSON"), and `_calculate_tac`'s string branch returns **1.0 for every completion** — so
TAC could not tell a well-formed correction from a malformed one. Two library fixes make it real:
`_process_single` parses the correction as JSON into a `{name, arguments}` call, and
`TrainingExampleGenerator(tool_schemas=...)` threads schemas so TAC validates the corrected arguments
(the parsed path engages only when schemas are supplied, so the schema-less legacy path is unchanged).

**Part 1** proves the reward now discriminates deterministically: a valid `lookup_order(order_id=123)`
scores `TAC=1.0`, a malformed `order_id="not-a-number"` scores `0.0` — where the old raw-string path
scored both `1.0`. **Part 2** runs the real generator end-to-end against a live `Qwen2.5-3B`: its
corrections (varied shapes like `{"function": "lookup_order", ...}`) are parsed and schema-scored
through the fixed reward path. **Honest scope:** the reward is proven real and discriminating; the
retrain-from-heal convergence is `self_heal_dpo_real.py`. Needs `Qwen/Qwen2.5-3B-Instruct` cached.

---

## lm-eval standardized benchmarks, for real — `lm_eval_real.py`

`agenttune.eval` exports `LMEvalConfig` / `LMEvalTask` / `LMEvalRunner` as the integration with
EleutherAI's `lm-eval` harness — but nothing in the repo ever invoked it (no test, no example shells
out to `lm_eval` and reads a score back). This runs it for real: `LMEvalRunner.evaluate_tasks` shells
out to the real `lm_eval --model hf` CLI on a live `Qwen/Qwen3-0.6B` over two standardized benchmarks
— **ARC-Challenge** and **WinoGrande** — from the local HF cache (offline), then parses the real
accuracies out of the results JSON and writes the combined summary. A verified run: `arc_challenge
acc=0.333`, `winogrande acc=0.6`, both files on disk, summary written. **Honest scope:** the
integration runs for real, but each task is capped at a small `--limit`, so these are **subset
estimates on a 0.6B model, not leaderboard numbers** — don't quote them as ARC/WinoGrande scores.
Needs `Qwen/Qwen3-0.6B` + the ARC-Challenge/WinoGrande datasets cached.

---

## Agentic inference

The nine examples above are the *trainer* surface. This one crosses into the *inference* surface:
the library's headline capability — agent **designs** (`ReActStrategy` + `run_episode`) and the
agentic eval metrics (`agentic_metrics`) — run over a **real model's** rollouts, not a scripted
`DemoRolloutEngine`.

### Real ReAct + agentic metrics — `agentic_strategy_real.py`

In the GPU-free studies the ReAct `policy` is a hand-written closure; the "model" is a stand-in.
Here a live `SmolLM2-360M` **is** the policy: at each step it's prompted with the running
`AgentState` (task + prior tool results) and its generation is parsed into the next action. The
task is arithmetic word problems, and the contrast is *why the tool framework exists* — both arms
on the same real model:

- **direct** — the model answers the arithmetic itself (no tool). A 360M model is bad at
  multi-step mental math.
- **react** — the same model, but it can call a `calc` tool: it emits the expression, the
  calculator computes it, and the model finishes with the number.

Output from an actual run (RTX PRO 6000):

```
[react]   one real trajectory ('A store sold 3 boxes of 12 apples and 5 loose apples. How many apples in total?'):
  CALL   {'name': 'calc', 'arguments': {'expression': '3*12+5'}, 'thought': 'TOOL: calc | ARGS: {"expression": "3*12+5"}'}
  RESULT '41'
  CALL   {'name': 'finish', 'arguments': {'answer': '41'}, 'thought': 'FINISH: 41'}
  RESULT '41'

[direct]  answer_match 1/6  |  agentic_metrics tac=1.00  ter=1.00  arr=0.00  scsr=1.00  rad=0.00  lcf=0.00
[react]   answer_match 5/6  |  agentic_metrics tac=1.00  ter=0.50  arr=0.00  scsr=1.00  rad=0.00  lcf=0.00
[verdict] a real model drove ReActStrategy/run_episode and agentic_metrics scored the REAL rollouts; the calc tool lifted answer_match 1/6 -> 5/6 on the same model
```

The headline is `answer_match`: the calc tool lifts the **same** model from 1/6 to 5/6 (it's not
6/6 — the model flubs one expression, reported honestly). The `agentic_metrics` are the honest
*characterisation* of the real rollouts, not a uniform higher-is-better scoreboard — note `ter`
(unique-outputs / total) is **1.00 for the do-nothing direct arm and 0.50 for the correct react
arm**, because the two-call react run repeats the answer as a tool output; `tac` (with tool
schemas passed to the evaluator) is 1.00 — the model's emitted arguments validate; and
`arr`/`lcf`/`scsr` stay flat because these short runs don't loop or error. The point that matters
is what it closes: the agentic surface now runs over a real model, not a stand-in.

### Live-LLM self-heal — `self_heal_llm_real.py`

The self-healing case studies run *detection* for real but **inject** the two stages that need a
language model — root-cause classification and corrective-example generation. This runs those two
stages for real, through the **unmodified** production path: `FailureClassifier.classify_batch` and
`TrainingExampleGenerator.generate_batch` make real `litellm.acompletion` calls, wrapped by the
real `as_sync_classifier` / `as_sync_generator` adapters and driven by the real `SelfHealLoop`.

Those classes call `litellm(model=…, api_base=…)`; in production those two strings point at a
hosted or on-prem endpoint. This sandbox is offline (no keys, no vllm/llama.cpp), so the example
stands up a tiny OpenAI-compatible server backed by a local `Qwen2.5-3B-Instruct` and points
litellm at it — the exact on-prem shape the case studies pitch. The server is **real
infrastructure running a real model**, not a mock of the library; deployment changes the two
strings and nothing else in the heal path moves.

Output from an actual run (RTX PRO 6000):

```
[detect]  2 failures in hand (real FailureDetector output shape)
[classify] real litellm -> Qwen root-cause calls (greedy, reproducible):
           t1: loop_collapse  (conf 0.95)  — The agent repeatedly searched for 'refund policy' without any progress
           t2: wrong_tool  (conf 1.0)  — The agent attempted to use 'send_email' which is not suitable for look
[generate] real litellm -> corrective preference rows (one representative sampled run):
           chosen  : '{"action": "search_web", "query": "how to contact customer support"}'
           rejected: "search_web(query='refund policy')"
           chosen  : '{"action": "lookup_order"}'
           rejected: "send_email(to='ops', body='what is order 123')"
[retrain] loop.trainer_factory fired: trained=True  train_result={'train_loss': 0.669}  (plumbing closes; the converged before/after is in self_heal_dpo_real.py)
[verdict] real LLM ran the classify+generate stages (2 classified, 2 generated, 2 rows) and the full detect->classify->generate->retrain loop executed end-to-end — no stage mocked
```

The real model classifies both failures correctly (`loop_collapse` 0.95, `wrong_tool` 1.0) and
writes a corrective action for each; the `rejected` side is the original failing turn. Honesty
notes: classification is greedy and reproduces; the `chosen` corrections are sampled at
temperature 0.7 (the generator hard-codes it), so they're one representative run; the row *prompt*
is a library placeholder, so the `chosen` / `rejected` pair is the real signal. The retrain step
uses the `SelfHealLoop.trainer_factory` hook to prove the loop closes end-to-end (real
`DPOTrainer`) — the converged before/after belongs to `self_heal_dpo_real.py`, not here.

---

## DECIDE closed loop — retrain → gate → deploy, for real — `closed_loop_real.py`

`FullClosedLoop` is the flagship DECIDE orchestrator: production failures → detect/classify/generate
→ buffer → trigger → **background retrain → deployment gate → deploy**. Every model boundary is an
injected callable so the loop is unit-testable GPU-free — and in every test/notebook those
boundaries are **stubs** (`_stub_retrain_job` returns `{"model_path": "/tmp/stub_model"}`, verdict
runners are fakes). So the load-bearing chain the loop exists for — *retrain → gate → deploy a real
adapter* — had never actually run end-to-end. This runs it for real.

Output from an actual run (RTX PRO 6000):

```
[audit]   wrote 6 successful pipelines -> audit.jsonl (the gate's ground-truth test set)
[baseline] loading base verdict runner (the currently-deployed model)...
[gate]    built test set: 6 cases from the audit log
[buffer]  submitted 6 real TrainingExamples
[tick]    trigger fired=True reason='total_failures_exceeded' -> background retrain
[retrain] success=True
[gate]    task accuracy  old=0.33  new=1.00  delta=+0.67  over 6 cases
[gate]    decision: approved=True  (approved: no regression on task or trajectory)
[apply]   deployed
[deploy]  config.yaml default_model -> 'retrained_adapter' (points at the new adapter: True)
[deploy]  adapter files on disk: ['adapter_config.json', 'adapter_model.safetensors']
[deploy]  reloaded the deployed adapter as PeftModelForCausalLM: True
[verdict] retrain→gate→deploy ran end-to-end; gate approved a REAL accuracy gain (0.33→1.00) and a real adapter is deployed+loadable: True
```

What is real: `retrain_job` runs `build_retrainer(...)` → a real TRL `DPOTrainer` (LoRA on
SmolLM2-360M) → `save_model` → a real adapter on disk (asserted present before it returns); the
**gate genuinely decides** on real task accuracy — the base model vs the retrained model are scored
by real verdict runners over a non-empty test set the gate reconstructs from a real Decide audit log
(old 0.33 → new 1.00, so it approves on merit, *not* a default-approve on an empty set); `apply`
deploys via the real `deploy_trained_model` bridge, which rewrites a scratch `config.yaml`; and the
example then **reads `config.yaml` back and reloads the deployed adapter as a real `PeftModel`** —
the literal proof a real adapter was produced *and* deployed.

Honest scope: this focuses on the never-real **retrain → gate → deploy** chain. Live-LLM
classify/generate is already proven in `self_heal_llm_real.py`, so here real `TrainingExample`s are submitted straight
into the loop's real buffer (`runner.submit`) rather than re-standing-up a local LLM — an explicit
boundary, not a hidden one. The gate runs on **task accuracy** (its primary signal); the optional
trajectory signal (which calls litellm) is left off. The "verdict" is contract-*format* adherence
(emitting the `SENTIMENT=` form the DPO step teaches) — a real behavioural shift the retrain
produces; exact-label correctness is a separate, harder axis and is not claimed. A fully-live-Path-A
variant is a straightforward extension on request.

---

## Real components & transport

The examples above make the *model* real. These make the parts that were only ever exercised
*in-process* real: the service run as an actual server, and the pluggable memory/environment
pillars driven by real backends instead of the offline stand-ins. No model claim is made here —
these close transport/component gaps honestly.

### Service as a real server — `service_real.py`

`create_app()` (the FastAPI operator surface) is only ever tested through
`fastapi.testclient.TestClient`, which calls the ASGI app in-process — the app is never bound to a
socket, the operator UI is never served, no real HTTP or WebSocket client connects. This stands it
up under a real `uvicorn` server on a real port and drives **every** endpoint over real `httpx`,
connects a **real** `websockets` client to the event stream, and fetches the operator UI over HTTP.

Output from an actual run (no GPU):

```
[server]  uvicorn up on http://127.0.0.1:36057  (server.started=True)
[ui]      GET /  ->  200  (16676 bytes, html=True, operator-shell=True)
[project] POST /projects  ->  730c45c75313
[collect] POST /collect_rollout  ->  3 trajectories
[eval]    POST /evaluate  ->  n=3  metrics: tac=0.00, ter=1.00, arr=0.00, scsr=1.00, rad=0.00, lcf=0.00
[distill] POST /distill  ->  3 rows, status='wired-runs-on-gpu'
[train]   POST /train  ->  status='wired-runs-on-gpu'
[strat]   run_strategy(react       ) -> answer='42' (8 events)
[strat]   run_strategy(plan_execute) -> answer='42' (8 events)
[strat]   run_strategy(reflexion   ) -> answer='42' (4 events)
[strat]   run_strategy(tot         ) -> answer='42' (5 events)
[strat]   run_strategy(memory      ) -> answer='42' (8 events)
[memory]  memory_demo(vector) -> recalled ['cat cat', 'dog dog']
[memory]  memory_demo(graph ) -> recalled ['Bob']
[heal]    POST /heal  ->  1 failure(s): ['loop_collapse']
[heal+]   POST /heal_loop  ->  classified=1 generated=1 rows=1 status='demo-stages'
[read]    GET /events -> 20 events   GET /trajectories -> 9
[ws]      real websockets client streamed 20 live events while the REST calls ran
[verdict] server real, UI served, 20 events over REST and 20 over WebSocket, all 5 strategies answered: True
[server]  shutdown requested, server thread stopped=True
```

Honest scope: the *server, transport (REST + WebSocket), and operator UI* are what was never proven
and are now proven real. The *endpoints* run the library's documented GPU-free control path — real
spine machinery (real `agentic_metrics`, real `run_episode` over all five strategies, real
`VectorMemory`/`GraphMemory`, real `FailureDetector` + `SelfHealLoop`) driven by model-free
policies, exactly as `service/app.py`'s docstring states. The real *model* execution these controls
represent is the rest of this suite. Standing the server up also surfaced and fixed a real bug: the
WebSocket handler never detected an *idle* client disconnect (it only sent, never received), so
disconnected clients leaked until shutdown and shutdown logged a spurious traceback — the handler
now polls for the disconnect and exits cleanly.

### VectorMemory with a real embedder — `vector_memory_real.py`

`VectorMemory(embed=...)` ranks items by cosine similarity of their embeddings. The GPU-free study
and the service pass a **bag-of-words** `embed` (a fixed-vocab count vector) so they run offline —
which only matches on *shared words*, not meaning. This plugs in a real sentence-transformer
(`all-MiniLM-L6-v2`, 384-d) and shows recall by *meaning*: a query that shares no salient word with
the correct note still retrieves it, while a lexical-overlap baseline picks the wrong note.

Output from an actual run (no GPU):

```
[embed]   sentence-transformers/all-MiniLM-L6-v2  (384-d, real embeddings)
[write]   stored 4 support notes
[query]   'My machine is completely dead and nothing happens when I press the power button.'
[recall]  top-1 (real embeddings) -> "Customer's laptop will not switch on at all after the latest update."
[lexical] Jaccard top-1 -> 'User cannot log in; the password reset email never arrives.'  (overlap with correct note = 0.04)
[verdict] real embedder retrieves the paraphrase: True; lexical baseline gets it: False -> semantic recall is doing real work: True
```

The real embedder retrieves the paraphrase (0.04 lexical overlap with the correct note); the
lexical baseline cannot. Same `VectorMemory` class and `write`/`read` calls — only the `embed`
function is real.

### OpenEnvHarness against a real OpenEnv env — `openenv_harness_real.py`

`OpenEnvHarness` wraps a gym-like OpenEnv `Environment` behind the `Harness` contract. Its unit
tests drive it with a hand-rolled `FakeEnv` (by design — the harness must import even if
`openenv` weren't installed). This builds a **genuine** OpenEnv env: a real `openenv.core.Environment`
subclass with real `openenv` `Observation`/`Action` Pydantic types and a real `openenv` `Rubric`
computing the reward — an interactive higher/lower guessing game — and drives it through the full
spine (`run_episode` + `run_conformance` + `replay`).

Output from an actual run (no GPU):

```
[env]     real openenv Environment  (HigherLowerEnv, secret=73, rubric=CorrectGuessRubric)
[reset]   obs -> "I'm thinking of a number between 1 and 100. Guess it."
[step]    guess 73 -> obs='Correct!' reward=1.0 done=True  (reward from the real Rubric)
[episode] agent guesses: [50, 75, 62, 68, 71, 73]
[episode] solved in 6 guesses (log tier=light, 25 events): True
[conform] run_conformance -> passed=True, drift=[]
[replay]  re-executed 2 tool calls through the real env -> ['higher', 'Correct!']
[verdict] real OpenEnv env driven through run_episode + conformance + replay, real rubric reward: True
```

A binary-search agent reads the env's real observations to solve it in 6 guesses; the reward comes
from the real `Rubric`; conformance passes and a recorded log replays deterministically through a
fresh real env. Honest scope: everything OpenEnv here is real (base class, `Observation`/`Action`,
`Rubric`); this is the *in-process* env — a fully *remote* OpenEnv env (HTTP `SyncEnvClient` ↔ a
containerised server) is the same wrapper over a network transport. The agent policy is a
deterministic binary search — what's proven here is the harness ↔ real-OpenEnv integration, not
model quality (that lives in the other examples above).
