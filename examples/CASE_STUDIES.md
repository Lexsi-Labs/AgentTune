# AgentTune Spine — Case Studies

Nineteen runnable case studies for the `agenttune.agentic` integration spine (8 core + 11
industry-domain), covering every value path in the pitch — flat under `examples/`, no
subfolders. Scripts are being converted from GPU-free/injected stand-ins to real open-source
models + real data, one at a time; each script's own docstring says which state it's in. The
output blocks below are captured **verbatim** from an actual run, not hand-authored, and are
refreshed as each script is converted.

```bash
python examples/build_and_train.py
python examples/agentic_distillation.py
python examples/self_healing.py
python examples/memory_recall.py
python examples/strategy_comparison.py
python examples/conformance_and_replay.py
python examples/end_to_end_improve.py
python examples/lifecycle_and_rewards.py
```

Coverage is audited: **[COVERAGE.md](COVERAGE.md)** maps every public capability to the example
that exercises it (all 11 `Project` lifecycle methods, all five agent designs, all three memory
drivers, conformance/replay, self-heal, and the GRPO reward pillar).

Each script uses the deterministic `DemoRolloutEngine` / injected callables so a genuine
trajectory flows through the *real* lifecycle machinery with no GPU.

---

## 1. Build an agent, then train it — `build_and_train.py` / `build_and_train.ipynb`

**Converted to a real model** (pilot for the rest of this file — see the note at the top).
A live `SmolLM2-360M-Instruct` drives the ReAct policy (BUILD) and produces the full-tier
rollout via the real `TransformersRolloutEngine` (COLLECT) → evaluate with real metrics →
assemble the exact SFT/GRPO dataset the trainer consumes. Output captured verbatim from an
actual GPU run:

```
[gpu]     NVIDIA L40S  (cuda: True)
[model]   HuggingFaceTB/SmolLM2-360M-Instruct loaded
[build]   one episode -> 9 events, tier=light
[eval]    programmatic metrics -> {'tac': 0.0, 'ter': 0.5, 'arr': 0.0, 'scsr': 1.0, 'rad': 0.0, 'lcf': 0}
[collect] 2 rollouts, tiers=['full', 'full']
[train]   SFT dataset -> 2 rows, row0 keys=['messages', 'segment_weights', 'loss_mask']
[train]   row0 first message role='assistant'
[stream]  lifecycle stages -> ['infer', 'infer', 'collect_rollout', 'collect_rollout']
```

## 2. Agentic distillation — `agentic_distillation.py`

The headline path: compress an agent DESIGN into a `<3B` student by SFT on the teacher's own
`full`-tier trajectories (behavior cloning, not weight-KD). The `Project` stays GPU-free; the
injected `trainer_factory` is where the real `.train()` would run on a GPU.

```
[teacher] collected 3 full-tier trajectories
[teacher] behavior-cloning dataset -> 3 SFT rows (schema: ['messages'])
[distill] trainer received model='student-1.5B', n_rows=3
[distill] result -> {'status': 'would-train-on-gpu', 'student': 'student-1.5B', 'n_rows': 3}
```

## 3. Closed-loop self-healing — `self_healing.py`

Detect a looping agent with the **real** `FailureDetector`, then run the full loop
(classify → generate corrective preference data). The litellm-bound stages are injected as
deterministic stand-ins; the dataset is real `{prompt, chosen, rejected}` preference rows.

```
[seed]    injected 1 looping trajectory (1 total)
[detect]  FailureDetector found 1 failure(s): ['loop_collapse']
[classify] root causes -> ['loop_collapse']
[dataset]  1 corrective preference row(s); keys=['prompt', 'chosen', 'rejected']
[summary] {n_failures: 1, n_classified: 1, n_dataset_rows: 1, trained: False}
```

## 4. Memory design: semantic vs relational — `memory_recall.py`

Why an agentic-RAG app picks one memory driver over another. Vector recalls what is *similar*
(cosine); Graph recalls what is *connected* (edge traversal) — note `Dave` is present but
unconnected, so the graph never surfaces it.

```
[vector]  query 'cat' -> ['cat cat', 'dog dog']   (semantic: nearest by cosine)
[graph]   query 'Alice' -> ['Bob']   (relational: Alice's neighbours, not 'Dave')
```

## 5. Compare all five agent designs — `strategy_comparison.py`

The agent-design pillar's five strategies run behind one `run_episode` / `EventLog` interface,
so they are interchangeable. Each reaches the answer with a canned, model-free policy.

```
design          events   answer
-------------------------------
react                8       42
plan_execute         8       42
reflexion            4       42
tot                  5       42
memory               8       42
```

## 6. Harness conformance and replay — `conformance_and_replay.py`

A trainable env needs two guarantees: it behaves as its declared capabilities claim
(`run_conformance`), and a recorded log's tool calls re-execute deterministically (`replay`).

```
[conformance] passed=True  drift=none
[record]      8 events, tool invoked 1x
[replay]      re-executed -> 7 events, tool now invoked 2x total
[replay]      deterministic: yes
```

## 7. End to end — collect, evaluate, find a failure, generate the fix — `end_to_end_improve.py`

The whole "improve an agent" loop in one script: collected trajectories feed both evaluation
and failure-recovery through one `EventLog`.

```
[collect]  2 healthy trajectories, tiers=['full', 'full']
[evaluate] mean metrics -> {'tac': 0.0, 'ter': 1.0, 'arr': 0.0, 'scsr': 1.0, 'rad': 0.0, 'lcf': 0.0}
[detect]   FailureDetector -> 1 failure(s): ['loop_collapse']
[heal]     corrective dataset -> 1 row(s) ['prompt', 'chosen', 'rejected']; ready to retrain the agent
```

## 8. Lifecycle tail + reward shaping — `lifecycle_and_rewards.py`

Closes the loop on the rest of the lifecycle (`collect`, `evaluate_agentic`) and the reward
pillar a GRPO run consumes: a spine-native `answer_match` off an `EventLog`, plus a weighted
reward built from `REWARD_REGISTRY` that scores candidate completions the way the trainer would.

```
[collect]  2 episodes collected, tiers=['light', 'light']
[evaluate] n=1  metrics -> {'tac': 0.0, 'ter': 1.0, 'arr': 0.0, 'scsr': 1.0, 'rad': 0.0, 'lcf': 0.0}
[reward]   answer_match(log, '42') -> 1.0
[registry] 16 built-in reward funcs, e.g. ['answer_correctness_reward', 'answer_format_reward', 'computation_reward']
[shape]    weighted reward over 2 candidates -> [1.14, 0.0] (correct+concise beats wrong+long)
```

---

## By domain

Same spine APIs, different industries. All flat under `examples/` (no per-domain subfolder).

### BFSI — `fraud_triage_end_to_end.py` · `kyc_agentic_distillation.py` · `compliance_graph_memory.py`

- **Fraud-triage agent, end to end**: a card-fraud triage agent inspects a flagged transaction
  (velocity → geo → decision); trajectories are evaluated, and a degraded agent that loops on one
  check is caught by the real `FailureDetector` and turned into corrective training data.
- **KYC distillation to an on-prem model**: banks often can't send PII to a hosted model. A
  strong teacher agent produces KYC-triage trajectories; agentic distillation behavior-clones
  them into a small (<3B) student that runs in the bank's own VPC.
- **Compliance QA over a regulation graph**: `GraphMemory` recalls what is *connected* (a
  control implementing a regulation), not what merely looks similar — an unrelated KYC control
  and an unlinked node never surface.

See also the DECIDE-pipeline KYC example at [`examples/bfsi_kyc.py`](bfsi_kyc.py).

### Healthcare — `clinical_triage_end_to_end.py` · `medical_coding_distillation.py`

Illustrative and deterministic — **not** medical devices or certified systems.

- **Clinical triage, end to end**: a nurse-triage agent reads a complaint, checks vitals and
  history, and assigns acuity (ESI 1–5); a looping agent is caught by the real `FailureDetector`.
- **Medical-coding distillation to an on-prem model**: clinical text can't leave the hospital
  network — a teacher agent codes encounters, distilled into a small (<3B) on-prem student.

### Legal — `contract_review_end_to_end.py` · `precedent_graph_memory.py`

Illustrative and deterministic — **not** legal advice.

- **Contract review, end to end**: an agent reads a clause, checks obligations/governing law,
  and decides ACCEPT / NEGOTIATE / REJECT; a reviewer that loops is caught by `FailureDetector`.
- **Precedent retrieval over a citation graph**: a citation graph recalls what a case actually
  *relies on*, never surfacing an unrelated matter just because the language is similar.

### Retail — `order_issue_triage.py` · `catalog_enrichment_distillation.py`

- **Order-issue triage, evaluated**: routes a shopper's message (WISMO / REFUND / REPLACEMENT)
  and scores routing accuracy with `Project.evaluate`.
- **Catalog-enrichment distillation**: distills a strong teacher into a small (<3B) student that
  runs cheaply at catalog scale.

### General — `competitor_research.py` · `support_ticket_triage.py`

- **Competitor-research agent**: the smallest useful ReAct loop — search → compare → brief.
- **Support-ticket triage, evaluated**: routes tickets (BILLING / BUG / ESCALATE) and scores
  accuracy with `Project.evaluate`.

## Beyond GPU-free

- **On-policy GRPO** — `Project.train(fmt='grpo', rollout_engine=...)` wires `create_rollout_fn`
  as the trainer's `rollout_func`; rollout runs inside the trainer with a real model on a GPU.
- **Real SFT/distill execution** — pass a real `trainer_factory` (e.g. AgentTune's
  `TRLSFTTrainer`); the dataset assembly shown above is identical, only `.train()` moves to GPU.
- **Full self-heal** — swap the injected `demo_classifier`/`demo_generator` for
  `as_sync_classifier(FailureClassifier(...))` / `as_sync_generator(TrainingExampleGenerator(...))`,
  which call litellm at their own site.

See the [spine reference](../src/agenttune/agentic/README.md) and
[`REAL_EXAMPLES.md`](REAL_EXAMPLES.md) for the same ground covered with real models on a GPU.
