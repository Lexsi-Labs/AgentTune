# Agentic RAG on AgentTune

An agentic retrieval-augmented-generation (RAG) training pipeline built as a
use-case package on top of **AgentTune**. This package trains a model to
*search a document corpus with a tool*, and *learn from the reward signal*
whether it searched, how well it used what it found, and whether its final
answer was correct via GRPO with LoRA.

Note: zero edits to AgentTune's core framework code (`agentic/`, `backends/`, `core/`, `decide/`). Everything here is new code under `agenttune/rag/`, consumed only through
AgentTune's existing public API (`create_agentic_trainer`, `create_rollout_fn`,
`combine_rewards`, `LLMJudge`, `HFLoader`).

**Scope note:** this package trains plain GRPO/LoRA on a RAG task. It does **not** use AgentTune's self-healing closed-loop retraining system (`decide/closed_loop/`) — that's a separate subsystem; nothing here imports it.

---

## 1. Architecture

- **Retrieval is backend-agnostic.** A `SearchBackend` Protocol
  (`retrieval/base.py`) lets you swap retrieval implementations. It ships two:
  `SQLiteFTSBackend` (lexical, stdlib `sqlite3` + FTS5's built-in
  `bm25()` ranking — zero new dependency) and `ChromaBackend` (vector, via the
  `chromadb` client's own embedding + search).
- **Reuses existing factories.** Tools, rewards, and dataset loading consume
  AgentTune's existing public factory functions rather than reimplementing
  training/rollout machinery. Chunking wraps `langchain-text-splitters`.
  QA scoring wraps HuggingFace `evaluate.load("squad")` rather than
  hand-rolling exact-match/F1 normalization.
- **Self-contained package pattern.** This mirrors the existing
  `decide/closed_loop/` package: new code lives entirely under one directory,
  imported only through public factory functions, so it can be reviewed,
  tested, and removed as a unit.

### Added it as a dependency

`pyproject.toml` has a `[project.optional-dependencies]`
table: `rag = ["chromadb>=0.5.0", "langchain-text-splitters>=0.3.0"]`. 

---

## 2. Directory tree

```
src/agenttune/rag/
├── README.md      # report 
├── trajectory_utils.py       # extract_tool_calls/is_tool_step/extract_question_text —
│                              # normalizes Step.action's actual runtime shape (§7.5)
├── retrieval/
│   ├── base.py                # SearchBackend Protocol + SearchResult TypedDict
│   ├── chunker.py              # chunk_text() — wraps langchain-text-splitters
│   ├── corpus_loader.py         # CorpusDocument, load/chunk from HF datasets, build_index()
│   ├── sqlite_fts.py             # SQLiteFTSBackend (stdlib sqlite3 + FTS5 bm25())
│   └── chroma_backend.py          # ChromaBackend (pure vector, chromadb's own similarity search)
├── tools/
│   ├── search_corpus.py       # SearchCorpusTool(BaseTool)
│   └── read_document.py       # ReadDocumentTool(BaseTool)
├── rewards/
│   ├── qa_metrics.py           # exact_match_score/f1_score wrapping evaluate.load("squad");
│   │                            # extract_answer_tag (bespoke <answer> tag parser)
│   ├── phase1_rewards.py        # format/search_usage/correctness reward fns + get_training_reward()
│   └── judge_eval.py             # build_groq_judge(), score_groundedness() — EVAL-ONLY, not a training reward
├── data/
│   └── hotpotqa.py             # load_hotpotqa_splits, build_corpus_from_hotpotqa, to_grpo_dataset,
│                                # DEFAULT_SYSTEM_PROMPT
└── scripts/    # every file here is a `python -m agenttune.rag.scripts.<name>` CLI entry
    │           # point (see each file's own docstring for the exact usage); build_index,
    │           # verify_masking, and train_grpo are invoked directly from a real example
    │           # notebook (examples/USECASES/15)
    ├── build_index.py           # builds SQLite/Chroma index from HotpotQA corpus
    ├── build_index_from_docs.py  # builds an index from an arbitrary doc corpus (not HotpotQA)
    ├── verify_masking.py         # empirical proof that env_mask excludes retrieved text
    ├── train_grpo.py              # GRPO LoRA training driver (+ --ablation unmasked)
    ├── evaluate.py                 # baseline vs trained, model-size + backend comparison, judge scoring
    │
    │   # FinDER/10-K legal-domain pipeline (examples/USECASES/15_financial_rag_agent.ipynb):
    ├── build_index_finder.py      # FinDER + 10-K filings -> ticker-disjoint FTS5 index;
    │                                # imported directly by USECASES/15 (`import main as
    │                                # build_index_finder_main`), not just CLI-invoked
    ├── build_index_mixed.py        # combined FinDER + CUAD index (shares build_index_finder's
    │                                # gold-chunk assignment logic)
    │
    │   # CUAD contract-QA pipeline (fetch -> build dataset -> index -> eval), chained
    │   # via each script's own docstring, not by direct Python import:
    ├── fetch_cuad_official.py      # downloads official CUAD train/test JSON
    ├── build_cuad_dataset.py        # CUAD JSON -> GRPO-ready dataset, consumed by
    │                                # eval_generated_dataset.py
    ├── eval_generated_dataset.py     # scores a generated CUAD dataset for quality
    │
    │   # M1/M2 curriculum eval pipeline (memory/m2_decisions.py, tools/xml_tool_parser.py
    │   # document the M2 condition; tests/test_curriculum_m2.py references this file by name):
    ├── eval_m1_zeroshot.py          # zero-shot eval harness, m1/m2 curriculum conditions
    ├── run_controlled_eval.py        # runs eval_m1_zeroshot.py per model config via subprocess
    ├── run_t1_t4_probe.py            # smaller T1-T4 probe variant of the curriculum eval
    │
    ├── analyze_run.py               # summarizes a `rag_experiments/runs/<run>/` directory
    └── make_e1_graphs.py             # plots for the E1 experiment set

tests/rag/   — 38 tests total:
  test_hotpotqa_loader.py        (5)  — dataset loading/determinism
  test_masking_verification.py   (6)  — pure-logic overlap-scoring helpers (CPU-only;
                                          the real GPU masking proof is @pytest.mark.gpu)
  test_retrieval_backends.py     (8)  — SQLiteFTSBackend + ChromaBackend indexing/search
  test_rewards.py                (14) — EM/F1 metrics, answer-tag extraction, all 3 reward fns
  test_tools.py                  (5)  — SearchCorpusTool / ReadDocumentTool schemas + execution
```

Run artifacts land under `rag_experiments/` (gitignored,
created at runtime) — see §6.4 for the exact layout.

---

## 3. Installation

```bash
cd AgentTune/
pip install -e ".[rag]"     # pulls chromadb + langchain-text-splitters
```

On a fresh box, run the test suite first to confirm the install is sane:

```bash
pytest tests/rag/ -v          # 37 tests, no GPU needed
pytest tests/rag/ -v -m gpu   # +1 test, needs a real GPU + model download
```

---

## 4. Masking

**tokens the model retrieves via search never contribute to the RL loss** — only tokens the model itself generates do.

`scripts/verify_masking.py` verifies this by building `[SearchCorpusTool, ReadDocumentTool]`, calls the real `create_rollout_fn(rollout_backend="transformers", model_path=..., tools=
tools, system_prompt=DEFAULT_SYSTEM_PROMPT, max_steps=4)` on a handful of
questions about facts the model can't know from pretraining, and for each
resulting trajectory:

- decodes the `env_mask==0` span and checks it substantially reproduces the
  actual tool output text (proves the masked span *is* retrieved content);
- decodes the `env_mask==1` span and checks it substantially reproduces the
  **model's own generated text** — not "differs from the tool text". A model
  legitimately paraphrasing retrieved facts in its answer naturally shares
  vocabulary with what it retrieved, so "differs from tool text" is the wrong
  check and produces false failures. Compare against the model's own
  generation instead.

Uses only the public `create_rollout_fn` entry point — no access to
`rollout_factory.py` internals.

**Run this before any training run**, especially on a new model:

```bash
python -m agenttune.rag.scripts.verify_masking \
    --model_path Qwen/Qwen3-0.6B --backend sqlite \
    --index_dir rag_experiments/indexes/sqlite
```

Passing output looks like `report.json` with `spot_check_passed: true` for
every question and `any_search_happened: true` — if a model never searches,
this can't prove anything about masking, it just means the model didn't use
the tool.

### Proving masking actually matters (the `--ablation unmasked` flag)

`scripts/train_grpo.py --ablation unmasked` trains a *second* run where
`env_mask` is forced to all-ones (every token, including retrieved text,
contributes to the loss) so you can compare training curves against the
masked default and see the difference masking makes. Implemented entirely in
the script's own process: it builds the real rollout fn once, wraps its
*output* to overwrite `env_mask`, and hands that wrapped callable to a second
`create_rollout_fn(custom_rollout_fn=wrapped)` call (a real, documented escape
hatch in `rollout_factory.py`) — `rollout_factory.py` itself is never touched.

---

## 5. Rewards

Training reward = weighted sum via `get_training_reward()`
(`rewards/phase1_rewards.py`), default weights **format=0.1, search_usage=0.2,
correctness=0.7**, composed through AgentTune's existing `combine_rewards`
(`agentic/rewards/composite.py`):

| Reward | Implementation | What it measures |
|---|---|---|
| `format_reward` | re-export of AgentTune's builtin `use_case.format_reward` | +0.1 if the completion wraps its final answer in `<answer>...</answer>` |
| `search_usage_reward` | wraps AgentTune's builtin `search_grounding_reward` | tiered by tool-call count: 0 calls→0.0, 1→0.2, 2→0.3, 3+→0.4 |
| `rag_correctness_reward` | new, this package | token-F1 between the `<answer>` tag content and the dataset's `gold_answer` column, via `evaluate.load("squad")` (same normalization HotpotQA's own official eval uses) |

Override weights: `get_training_reward(weights={"format": ..., "search_usage":
..., "correctness": ...})`.

`rag_correctness_reward` returns `[0.0] * len(completions)` if `gold_answer`
isn't present in the batch — a defensive default, not a silent failure to
special-case around.

**Calling convention** (matters if we write a new reward fn or debug one):
`combine_rewards`'s composed callable is invoked as `fn(completions=...,
**kwargs)`, where `kwargs` includes `prompts`, plus whatever extra dataset
columns TRL forwards (here: `gold_answer`) and `tool_call_counts` (computed by
`rollout_factory.py` from `trajectory.metadata["tool_call_count"]` and passed
automatically). Every reward fn in this package accepts `(prompts,
completions, **kwargs)` to match.

### Eval-only: LLM-judge groundedness 

`rewards/judge_eval.py`'s `score_groundedness()` (via `build_groq_judge()`)
scores whether the final answer is grounded in retrieved passages and matches
the gold answer, using a Groq-hosted `llama-3.3-70b-versatile` judge. It's
kept out of GRPO's hot training loop — token-F1 + search-usage
already give a dense, free, deterministic training signal, and an API call
per rollout inside GRPO risks rate limits/latency for little added signal.
Only invoked from `scripts/evaluate.py --use_judge` on a small held-out set.
Requires `GROQ_API_KEY` in the environment.

---

## 6. Running it

### 6.1 Model choice

`Qwen3-0.6B` calls the
tool reliably (5/5 on `verify_masking`'s test questions). Run `verify_masking` (§4) on any new
model before a real training run.

### 6.2 Quick smoke test (few minutes, confirms the loop works end to end)

```bash
python -m agenttune.rag.scripts.build_index \
    --backend sqlite --train_size 4 --eval_size 2 \
    --output_dir rag_experiments/indexes/sqlite

python -m agenttune.rag.scripts.verify_masking \
    --model_path Qwen/Qwen3-0.6B --backend sqlite \
    --index_dir rag_experiments/indexes/sqlite

python -m agenttune.rag.scripts.train_grpo \
    --model Qwen/Qwen3-0.6B --backend sqlite \
    --index_dir rag_experiments/indexes/sqlite \
    --train_size 4 --eval_size 2 \
    --max_steps 4 --gradient_accumulation_steps 2 --num_generations 2 \
    --per_device_train_batch_size 1 --max_completion_length 256 --max_rollout_steps 2 \
    --device_map single_gpu \
    --output_dir rag_experiments/runs/smoke_test
```

### 6.3 Full training run

```bash
python -m agenttune.rag.scripts.build_index \
    --backend sqlite --train_size 2000 --eval_size 200 \
    --output_dir rag_experiments/indexes/sqlite   # chunk_size=512 default — don't shrink this, see §7.3

python -m agenttune.rag.scripts.train_grpo \
    --model Qwen/Qwen3-0.6B --backend sqlite \
    --index_dir rag_experiments/indexes/sqlite \
    --train_size 2000 --eval_size 200 \
    --max_steps 150 --gradient_accumulation_steps 8 --num_generations 4 \
    --per_device_train_batch_size 2 --max_completion_length 1024 --max_rollout_steps 6 \
    --device_map single_gpu \
    --output_dir rag_experiments/runs/full_run

python -m agenttune.rag.scripts.evaluate \
    --checkpoints Qwen/Qwen3-0.6B "Qwen/Qwen3-0.6B::rag_experiments/runs/full_run" \
    --backends sqlite --use_judge
```

### `train_grpo.py` CLI reference

| Flag | Default | Notes |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | see §6.1 for model choice guidance |
| `--backend` | `sqlite` | or `chroma` |
| `--index_dir` | *required* | |
| `--train_size` / `--eval_size` | 2000 / 200 | HotpotQA rows |
| `--lora_r` / `--lora_alpha` | 16 / 32 | LoRA on `target_modules="all-linear"` |
| `--max_steps` | 100 | optimizer steps |
| `--gradient_accumulation_steps` | 8 | doesn't raise peak memory (microbatches processed and freed sequentially) — use this, not batch size, to grow effective batch on a memory-constrained GPU |
| `--num_generations` | 4 | GRPO group size; must evenly divide `per_device_train_batch_size × gradient_accumulation_steps` — a real TRL constraint, see §7.4 |
| `--per_device_train_batch_size` | 2 | |
| `--max_completion_length` | 1024 | |
| `--max_rollout_steps` | 6 | max tool-call turns per trajectory |
| `--ablation` | `none` | `unmasked` disables masking — see §4 |
| `--device_map` | `single_gpu` | see §7.2 — `auto` silently corrupted generation on an 8GB single-GPU box |
| `--output_dir` | *required* | |
| `--trace_log_path` | `<output_dir>/trace.jsonl` | |

**Effective "epochs":** with `train_size` rows, `per_device_train_batch_size=B`,
`gradient_accumulation_steps=G`, one epoch ≈ `train_size / (B×G)` optimizer
steps. `run_manifest.json` records the actual `approx_epochs` for a run.

### 6.4 Output layout

```
rag_experiments/
├── indexes/{sqlite,chroma}/           # built retrieval index + manifest.json
├── phase0_masking/report.json          # verify_masking.py output
├── runs/<run_name>/
│   ├── training_log.json                # per-logging-step loss/reward/kl/grad_norm/entropy —
│   │                                     # THE file to read for "is training healthy" (§7.7)
│   ├── trace.jsonl                       # one line per trajectory: question, tool calls
│   │                                     # (query + retrieved text), final answer, reward, gold answer
│   ├── run_manifest.json                  # full run config + result summary
│   ├── training_stats.json                 # AgentTune's own stats dump
│   ├── grpo_training_config.yaml            # resolved GRPOConfig
│   ├── adapter_model.safetensors             # final LoRA weights
│   └── checkpoint-<N>/                        # full checkpoint incl. optimizer/scheduler/rng state
└── results/
    ├── eval_matrix.csv                    # evaluate.py output: EM/F1/search-efficiency/groundedness
    └── traces/*.jsonl
```

---

## 7. Known troubleshooting and Fixes applied

Ordered roughly by how likely you are to hit them. All fixes below are
contained to this package's own scripts — none require touching AgentTune's
core framework.

### `device_map="auto"` can silently corrupt generation on a memory-constrained single GPU

 **Fix applied:** `--device_map single_gpu` (the default)
passes `model_init_kwargs={"device_map": {"": 0}}` to `create_agentic_trainer`
to pin the model to GPU 0 explicitly. Use `--device_map auto` only on a
multi-GPU box.

Related: the rollout engine's *own* model copy (as opposed to the trainer's)
is dead weight once wired into real `GRPOTrainer` training — once a `trainer`
object is attached, `rollout_factory.py`'s `_gen` generates exclusively via
`trainer.model`/`trainer.model_wrapped`, never `engine.model` (only
`engine.tokenizer` is used elsewhere). `build_rollout_fn` already forces this
copy onto CPU (`engine_kwargs={"device_map": "cpu"}`) so it doesn't compete
for VRAM for no reason. Don't "fix" this back onto GPU without re-reading why.

### Shrinking the retrieval index's chunk size breaks retrieval quality

Shrinking `--chunk_size` on `build_index.py` (default 512) below ~256 will break FTS5 phrase matching against paraphrased questions — searches start returning "No results found" for everything,
which zeroes the `search_usage_reward`/`rag_correctness_reward` signal (not
because training is broken, but because retrieval found nothing). Use the
default `chunk_size=512` for anything where retrieval quality/reward signal
matters. Address VRAM pressure via `--gradient_accumulation_steps`,
`--num_generations`, `--max_completion_length`, or `--max_rollout_steps`
instead — those don't degrade the corpus.

### TRL's `num_generations` divisibility constraint

`per_device_train_batch_size × gradient_accumulation_steps` must be evenly
divisible by `num_generations` .

### `Step.action`'s runtime shape differs from its own docstring

`agentic/trajectory/dataset.py`'s `Step.action` docstring claims a flat
`{"name": ..., "arguments": ...}` dict, but the real rollout populates
OpenAI-style `{"tool_calls": [{"type": "function", "function": {"name": ...,
"arguments": "<json string>"}}]}`. Don't inspect `.action` directly — use
`trajectory_utils.extract_tool_calls`/`is_tool_step` everywhere, which
normalize this. Same file's `extract_question_text` handles a related quirk:
`Trajectory.task` is a plain string in standalone/debug scripts, but a
chat-message list (`[{"role", "content"}, ...]`) during real GRPO training
since that's what `to_grpo_dataset()`'s `prompt` column contains.

### The tool-call parser needs the `<tool_call>` block to be the model's entire output

`rollout_factory.py`'s `_extract_tool_calls` does `json.loads()` on the whole
string after stripping `<tool_call>` tags — a model that writes even one
lead-in sentence before the tag breaks parsing silently (no tool call
detected, treated as a final answer instead). Handled via
`data/hotpotqa.py`'s `DEFAULT_SYSTEM_PROMPT`, which explicitly instructs "when
you call a tool, your entire response must be ONLY the `<tool_call>` block" —
a prompt fix in this package, not a framework edit. Keep this instruction if
you customize the system prompt.

### training health

`TrlAgenticGrpo.get_training_stats()`'s `"training_history"` field is dead
code — always `[]`, never populated anywhere in `agentic_grpo.py`. The real
per-logging-step record (loss, reward, kl, grad_norm, learning_rate, entropy)
lives on the underlying HF `GRPOTrainer`'s `trainer.trainer.state.log_history`.
`train_grpo.py` saves this explicitly to `training_log.json` — **this is the
file to read**, not `training_stats.json`'s `training_history` field.

### `gradient_checkpointing=True` needs an explicit `enable_input_require_grads()` call

Setting `gradient_checkpointing=True` alone had *zero* measurable effect on
peak memory in a real OOM trace.
Root cause is a documented PEFT caveat: checkpointing's
backward hook needs a tensor with `requires_grad=True` at the checkpoint
boundary; with the base model frozen, the embedding layer's output has
`requires_grad=False`, so the hook never fires and every activation gets kept
anyway. **Fix already applied:** `train_grpo.py` calls
`model.enable_input_require_grads()` on the trainer's model right before
`.train()`.

###  Don't pass `tools=[...]` to the trainer alongside `rollout_func=`

TRL's native `GRPOTrainer.__init__` builds its own tool-sync dict from any
`tools=` kwarg by indexing `tool.__name__` — it expects plain named callables,
not `BaseTool` instances, and crashes (`AttributeError: 'SearchCorpusTool'
object has no attribute '__name__'`). Since `rollout_func` already wires tools
into AgentTune's own masking-aware rollout loop, `tools=` on the trainer is
redundant and actively harmful — `train_grpo.py` deliberately omits it (still
passes `tools=` to `create_rollout_fn`, which is correct and necessary).

**Passing `tools=[...]` to the trainer alone
does not guarantee the masking-aware rollout path runs during training** — TRL
only patches in AgentTune's rollout when `rollout_func` is explicitly set.
`train_grpo.py` builds the rollout via `create_rollout_fn(...)` and passes it
as `rollout_func=...` for exactly this reason — this is what makes §4's
masking guarantee hold during *real* training, not just in the standalone
`verify_masking.py` check.

---

## 8. Validation status

| Layer | Status |
|---|---|
| Unit tests (CPU) | 37/37 passing |
| Unit tests (GPU) | 1/1 passing |
| Masking proof | Passing on 2 models (`Qwen2.5-1.5B-Instruct`, `Qwen3-0.6B`) × 2 GPU classes — ~92% masked-span/tool-output overlap, ~100% unmasked-span/own-generation overlap |
| Training loop (forward + rollout + backward + optimizer step + checkpoint save) | Confirmed working, multiple scales (1 step; 4 rows × 2 epochs) |
| Reward signal (`trace.jsonl` per-trajectory) | Correct as of the fix in §7.12 below — verified against the actual reward formula by hand |
| Full-scale matrix run (large `train_size`, multi-model/backend comparison, `--ablation unmasked` comparison, judge-scored `evaluate.py`) | **Not yet run to a final results archive.** Every individual component is validated; this is the remaining step. |

---

## 9. Changes from PR10 (merge onto `secondary`)

This branch (`agenticrag-on-secondary`) layers the agentic-RAG package on top of
the `secondary` branch (PR10: heal loop, memory, project, service, langgraph
spine, eval, strategy, real-GPU examples).

### 9.1 RAG package overlay 

- **Replaced** `secondary`'s RAG skeleton — `environment.py`, `retriever.py`,
  `rewards.py`, `search_tool.py`, and the skeleton `__init__.py` — with full
  package (`retrieval/` backends, `tools/`, `rewards/`, `data/hotpotqa.py`,
  `scripts/`, `colab/`, `trajectory_utils.py`). agenticrag branch masking-aware rollout,
  SQLite-FTS + Chroma backends, and phase-1 reward composition are kept.
- **Kept** `secondary`'s `datagen.py` (docs→QA generation + difficulty
  curriculum) **logic verbatim**, but made it self-contained: its two imports
  (`Chunk` from `retriever`, `answer_correctness_reward` from `rewards`) were
  inlined into `datagen.py` so it no longer depends on the dropped skeleton
  modules. Only the imports changed; the QA-generation files are kept.
- `tests/rag/`: 5 test modules. **44/44 tests pass** on the merged branch (incl. 6 datagen
  tests + the GPU masking test).
- `pyproject.toml`: re-added the `[rag]` extra (`chromadb`, `langchain-text-
  splitters`); kept all `secondary` extras (`openenv`/`service`/`docs`) and the
  `transformers>=4.56,<6` pin.

### 9.2 `langgraph` is now a base dependency
**Fix:** added `langgraph>=0.2` to `[project.dependencies]` in `pyproject.toml`.

### 9.3 New: vLLM rollout backend (faster generation than hf, masking preserved)
Added vLLM as a generation backend for `train_grpo.py`. vLLM generates rollout
completions via TRL's colocated engine while **still running the full agentic
tool-calling loop with `env_mask` masking** .

**Usage:**
```bash
python -m agenttune.rag.scripts.train_grpo \
    --model Qwen/Qwen3-0.6B --backend sqlite \
    --index_dir rag_experiments/indexes/sqlite \
    --use_vllm --vllm_mode colocate \
    --vllm_max_model_length 2048 --vllm_gpu_memory_utilization 0.45 \
    --output_dir rag_experiments/runs/vllm_run
```

New flags: `--use_vllm`, `--vllm_mode {colocate,server}`,
`--vllm_max_model_length` (default 2048 — ample for RAG; lowering from the
model's default e.g. 40960 shrinks the KV cache so the colocated vLLM fits
beside the training model), `--vllm_gpu_memory_utilization` (default 0.45 —
leaves the majority of VRAM for training on a 16GB GPU).

**fixes required to make vLLM + the agentic rollout coexist**:

1. `agentic_grpo.py`: removed the `not use_vllm` guard on the
   `_generate_single_turn` monkeypatch. Previously, `use_vllm=True` skipped the
   patch, so TRL's native `_generate_single_turn` ran and **bypassed our
   `rollout_func` entirely** — no tool calls, no `env_mask`. Now the patch
   installs for both backends; with vLLM it first calls
   `trainer.vllm_generation.sync_weights()` (so the inference engine matches the
   trainer's current step), then routes through `rollout_func`.
2. `rollout_factory.py`: the vLLM branch in `_execute_trajectory`'s `_gen` helper
   now lazily imports `trl.experimental.openenv.generate_rollout_completions`
   (the import was commented out, so the branch was dead code that would
   `NameError` if reached). This is TRL's supported helper for custom agentic
   rollouts under `use_vllm` — it generates via `trainer.vllm_generation` and
   returns `prompt_ids`/`completion_ids`/`logprobs`/`text`, feeding the same
   tool-loop structure as the transformers path.
3. `rollout_factory.py`: `rollout_fn`'s returned `logprobs` are now
   `list[list[float]]` (plain floats), not `list[list[(logprob, token_id)]]`
   tuples. TRL's `_generate_and_score_completions` does `torch.tensor(logps)` on
   this to build `sampling_per_token_logps`; tuples made it shape `[B,N,2]` and
   crashed the importance-sampling subtraction `old_per_token_logps -
   sampling_per_token_logps` under vLLM (where `old_per_token_logps` is always
   computed) with `tensor a (N) must match tensor b (2) at dim 2`. The
   transformers path tolerated tuples only because it sets
   `old_per_token_logps=None` and skips that subtraction; emitting floats is
   correct for both. (The surrounding code comment already documented floats as
   the intended format — this makes the code match it.)

### 9.4 Validation of the merged branch 

Ran on a vast.ai GPU (RTX 4060 Ti 16GB, torch 2.12+cu130, trl 1.7.1,
transformers 5.13.0, vllm 0.23.0, chromadb 1.5.9):

| Check | Result |
|---|---|
| `pytest tests/rag/` | **44/44 passed** |
| 4-sample smoke (transformers backend) | build_index → verify_masking → train_grpo all green; LoRA adapter saved |
| **16 sample, 64 sample vLLM training** | working |


### 9.5 Backend-aware system prompt (BM25 vs dense)

The system prompt is selected automatically from `--backend` via
`hotpotqa.get_system_prompt(backend)`, because the two retriever families need
different query styles (per the retriever-adaptive query formulation finding;
see Search-R1 / R1-Searcher):

- **`--backend sqlite` (BM25/FTS5, lexical):** the prompt instructs short
  2-5 keyword queries (long natural-language queries dilute the token-overlap
  match), multi-hop decomposition (search → read → search again with a query
  informed by the prior result), and **retry-on-empty**: if a search returns
  "No results found", retry with a shorter query (drop the least distinctive
  keyword) or a different proper noun — only answer from memory after 2-3
  failed searches with different terms.
- **`--backend chroma` (dense embeddings):** the prompt instructs short
  natural-language queries (the retriever matches on semantics, not exact
  words), one hop per query.

Both prompts teach multi-hop decomposition, which HotpotQA questions require.
No flag needed — the prompt follows the `--backend` you pass to `train_grpo.py`,
`verify_masking.py`, or `evaluate.py`.

```bash
# BM25 (faster, lexical, keyword queries + retry-on-empty)
python -m agenttune.rag.scripts.build_index --backend sqlite --train_size 2000 \
    --output_dir rag_experiments/indexes/sqlite
python -m agenttune.rag.scripts.train_grpo --backend sqlite \
    --index_dir rag_experiments/indexes/sqlite --use_vllm ...

# Chroma (dense, natural-language queries)
python -m agenttune.rag.scripts.build_index --backend chroma --train_size 2000 \
    --output_dir rag_experiments/indexes/chroma
python -m agenttune.rag.scripts.train_grpo --backend chroma \
    --index_dir rag_experiments/indexes/chroma --use_vllm ...
```

---
