# agenttune.rag.synthesis — Synthetic Multi-hop RAG QA Dataset Generation

Docs-in → multi-hop question/answer dataset out, ready to feed
`agenttune.rag.scripts.train_grpo`. Built as a self-contained use-case package
under `agenttune.rag`, reusing the package's existing primitives and adapting
ideas from GRADE, RAGAS, MHTS, SAGE, and Castform (all verified against source).

**Status: VALIDATED end-to-end on real models** (BGE-M3 + Qwen3.6 via an OpenAI-compatible endpoint,
`enable_thinking=False`). Sample outputs preserved at
`synth_outputs/synth_out_v3/`.

## Pipeline

```
docs ─► Stage 0: chunk + extract entities + build typed graph
     ─► Stage 1: sample typed 2–5 hop paths (networkx, bounded multi-path)
     ─► Stage 2: answer-first QA generation (MHTS)
     ─► Stage 3: closed-loop verification — retrieval-necessity + chain-dependency (1 retry)
     ─► Stage 4: 2D difficulty label (GRADE power-mean) + balance (Know Your RAG)
     ─► Stage 5: emit GRPO-ready dataset + full evaluation + tabular dumps
```

> **Qwen3 reasoning models:** pass `enable_thinking=False` (auto-on in
> `OpenAICompatLLMClient`). The thinking trace otherwise breaks generation
> (echoes placeholders) and the solver (eats the token budget → 0.0
> answerability). One parameter fixed answerability 0→0.8, diversity 0→0.93.

Every stage dumps CSV **and** Parquet to `out_dir/`, with per-LLM-call metadata
(tokens, cost, latency, request/response previews) captured for reproducibility.

## Reuse map (what we picked up vs built)

| Component | Source | Reused how |
|---|---|---|
| Chunking | our `retrieval/chunker.py` | langchain `RecursiveCharacterTextSplitter` |
| Retrieval (Stage 3 check 1) | our `retrieval/sqlite_fts.py` | `SearchBackend.search` |
| SQuAD F1 (scorer) | our `rewards/qa_metrics.py` | `f1_score` (wraps HF `evaluate`) |
| Corpus loading | our `data/hotpotqa.py` | `load_hotpotqa_splits`, `build_corpus_from_hotpotqa` |
| GRPO dataset schema | our `data/hotpotqa.py` | `to_grpo_dataset` shape + `get_system_prompt` |
| LLM client | our `rewards/judge_eval.py` pattern | Groq SDK, OpenAI-compatible |
| Graph path sampling | GRADE `find_DAG_path_shortest.py` | `networkx.all_simple_paths` + dedup-by-(start,end) |
| Typed edges (exact/contextual) | GRADE `find_same_entity.py` | entity-set overlap + LLM coreference |
| Abstract edges | RAGAS `MultiHopAbstractQuerySynthesizer` | summary cosine similarity |
| 2D difficulty | GRADE `retriever_correlation.py` | power-mean p=-3 (emphasizes weakest chunk) |
| Answer-first generation | MHTS (arXiv:2504.08756) | fix answer, work backward |
| Closed-loop feedback | SAGE (bounded to 1 retry, Castform cost) | |
| Chain-dependency check | **new** (Min et al. 2019 fix) | mask a hop, re-solve |
| Difficulty balance | Know Your RAG | resample across matrix cells |

All reused repos (GRADE, RAGAS, DeepEval) are Apache-2.0.

## Evaluation metrics 

Each metric is drawn from an established framework. No ad-hoc scores.

| Metric | Source / basis | What it measures | Implemented |
|---|---|---|---|
| **Answerability pass-rate** | ARES (Automated RAG Evaluation) | can a solver answer from the gold path? (no external gold needed) | solver LLM + our SQuAD F1 ≥ 0.5 |
| **Faithfulness** | RAGAS `faithfulness`, SelfCheckGPT family | is the answer grounded in the passages (not hallucinated)? | LLM-judge 0.0–1.0 |
| **Retrieval recall@k** | standard IR | do gold chunks surface in top-k? (is the Q retrievable by real RAG) | our `SearchBackend` |
| **Multi-hop necessity** | Min et al. 2019 (EMNLP) | fraction where masking a hop BREAKS the solver (= genuinely multi-hop) | Stage 3 check 2, aggregated |
| **Diversity (lexical)** | n-gram novelty | unique 4-gram ratio across questions | deterministic |
| **Diversity (structural)** | type entropy | question-type (bridge/comparison/...) distribution | deterministic |
| **Diversity (semantic)** | pairwise cosine | mean question-question cosine (lower = more diverse) | Qwen3 embeddings |
| **Coverage** | corpus coverage | fraction of chunks in ≥1 gold path | deterministic |
| **Difficulty distribution** | GRADE 2D matrix | (hop × retrieval-difficulty) cell histogram | deterministic |

The headline metric is **multi-hop necessity** — it's the one that distinguishes
a genuine reasoning-chain dataset from a "bag of loosely related facts" dataset,
and it's the exact failure Min et al. (2019) showed most multi-hop benchmarks
suffer from. Our Stage-3 chain-dependency check is what makes this measurable.

## Schema — what each column / field means

Every output row (`dataset_all.csv` = all samples incl. rejected;
`dataset_final.csv` = accepted only) carries these fields:

| Field | Meaning |
|---|---|
| `sample_id` | `qa_<path_id>`, e.g. `qa_path_2hop_3`. The path_id is `path_{hop}hop_{index}` where **`hop`** = number of edges in the reasoning path (2 = a 2-hop question, needs 3 chunks) and **`index`** = the path's position within its hop-band after sampling (0-based, arbitrary order). So `qa_path_2hop_3` is the 4th-sampled 2-hop question. The `2`/`3` in `qa_path_2hop_3`: **2 = hop count, 3 = sample index**. |
| `question` / `answer` | the generated multi-hop question + its short concrete answer |
| `gold_chunk_ids` | JSON list of the ordered chunk ids the question traverses to reach the answer, e.g. `["A::0","B::1","C::0"]`. Format is `<doc_id>::<chunk_index>`. These are the passages a RAG system must retrieve to answer. Length = hop_count + 1. |
| `gold_reasoning` | one sentence: how the chain resolves the answer (human/LLM-readable) |
| `hop_count` | number of edges (= reasoning steps) in the path. 2-hop = traverse 3 chunks. |
| `question_type` | the 2WikiMultiHopQA-style category. **Two values:** `bridge` (entity-anchored: hop 2's entity is found via hop 1) and `compositional` (the question combines facts across hops, includes abstract/semantic edges). |
| `specificity` | RAGAS-style specific-vs-abstract. **Two values:** `specific` (names concrete entities — entity-anchored, lexical) and `abstract` (refers to concepts/themes, needs semantic matching). |
| `path_entities` | JSON list of named entities the path traverses (for topic analysis/filtering) |
| `retrieval_necessity` | Stage-3 check 1 (Castform): the rank (0-indexed) of the answer/seed chunk in top-5 retrieval over the **full** corpus. `-1` = not retrieved (good — the question doesn't leak its own answer). `_pass` = True if it did NOT appear at rank 0. |
| `chain_dependency` | Stage-3 check 2 (the novel part, Min et al. 2019 fix): the **worst-case solver accuracy** when each intermediate hop is masked in turn. `0.0` = masking any hop broke the solver (every hop is load-bearing = good). `_pass` = True if < 0.5. See "What chain-dependency means" below. |
| `revision_count` | how many times the question was regenerated (0 = first try passed; 1 = it failed a check and was revised once) |
| `original_question` | the pre-revision question (empty if never revised) |
| `retrieval_difficulty` | GRADE 2D difficulty: `1 - power_mean(cosine(question, each gold chunk))`, p=-3 (emphasizes the weakest/hardest-to-retrieve chunk). High = hard to retrieve. |
| `difficulty_cell` | the 2D bucket, e.g. `3hop_hard` = 3-hop + high retrieval difficulty |
| `answerability_f1` | Stage-5 eval: solver F1 vs the gold answer when given the full gold path (ARES answerability). `1.0` = solver nailed it; `0.0` = wrong/UNANSWERABLE. |
| `faithfulness_score` | Stage-5 eval: LLM-judge groundedness of the answer in the gold passages (RAGAS), 0.0–1.0. Populated only if a judge LLM is wired. |
| `status` | `accepted` (passed both checks, first try) / `revised_accepted` (passed after 1 revision) / `rejected` |
| `discard_reason` | why a rejected sample failed: `non_load_bearing_hop` (chain-dep failed), `retrieval_leak` (answer leaked in top-k), `empty_qa` (generation produced nothing), `generate_parse_failure` / `revise_failed` / `revise_parse_failure` |
| `n_llm_calls` / `total_tokens` / `cost_usd` | per-sample LLM cost accounting |
| `created_at` | ISO timestamp |

### What chain-dependency means

A multi-hop question is only *genuinely* multi-hop if **every hop is necessary**
— i.e., you can't answer it without traversing the full chain. Most synthetic
multi-hop datasets fail this (Min et al. 2019, EMNLP): they produce questions
that are really "bags of loosely related facts" a reader can triangulate, so
masking any single hop doesn't actually break answerability.

**Our check (Stage 3):** for each intermediate hop in the path, mask that one
chunk, hand a solver LLM the remaining chunks, and ask it to answer. If the
solver *still* answers correctly (F1 ≥ 0.5), that hop wasn't load-bearing →
the question fails `chain_dependency_pass` and is revised or rejected.
`chain_dependency` stores the **worst case** (max solver accuracy across all
masks) — the hop whose removal hurt least. Low = good (every hop matters).

Retrieval-only filters (Castform) miss this failure mode entirely; only re-solving with a hop masked catches it.
`multi_hop_necessity_rate` (in `metrics.json`) is the fraction of accepted
questions that pass — 1.0 means every accepted question is genuinely multi-hop.

## Prerequisites & install

- Python 3.10+ and a CUDA GPU (tested on RTX 5060 Ti 16GB; BGE-M3 uses ~1.1 GB VRAM).
- An LLM endpoint: Groq (`GROQ_API_KEY`) or any OpenAI-compatible endpoint
  (vLLM, TGI, hosted providers, etc.) via `OpenAICompatLLMClient`.

```bash
cd AgentTune/
pip install -e ".[rag]"                          # chromadb, langchain-text-splitters
pip install networkx sentence-transformers groq  # synthesis-specific
# on a vast.ai PyTorch image, install into the existing venv:
#   /venv/main/bin/pip install -e ".[rag]" networkx sentence-transformers groq openai
```

## Running

```bash
# CLI (Groq + BGE-M3, HotpotQA corpus):
export GROQ_API_KEY=gsk_...
python -m agenttune.rag.synthesis.build_dataset \
    --out_dir rag_experiments/synth_out \
    --corpus hotpot --train_size 60 --per_hop 40 \
    --target_per_cell 15 --backend sqlite --embedder bge-m3 \
    --val_fraction 0.2 --split_level chunk
# --fresh forces a Stage 0 rebuild (ignores stage0_cache.pkl); use after a
# model/chunk-size change. Only Stage 0 is cached (see Caching policy below).

# For an OpenAI-compatible endpoint (vLLM / a hosted provider), skip the CLI and build the
# client directly:
#   llm = OpenAICompatLLMClient(model=..., api_key=..., base_url=...)
#   run_pipeline(docs=docs, llm=llm, embedder=emb, out_dir=..., backend="sqlite")
```

Flags: `--train_size` (HotpotQA rows; ×4 ≈ docs), `--per_hop` (paths sampled per
hop band), `--target_per_cell` (balance cap), `--backend sqlite|chroma` (BM25 vs
dense retrieval), `--embedder bge-m3|qwen3-8b` (qwen3-8b needs ~30 GB VRAM, A100),
`--val_fraction` (target val share for the gold-chunk-disjoint split),
`--split_level chunk|entity` (disjointness level; entity is stricter),
`--fresh` (force a Stage 0 rebuild — the staleness escape hatch).

**Generating N unique samples:** the pipeline runs once and produces whatever the
graph yields (verification rejects a chunk each pass). To hit a target row count
*without tuning the validated sampler/generation params to the sample size*,
loop `run_pipeline` across seeds and accumulate unique accepted samples (dedup by
question) until N — Stage 0 is cached after iteration 1, so iterations 2+ only
re-run Stages 1–5 (~1 min each). In one run this reached 25 unique accepted
over 5 iterations on a 365-chunk corpus.

> **Qwen3 reasoning models: `enable_thinking=False` is required** (auto-on in
> `OpenAICompatLLMClient` for Qwen3 models). The thinking trace otherwise
> consumes the token budget → spurious UNANSWERABLE → 0.0 answerability, and
> makes generation echo prompt placeholders. Verified: 7-token clean JSON vs
> 128 tokens of trace. Pass via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
> A non-reasoning model (Qwen2.5-Instruct) has no trace at all.

## Outputs

Every stage dumps CSV **and** Parquet to `out_dir/`:

| File | What it holds |
|---|---|
| `stage0_chunks.csv` | input chunks + extracted entities/keyphrases/summaries |
| `stage0_graph_edges.csv` | typed edges: exact / contextual / abstract |
| `stage1_paths.csv` | sampled reasoning paths (hop count, type, specificity) |
| `stage2_generated.csv` | generated Q/A + generation-call metadata |
| `stage3_verified.csv` | verification results + discard reasons + original-vs-revised |
| `stage4_balanced.csv` | difficulty-labeled + balanced subset |
| `dataset_all.csv` | **every** sample (accepted + revised + rejected) with `discard_reason` — full audit trail |
| `dataset_final.csv` | accepted samples only, eval scores populated |
| `dataset_train.csv` / `dataset_val.csv` | **train/val split by gold-chunk disjointness** — zero gold-chunk overlap between splits (leakage prevention) |
| `dataset_grpo.jsonl` | **GRPO-ready** — `prompt`/`gold_answer`/`question_id`/`gold_path`/`hop_count`/`difficulty_cell` |
| `dataset_train_grpo.jsonl` / `dataset_val_grpo.jsonl` | per-split GRPO-ready (same schema, disjoint gold chunks) |
| `append_report.json` | (append mode only) what was added / dedup stats / coverage delta |
| `metrics.json` | full evaluation suite |
| `cost_summary.json` | aggregate token/cost/latency |
| `manifest.json` | full run config + metrics + split report (reproducibility) |
| `corpus.sqlite` | the indexed corpus (for the retrieval-necessity check) |
| `stage0_cache.pkl` | Stage 0 cache — reruns skip entity extraction (`--fresh` to force a rebuild; only Stage 0 is cached) |

## Train/val split, append mode, and caching

### Train/val split by gold-chunk disjointness (leakage prevention)

A random row-split leaks validation into training: two questions can traverse
*overlapping* gold chunks, so a chunk a model memorizes from a train question
can surface as a val answer. For retrieval-augmented RL this is a correctness
bug, not an optimization.

`run_pipeline` runs a disjoint split as a post-Stage-5 step (see `split.py`).
It builds a graph where two samples are coupled if they share a gold chunk,
finds **connected components** (Union-Find), and bin-packs whole components
into train/val to hit `--val_fraction` — so no chunk appears in both splits'
gold paths (the split report's `gold_overlap_train_val` is asserted `0`).

- `--split_level chunk` (default): gold chunk-id disjointness (strict guarantee).
- `--split_level entity`: path-entity disjointness (stricter; may shrink the
  usable split on small/dense corpora where most questions touch the same
  entities).
- **Degeneracy guard:** if the corpus is too coupled/small for a non-trivial
  split (one giant component, or too few samples), it returns all-train + an
  empty val and flags `degenerate: true` in the report — rather than silently
  breaking a component (which would violate the disjointness guarantee).
  Grow the corpus when this happens.

### Static-corpus append mode (grow a dataset across runs)

The corpus is static, so the Stage-0 graph and retrieval index never change.
`accumulate_and_append` (see `append_dataset.py`)
loads the questions an existing `dataset_all.csv` already holds as the "seen"
set, loops `run_pipeline` across seeds, accumulates NEW unique accepted
samples (dedup by normalized question against existing + new), and appends.

```python
from agenttune.rag.synthesis import accumulate_and_append
report = accumulate_and_append(
    run_pipeline_fn=run_pipeline, docs=docs, llm=llm, embedder=emb,
    out_dir="rag_experiments/synth_out",
    existing_csv="rag_experiments/synth_out/dataset_all.csv",  # None = fresh
    target_total=25, max_iters=20, per_hop=40, target_per_cell=15,
    val_fraction=0.2, split_level="chunk", use_cache=True)
```

The report carries `n_existing_accepted`, `n_new_unique`, `n_total_after`, and a
coverage delta (`coverage_chunks_before`/`after`, `n_new_chunks_covered`) — the
escalation signal: if new samples add ~0 chunks, they're re-treading the same
few paths; escalate to pattern-aware generation (not built; the report tells
you when). Rejected rows in the existing CSV don't count as "seen" — a
regenerated version of a previously-rejected question is fine to keep.

### Caching policy (only Stage 0 is cached — by design, not omission)

- **Stage 0** (entity extraction + embedding + graph build) is the one
  expensive LLM-heavy stage deterministic in the corpus — cached to
  `stage0_cache.pkl` so reruns skip ~hundreds of LLM calls.
- **Stages 1** (path sampling) **and 4** (difficulty) are cheap pure-CPU/embed
  steps where a cache only adds staleness risk — not cached.
- **Stages 2 / 3 / 5** are LLM calls whose output you WANT fresh when the
  prompt, model, or temperature changes — a stale cache there would silently
  serve last week's questions/verdicts. Not cached. (Mid-batch resume for a
  failed Stage 3 is a separate, deferred concern; whole-stage skip is the wrong
  granularity for it.)

`--fresh` (or `use_cache=False` on `run_pipeline`) forces a Stage 0 rebuild
even when a cache exists — the staleness escape hatch after a model or
chunk-size change. Threaded through `accumulate_and_append` too.

## Feeding the dataset to training

`dataset_grpo.jsonl` matches `hotpotqa.to_grpo_dataset`'s schema, so it drops
straight into the existing trainer:

```bash
python -m agenttune.rag.scripts.train_grpo \
    --model Qwen/Qwen3-0.6B --backend sqlite \
    --index_dir rag_experiments/indexes/sqlite \
    --train_size <N> --eval_size <M> ...
# load dataset_grpo.jsonl as the prompt/gold_answer/question_id columns
```

## Your own corpus

To run over your own docs instead of HotpotQA, build a list of
`CorpusDocument(doc_id=..., title=..., text=...)` and pass it to
`run_pipeline(docs=..., ...)` directly (see `build_dataset.run_pipeline`).

## Testing

```bash
pytest tests/rag/test_synthesis.py -v   # 18 CPU tests, fakes — no GPU/key needed
```

These exercise every stage (graph build, path sampling, answer-first generation,
both verification checks, difficulty, balancing, full pipeline + artifact dumps),
the train/val split (zero-overlap, degeneracy, entity-level, artifact dump),
the append mode (dedup-against-existing, coverage delta, fresh start), and the
Stage 0 cache control (`use_cache=False` forces a rebuild). All deterministic
with `FakeLLMClient`/`FakeEmbedder`. If they pass, the only
remaining risk on a real run is LLM/embedding behavior, not the pipeline code.
