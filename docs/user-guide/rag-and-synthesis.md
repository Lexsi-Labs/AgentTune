# RAG & Data Synthesis

`agenttune.rag` has the highest concentration of genuinely-usable, zero-setup utilities
in the whole package: several of them (the FTS backend, the trajectory store, the
leakage-safe splitter) are worth using standalone even if you never touch the full
synthesis pipeline. This page is the practical "how do I call this" version; for the
condensed fact table (ratings, exact gaps) see
[Python API: RAG & Data Synthesis](rag-and-synthesis.md).

Every snippet below was actually run against the current source while writing this page,
including the full 6-stage synthesis pipeline. Two real bugs turned up along the way
and are called out explicitly rather than glossed over (see
[Notes & gotchas](#notes-gotchas)).

## 1. Lexical search: `SQLiteFTSBackend`

Real FTS5 full-text search, stdlib `sqlite3` only, no new dependency. Build an index
from an in-memory corpus and query it:

```python
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

backend = SQLiteFTSBackend(db_path="my_corpus.db")
backend.index([
    {"chunk_id": "c1", "doc_id": "d1", "title": "Onboarding", "chunk_index": 0,
     "text": "New employees should complete security training within their first week."},
    {"chunk_id": "c2", "doc_id": "d1", "title": "Onboarding", "chunk_index": 1,
     "text": "Security training covers phishing awareness and password hygiene."},
    {"chunk_id": "c3", "doc_id": "d2", "title": "Expenses", "chunk_index": 0,
     "text": "Expense reports must be submitted within 30 days of the purchase."},
])

for r in backend.search("security training", top_k=2):
    print(r["metadata"], round(r["score"], 3), r["content"][:50])
# {'chunk_id': 'c2', 'doc_id': 'd1'} 0.0 Security training covers phishing awareness...
# {'chunk_id': 'c1', 'doc_id': 'd1'} 0.0 New employees should complete security t...
```

Every result is a plain `dict` (typed as `SearchResult`, a `TypedDict`) with
`content`/`source`/`metadata`/`score` keys. **Subscript it (`r["metadata"]`), don't
attribute-access it (`r.metadata`)**, it's a real dict at runtime, not an object. (This
distinction matters; see the [Notes & gotchas](#notes-gotchas) bug below.)

`backend.get_document("d1")` returns the full concatenated text of a document by id.
Ranking uses SQLite's own `bm25()` function (reused, not reimplemented); lower is
better internally, but the backend flips the sign so `score` is higher-is-better.

By default (`match_all=True`), query tokens are AND-ed: every token must appear in the
same chunk. Pass `match_all=False` for OR semantics with `bm25()` ranking: standard BM25
behavior, needed when your queries are paraphrased and don't literally share every word
with the target passage (this is also why the synthesis pipeline's "did the question leak
its answer" check in §3 tends to under-fire with default settings, not over-fire).

If you already have real documents rather than hand-built chunk dicts, chunk+index in one
call via `corpus_loader`:

```python
from agenttune.rag.retrieval.corpus_loader import CorpusDocument, build_index

docs = [CorpusDocument(doc_id="onboarding", title="Onboarding", text=open("onboarding.txt").read())]
n_chunks = build_index(backend, docs, chunk_size=512, overlap=64)
```

### Vector search: `ChromaBackend`

Same `SearchBackend` interface, backed by `chromadb`'s own collection API end-to-end (its
default embedding function, its own similarity search, no custom fusion code). New
dependency: `chromadb`.

```python
from agenttune.rag.retrieval.chroma_backend import ChromaBackend

backend = ChromaBackend(persist_dir="./chroma_store", collection_name="my_corpus")
backend.index(chunks)   # same chunk-dict shape as SQLiteFTSBackend
backend.search("security training", top_k=5)
```

Both backends are pickle-safe by design (constructors store only file paths, connections
open lazily); required because rollout workers may run in separate processes.

## 2. Wiring search into an agent as a tool

`SearchCorpusTool` wraps any `SearchBackend` as a standard `agenttune.agentic.tools.base.
BaseTool`, same pattern as the built-in `SQLDatabaseTool` (see
[RL Training](rl-training.md)). It is purely additive: not registered in `ToolRegistry`,
just passed directly wherever tools are accepted.

```python
from agenttune.rag.tools.search_corpus import SearchCorpusTool

tool = SearchCorpusTool(backend, top_k=3)
print(tool.to_schema())
# {'type': 'function', 'function': {'name': 'search_corpus',
#  'description': 'Search the document corpus for passages relevant to a query. ...',
#  'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', ...}},
#                 'required': ['query']}}}

result = tool.execute(query="security training")
print(result.output)   # formatted string: "[chunk_id=... doc_id=... score=...]\n<passage text>" per hit
```

### Option A: low-level, `DictToolHarness`

For a minimal, dependency-free gym-style loop (reset/step, no training stack involved),
wrap the tool in a plain callable and hand it to `DictToolHarness`
(`agenttune.agentic.harness`):

```python
from agenttune.agentic.harness import DictToolHarness

def search_corpus(query: str) -> str:
    return tool.execute(query=query).output

harness = DictToolHarness(tools={"search_corpus": search_corpus}, max_steps=5)
obs = harness.reset("When should new employees complete security training?")

obs, reward, done, info = harness.step(
    {"name": "search_corpus", "arguments": {"query": "security training"}}
)
print(obs.text[:100])   # the formatted search hits
print(done)             # False, not finished yet

obs, reward, done, info = harness.step(
    {"name": "finish", "arguments": {"answer": "Within the first week."}}
)
print(done, obs.text)   # True Within the first week.
```

`DictToolHarness` is deliberately minimal: no LLM in the loop, you (or a test) drive
`step()` directly with whatever action dict you want. Useful for unit-testing tool
wiring, or as the harness half of a hand-rolled agent loop.

### Option B: production, `create_rollout_fn` / `create_agentic_trainer`

For an actual LLM-driven rollout (the path GRPO/DPO/PPO training uses), tools are plain
Python callables with type hints and a Google-style docstring. AgentTune builds the
model-facing JSON schema from those via `transformers.utils.get_json_schema` (same
pattern as `rl-training.md`'s calculator example), but `rollout_factory.py`'s
`_get_callable`/schema-building helpers explicitly check `hasattr(tool, "execute")`
first, so a `BaseTool` **instance** (like `SearchCorpusTool` itself) can be passed
straight through, no wrapper needed. It uses the tool's own `to_schema()` for the model-
facing schema and calls `.execute(**args)` at run time:

```python
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

# Pass the BaseTool instance directly, no wrapper function required.
rollout_fn = create_rollout_fn(
    rollout_backend="transformers", model_path="Qwen/Qwen3-0.6B",
    tools=[tool], max_steps=10,
)

# Or directly inside training:
from agenttune.core.backend_factory import create_agentic_trainer

trainer = create_agentic_trainer(
    algorithm="grpo", model="Qwen/Qwen3-0.6B",
    tools=[tool], reward_funcs=[my_reward_fn],
    train_dataset=train_dataset, output_dir="./runs/grpo-rag",
)
```

If you want a custom description or a renamed tool in the model-facing schema instead of
`SearchCorpusTool`'s defaults, bind a thin wrapper function instead: the same
`SQLDatabaseTool.query`-as-a-bound-method pattern used elsewhere in the repo (see
[RL Training](rl-training.md)):

```python
def search_corpus(query: str) -> str:
    """Search the document corpus for passages relevant to a query.

    Args:
        query: Search query — keywords or a natural-language question.
    """
    return tool.execute(query=query).output

rollout_fn = create_rollout_fn(rollout_backend="transformers",
                               model_path="Qwen/Qwen3-0.6B", tools=[search_corpus])
```

`ReadDocumentTool` (`rag/tools/read_document.py`) is the companion tool: fetches a
document by `doc_id` from a search hit's metadata, same wiring pattern (pass the instance
directly, or bind a wrapper). The returned excerpt is capped at 2500 characters (long
documents get truncated with a note to use `search_corpus` for specific passages instead,
keeping a single tool result from flooding a training rollout's context). See the
[Local Notebooks](../notebooks/local-notebook.md) index for a full search-then-answer episode.

### Option C: one call, `create_rag_trainer`

Options A/B above (and `examples/rag_training_real.py`) show the pieces individually:
build a backend, index it, wrap it in `SearchCorpusTool`, reshape the dataset onto
`prompt`/`answer`, pick reward functions and a matching system prompt, *then* call
`create_agentic_trainer`. `agenttune.rag.create_rag_trainer` does all of that in one
call — pass a model, a corpus (or an already-indexed backend), and QA rows, and it
returns the same kind of trainer `create_agentic_trainer` would, with `.train()` and
the Hub-push helpers already attached:

```python
from agenttune import create_rag_trainer  # or: from agenttune.rag import create_rag_trainer

trainer = create_rag_trainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    corpus=[
        {"doc_id": "onboarding", "title": "Onboarding",
         "text": "New employees should complete security training within their first week."},
        {"doc_id": "expenses", "title": "Expenses",
         "text": "Expense reports must be submitted within 30 days of the purchase."},
    ],
    train_dataset=[
        {"prompt": "When should new employees complete security training?",
         "answer": "within their first week"},
    ],
    output_dir="./out_rag",
    max_steps=20,
)
results = trainer.train()
```

Under the hood this builds a `SQLiteFTSBackend` (or pass `retrieval_type="chroma"`),
chunks + indexes `corpus` into it, wraps it in `SearchCorpusTool(top_k=...)`, and
defaults `reward_funcs` to `["search_grounding_reward", "format_reward",
"answer_correctness_reward"]` (weights `[1.0, 0.2, 1.0]`) with a matching system prompt
that tells the model to search, cite `chunk_id`, and answer inside `<answer>` tags.

None of that is mandatory:

- **Bring your own retrieval backend**: pass `retrieval_backend=<already-indexed
  SearchBackend>` instead of `corpus=` to skip building/indexing entirely (e.g. reuse
  one built via `build_index` in [§1](#1-lexical-search-sqliteftsbackend)).
- **Bring your own reward(s)**: `reward_funcs=` accepts any callable(s), any
  `REWARD_REGISTRY` string name(s), or a mix, exactly like `create_agentic_trainer`
  itself — it is not restricted to the three RAG defaults above.
  ```python
  def my_reward(prompts, completions, answer=None, **kwargs):
      return [1.0 if answer[i].lower() in c.lower() else 0.0 for i, c in enumerate(completions)]

  trainer = create_rag_trainer(..., reward_funcs=[my_reward], reward_weights=[1.0])
  ```
- **Bring your own tool(s)**: `tools=` adds any additional `BaseTool` instance or plain
  callable alongside the auto-built `search_corpus` tool (e.g. `ReadDocumentTool`, or a
  custom tool of your own):
  ```python
  from agenttune.rag.tools.read_document import ReadDocumentTool
  trainer = create_rag_trainer(..., tools=[ReadDocumentTool(backend)])
  ```
- **Any GRPO/DPO/PPO/RLOO/BCO knob** (`algorithm=`, `peft_config=`, `num_generations=`,
  `use_vllm=`, ...) forwards straight through in `**kwargs`, same as
  `create_agentic_trainer`.

The built retrieval backend and tool are attached to the returned trainer as
`trainer.retrieval_backend` / `trainer.search_tool` if you need to inspect or reuse them
(e.g. to run a search manually, or pass the same backend into a second trainer).

## 3. The docs-to-training-data synthesis pipeline

`rag/synthesis/build_dataset.py`'s `run_pipeline` turns a document corpus into a
difficulty-tagged, grounded multi-hop QA training set through six real stages, not
stubs, every stage below does real work:

| Stage | File | What it does |
|---|---|---|
| 0. Corpus → typed graph | `graph.py` | Chunk, LLM-extract entities/keyphrases/summary per chunk, embed, build a typed-edge graph (`exact`/`contextual`/`abstract`) |
| 1. Path sampling | `paths.py` | `networkx`-based bounded simple-path enumeration, 2–5 hops, no LLM call |
| 2. Answer-first generation | `generate.py` | Fix the answer from the path's last chunk first, then generate a question requiring the full chain: structured JSON-mode LLM call per path |
| 3. Closed-loop verification | `verify.py` | Two checks: does top-k retrieval leak the answer chunk? Does masking an intermediate hop still let a solver LLM answer correctly (= that hop wasn't load-bearing)? One targeted regeneration retry on failure |
| 4. Difficulty labeling | `difficulty.py` | 2D (hop-count × embedding-based retrieval-difficulty) matrix labeling + balanced resampling |
| 5. Evaluation + split | `evaluate.py` + `split.py` | Answerability/faithfulness/diversity/coverage metrics, then a leakage-safe train/val split by shared-gold-chunk disjointness (union-find + greedy bin-packing, pure stdlib) |

### Zero-dependency run: `FakeLLMClient` + `FakeEmbedder`

Both the LLM and the embedder are injected (`LLMClient`/`Embedder` Protocols), so the
whole pipeline is exercisable on CPU with no network, no API key, and no GPU, as long
as you script the fake LLM's responses to look like real JSON for each call `purpose`.
This is verified end-to-end below (`per_hop`/`target_per_cell` are set very small so a
3-chunk corpus produces a full run in under a second):

```python
import json
from agenttune.rag.synthesis.build_dataset import run_pipeline
from agenttune.rag.synthesis.llm_client import FakeLLMClient
from agenttune.rag.synthesis.embeddings import FakeEmbedder

docs = [
    {"doc_id": "d1", "title": "Northwind Robotics",
     "text": "Northwind Robotics was founded in 2015 by Elena Cho in Austin, "
             "building the original Aria assembly-line robot for small factories."},
    {"doc_id": "d2", "title": "Career Move",
     "text": "Elena Cho left Northwind Robotics in 2019 to become CTO of Delta "
             "Systems, overseeing its industrial automation division."},
    {"doc_id": "d3", "title": "Delta Systems HQ",
     "text": "Delta Systems is headquartered in Denver, Colorado, and ships the "
             "Aria-2 robot line to warehouses across the region."},
]

def responder(purpose, messages):
    content = messages[0]["content"]
    if purpose == "extract_entities":
        if "Northwind Robotics was founded" in content:
            ents = ["northwind robotics", "elena cho"]
        elif "left Northwind Robotics" in content:
            ents = ["elena cho", "delta systems"]
        elif "headquartered in Denver" in content:
            ents = ["delta systems", "denver"]
        else:
            ents = []
        return json.dumps({"entities": ents, "keyphrases": ["robot"], "summary": "s"})
    if purpose == "contextual_equiv":
        return json.dumps({"mappings": []})
    if purpose == "generate_answer_first":
        return json.dumps({
            "answer": "Denver",
            "question": "Where is the automation division now based that a former "
                        "Austin robotics founder went on to lead?",
            "reasoning": "Elena Cho founded a robotics firm, later led Delta "
                        "Systems's automation division, which is based in that city.",
        })
    if purpose.startswith("chain_dep_mask") or purpose == "solver":
        return json.dumps({"answer": "UNANSWERABLE"})
    if purpose == "eval_answerability":
        return json.dumps({"answer": "Denver"})
    if purpose == "revise_question":
        return json.dumps({"question": "", "answer": ""})
    return "FAKE_RESPONSE"

result = run_pipeline(
    docs=docs, llm=FakeLLMClient(responder=responder), embedder=FakeEmbedder(),
    out_dir="./rag_pipeline_demo", backend="sqlite",
    per_hop=3, target_per_cell=3, val_fraction=0.5,
)
print(len(result), "accepted samples")
for s in result:
    print("-", s.question, "->", s.answer, "| hop:", s.hop_count)
# 2 accepted samples
# - Where is the automation division now based that a former Austin robotics founder
#   went on to lead? -> Denver | hop: 2
# (same question twice — two distinct routings through the same 3-chunk graph)
```

This writes a full artifact tree to `out_dir`: `stage0_chunks.csv`/`stage0_graph_edges.
csv`, `stage1_paths.csv`, `stage2_generated.csv`, `stage3_verified.csv`,
`dataset_all.csv` (every sample including rejected, with `discard_reason`),
`dataset_final.csv`, `dataset_grpo.jsonl` (prompt/gold_answer/gold_path, ready for GRPO
training), `metrics.json`, `cost_summary.json`, and `manifest.json`. With only 3 chunks
you'll see most candidate questions get rejected (a 3-chunk toy corpus barely supports
one real 2-hop path); that's the verification stage doing its job, not a bug.

### Swapping in the real clients

For an actual production run, swap `FakeLLMClient`/`FakeEmbedder` for the real
implementations; everything else in the call is identical:

```python
from agenttune.rag.synthesis.llm_client import GroqLLMClient, OpenAICompatLLMClient
from agenttune.rag.synthesis.embeddings import BGEM3Embedder

# Groq (needs GROQ_API_KEY env var or api_key=...)
llm = GroqLLMClient(model="qwen/qwen3.6-27b")

# OR any OpenAI-compatible endpoint (vLLM, TGI, a local server, etc.)
llm = OpenAICompatLLMClient(model="Qwen/Qwen3-8B", api_key="EMPTY",
                             base_url="http://localhost:8000/v1")

embedder = BGEM3Embedder(device="cuda")   # ~2.3GB VRAM — the default, robust choice

result = run_pipeline(docs=real_docs, llm=llm, embedder=embedder,
                       out_dir="./rag_pipeline_real", per_hop=30, target_per_cell=15)
```

`GroqLLMClient`'s default model (`qwen/qwen3.6-27b`) is a reasoning model that emits a
thinking trace before its JSON answer. The client already sets `max_tokens=1024`
internally in the pipeline's own calls to leave room for it, but if you call `.chat()`
directly with a small `max_tokens` you can spuriously get an empty/truncated answer.
`OpenAICompatLLMClient` auto-disables Qwen3's thinking trace by default
(`enable_thinking=False` via `extra_body`) for exactly this reason.

Only Stage 0 is cached (`stage0_cache.pkl` in `out_dir`); it's the one expensive,
corpus-deterministic, LLM-heavy stage. Pass `use_cache=False` (or delete the cache file)
after a model or chunk-size change. Stages 1–5 are always fresh by design.

See the [Local Notebooks](../notebooks/local-notebook.md) index for the generator notebook
and the full synthesis pipeline notebook.

### Standalone utilities worth using on their own

**`synthesis/io_utils.py`'s `extract_json(text)`**: the JSON-from-messy-LLM-text
extractor every stage above uses internally (direct parse → fenced-code-block →
brace-scan fallback). Broadly useful beyond this pipeline for any LLM-JSON-mode call.

**`synthesis/split.py`'s `split_by_gold_chunks`** is the leakage-safe splitter
`run_pipeline` calls internally at the end of Stage 5, but it's also directly callable
on any list of `QASample`-shaped objects that have `gold_chunk_ids`, independent of the
rest of the pipeline. It returns a 3-tuple, not 2: `(train, val, report)`:

```python
from agenttune.rag.synthesis.split import split_by_gold_chunks
from agenttune.rag.synthesis.schema import QASample

samples = [
    QASample(sample_id="s1", gold_chunk_ids=["c1", "c2"]),
    QASample(sample_id="s2", gold_chunk_ids=["c2", "c3"]),   # shares c2 with s1 -> same component
    QASample(sample_id="s3", gold_chunk_ids=["c9"]),
    QASample(sample_id="s4", gold_chunk_ids=["c10"]),
]
for s in samples:
    s.status = "accepted"   # only accepted/revised_accepted samples are eligible

train, val, report = split_by_gold_chunks(samples, val_fraction=0.3, seed=0,
                                          min_train=1, min_val=1)
print([s.sample_id for s in train], [s.sample_id for s in val])
# ['s4', 's3'] ['s1', 's2']  (s1/s2 share chunk c2, so they're coupled into the same split)
print(report["gold_overlap_train_val"])   # 0 — the guarantee, asserted defensively
```

The `report` dict also carries `n_components` (connected components in the
sample-sharing graph) and `largest_component_fraction`, a diagnostic for whether your
corpus is so densely interlinked that a meaningful split is even possible. If `min_train`/
`min_val` can't be satisfied without breaking a component's disjointness guarantee, it
returns everything as train, an empty val, and `report["degenerate"] = True`; it never
silently violates the leakage guarantee to hit your requested fraction.

**`synthesis/curriculum.py`'s `build_curriculum` + `CurriculumSampler`**: an easy→hard
training curriculum builder. `build_curriculum` combines a solve-difficulty signal (from
a base-model probe) and a retrieval-difficulty signal into one ordering; `CurriculumSampler`
paces batches across training steps so early steps see only easy questions:

```python
from agenttune.rag.synthesis.curriculum import build_curriculum, CurriculumSampler

questions = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What year did Northwind Robotics move its HQ?", "answer": "2019"},
]
curriculum = build_curriculum(questions)   # no probe/fn given -> neutral difficulty defaults

sampler = CurriculumSampler(curriculum, max_steps=10, batch_size=2)
sampler.sample(0)   # early step -> restricted to the easy pool
sampler.sample(9)   # late step -> full pool available
```

Without `solve_difficulty_labels=`/`retrieval_difficulty_fn=`, every question gets a
neutral default difficulty (this is the CPU-only, no-model path); the real signal comes
from `probe_solve_difficulty` (needs a base model, GPU) and a GRADE-style embedding
retrieval-difficulty function. `.reprobe()` on the sampler is the mid-training hook for
re-running that GPU probe periodically. Core sampling logic (`build_curriculum`,
`CurriculumSampler.sample`) is pure Python and needs neither.

## 4. Reward functions and tool-call parsing for RAG agents

Two more standalone pieces worth knowing about if you're building reward functions or
handling non-JSON tool-calling models:

**`rag/tools/xml_tool_parser.py`'s `parse_xml_tool_calls(text)`**: Qwen3.5's chat
template emits tool calls in a custom `<function=NAME><parameter=ARG>VALUE</parameter>
</function>` XML format instead of JSON, which the standard rollout tool-call extractor
doesn't understand. This parses it into the same normalized OpenAI-style structure the
rollout loop expects:

```python
from agenttune.rag.tools.xml_tool_parser import parse_xml_tool_calls

xml = "<function=search_corpus> <parameter=query>revenue growth</parameter> </function>"
parse_xml_tool_calls(xml)
# [{'type': 'function', 'function': {'name': 'search_corpus',
#   'arguments': '{"query": "revenue growth"}'}}]

parse_xml_tool_calls("plain text, no function tags")   # None
```

`patch_xml_tool_call_parser()` monkeypatches `rollout_factory._extract_tool_calls` to
fall back to this parser only when the original JSON parser returns `None`, idempotent,
safe to call more than once, and a no-op for JSON-emitting models.

**`rag/rewards/finder_rewards.py`'s `numeric_correctness_reward`**: despite the name,
this is **not** pure regex/stdlib. It always computes a token-F1 fallback via
`rewards/qa_metrics.py`'s `f1_score`, which wraps HuggingFace's `evaluate.load("squad")`,
meaning every call needs the `evaluate` package installed and, on the first call in a
process, a network round-trip to fetch the SQuAD metric's builder script from the Hub:

```python
from agenttune.rag.rewards.finder_rewards import numeric_correctness_reward

numeric_correctness_reward(
    prompts=["What was revenue growth?"],
    completions=["<answer>Revenue grew 12%.</answer>"],
    gold_answer=["12%"],
)
# [1.0] — the regex-extracted number (12) matched gold within the 2% tolerance;
# the F1 fallback still runs underneath regardless, since it's unconditional
```

If `evaluate`/network access isn't available, use `rag/datagen.py`'s `token_f1`
([§6](#6-a-simpler-dependency-light-qa-toolkit) below) or the fully pure-stdlib scorers in
`eval.agent_eval` (`token_f1()`, `numerical_recall()`) instead, same idea, no HF
`evaluate` dependency.

## 5. Memory compression (MEM1-style)

`rag/memory/m1_rewrite.py` implements a MEM1-paper-style running-state compressor: after
each turn, the agent's accumulated conversation history is replaced with a compact
rewritten state so context size stays roughly constant regardless of how many searches it
runs. Two pure regex/string functions are usable standalone to inspect what compression
would actually produce, no model or training loop required:

```python
from agenttune.rag.memory.m1_rewrite import extract_internal_state, compress_tool_output

# Pulls the model's <think> (or structured <state>) block out of a response
response = "<think>The user wants revenue. I found 12% in doc1.</think>Revenue grew 12%."
extract_internal_state(response)
# 'The user wants revenue. I found 12% in doc1.'

# Compresses a search_corpus-formatted tool result into an evidence note,
# keeping chunk_id/doc_id/score (so groundedness/citation checks still work)
# and just the first sentence of each chunk, capped.
tool_output = "[chunk_id=c1 doc_id=d1 score=0.900]\nRevenue grew 12 percent year over year. Costs also rose."
compress_tool_output(tool_output)
# 'evidence[chunk_id=c1 doc_id=d1 score=0.900]: Revenue grew 12 percent year over year.'
```

`mem1_post_step_hook` is the piece that actually wires this into a live rollout; it's
passed as `post_step_hook` to `create_rollout_fn` and rewrites the conversation in place
after each turn. That one needs the accompanying core edit to `rollout_factory.py` that
threads `conversation` through the hook (documented in the module's own docstring), so
it's real but not something to demo standalone the way the two functions above are.

## 6. A simpler, dependency-light QA toolkit

`rag/datagen.py` is deliberately decoupled from the retrieval/reward stack above; its
own `Chunk` dataclass, no `SearchBackend` dependency:

```python
from agenttune.rag.datagen import answer_correctness_reward, token_f1

answer_correctness_reward("Paris is the capital.", "Paris", mode="f1")   # SQuAD-style F1
token_f1("Paris is the capital.", "Paris")                                # same scorer, direct
```

`generate_qa_from_corpus(chunks, generator)` and the difficulty-curriculum helpers
(`label_difficulty`, `balance_by_difficulty`, `sort_by_difficulty`) all take injected
callables, so they're testable with a fake generator/solver and swappable to a real LLM
without touching the calling code.

## 7. Trajectory storage

`rag/storage/trajectory_store.py`'s `TrajectoryStore` is a stdlib-`sqlite3`-only,
pickle-safe (stores only the file path) persistent log of every agent rollout: question,
gold answer, final answer, reward + reward components, per-step thoughts/actions/
observations, and tool calls.

```python
from agenttune.rag.storage.trajectory_store import TrajectoryStore, TrajectoryRecord

store = TrajectoryStore("trajectories.db")
store.register_run("run-1", run_type="eval", model="Qwen/Qwen3-0.6B", condition="plain")

record = TrajectoryRecord(
    run_id="run-1",
    question="When should new employees complete security training?",
    gold_answer="Within the first week.",
    final_answer="Within the first week.",
    reward=1.0, reward_components={"format": 1.0, "answer": 1.0},
    n_tool_calls=1, has_answer_tag=True, condition="plain", model="Qwen/Qwen3-0.6B",
    steps=[{
        "step_number": 0, "thought": "search the corpus", "action_name": "search_corpus",
        "action_args": {"query": "security training"}, "observation": "found 2 chunks",
        "is_tool_step": True,
        "tool_calls": [{"tool_name": "search_corpus", "query": "security training",
                        "result": "found 2 chunks"}],
    }],
)
trajectory_id = store.store(record)

store.get_by_question(record.question)             # list[dict] — every rollout for this question
store.get_by_reward_range(0.5, 1.0)                 # list[dict] — for curriculum/dataset building
store.get_full_trajectory(trajectory_id)            # dict with nested steps + tool_calls
store.stats()                                       # {'total_trajectories': 1, 'avg_reward': 1.0, ...}
store.export_jsonl("trajectories.jsonl")            # full JSONL dump
```

`make_trajectory_callback(store, run_id, model, ...)` builds a ready-to-use
`on_trajectory_end` callback for `create_rollout_fn`; pass it and every rollout during
training/eval gets logged automatically, no manual `.store()` calls needed.

## 8. Bring-your-own-documents CLI

`rag/scripts/build_index_from_docs.py` is a real, argparse-backed `__main__` entry
point, verified against the current source, including the exact flag names (the module's
own docstring says `--input-dir`; the real argparse flag is `--input_dir` with an
underscore, use the one below):

```bash
# From a directory of mixed-format files:
python -m agenttune.rag.scripts.build_index_from_docs \
    --backend sqlite \
    --input_dir /path/to/documents \
    --output_dir rag_experiments/indexes/custom \
    --recursive

# From specific files:
python -m agenttune.rag.scripts.build_index_from_docs \
    --backend sqlite \
    --files doc1.pdf doc2.docx doc3.pptx \
    --output_dir rag_experiments/indexes/custom
```

Plain text/Markdown/CSV/JSON need nothing extra; PDF/DOCX/PPTX/HTML each need one
optional library (see `document_loaders.get_supported_formats()` for the current list).
It writes a `manifest.json` (doc/chunk counts, formats seen, chunk size) alongside the
index. Under the hood it's just `document_loaders.load_documents`/`load_file` →
`corpus_loader.build_index` → whichever backend you chose, the same two functions shown
in [§1](#1-lexical-search-sqliteftsbackend) above, so nothing here is CLI-only magic.

## Notes & gotchas

- **`retrieval_recall()` in `synthesis/evaluate.py` used to crash if a search actually
  returned a hit** (`r.metadata.get("chunk_id")`: dot access on what is a plain `dict`
  at runtime, since `SearchResult` is a `TypedDict`, not a real class). **Fixed**, now
  reads `r["metadata"].get("chunk_id")` like `check_retrieval_necessity` in `verify.py`
  already did a few lines away in the same package. Verified with a real hit:

  ```python
  >>> retrieval_recall([sample], backend)   # backend.search() returns >= 1 result
  {'retrieval_recall_at_5_mean': 1.0}
  ```
- **`build_index_from_docs.py`'s own module docstring says `--input-dir`; the real flag
  is `--input_dir`.** argparse option strings are literal; `--input-dir` will fail with
  "unrecognized arguments". Verified against the actual `argparse.ArgumentParser` calls
  in the file.
- **`BaseEvaluator`-style debug printing isn't unique to `eval/`**: `run_pipeline`'s
  stage functions print real progress (`[stage0] built + cached: ...`), which is useful,
  but there's no way to silence it short of redirecting stdout.

For the full list of known gaps across the whole repo, see
[Known Issues](../community/known-issues.md).
