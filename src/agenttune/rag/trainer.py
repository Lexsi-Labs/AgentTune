"""
create_rag_trainer — one-call agentic-RAG training.

Every RAG example/notebook in this repo (``examples/rag_training_real.py``,
``temp/rag_train.ipynb``) repeats the same boilerplate before it ever reaches
``create_agentic_trainer``: build a retrieval backend, index a corpus into it,
wrap it in ``SearchCorpusTool``, reshape the QA rows onto a ``prompt``/``answer``
schema, and assemble a RAG-flavored ``reward_funcs``/``system_prompt`` pair.
``create_rag_trainer`` does all of that internally and hands the rest straight
through to ``create_agentic_trainer`` — no script, no manual tool/reward wiring.

    trainer = create_rag_trainer(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        corpus=[{"doc_id": "policy", "title": "Founding", "text": "..."}],
        train_dataset=[{"prompt": "Who founded ...?", "answer": "Elena Cho"}],
        output_dir="./out_rag",
        max_steps=20,
    )
    trainer.train()
"""

from __future__ import annotations

import tempfile
from typing import Any

from datasets import Dataset

from .retrieval.base import SearchBackend
from .retrieval.corpus_loader import CorpusDocument, build_index
from .retrieval.sqlite_fts import SQLiteFTSBackend
from .tools.search_corpus import SearchCorpusTool

_DEFAULT_REWARD_FUNCS = ["search_grounding_reward", "format_reward", "answer_correctness_reward"]
_DEFAULT_REWARD_WEIGHTS = [1.0, 0.2, 1.0]
_DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant. Call the `search_corpus` tool to find relevant "
    "passages before answering. Cite the chunk_id you relied on, and put your final "
    "answer in <answer></answer> tags."
)
# Aliases folded onto the schema create_agentic_trainer's default RAG rewards expect:
# `prompt` (the GRPO rollout input) and `answer` (what answer_correctness_reward reads).
_COLUMN_ALIASES = {"question": "prompt", "query": "prompt", "gold_answer": "answer"}


def _build_backend(retrieval_type: str, db_path: str | None, match_all: bool) -> SearchBackend:
    if retrieval_type == "sqlite_fts":
        path = db_path or tempfile.mktemp(suffix=".sqlite", prefix="agenttune_rag_")
        return SQLiteFTSBackend(path, match_all=match_all)
    if retrieval_type == "chroma":
        from .retrieval.chroma_backend import ChromaBackend

        path = db_path or tempfile.mkdtemp(prefix="agenttune_rag_chroma_")
        return ChromaBackend(path)
    raise ValueError(
        f"Unsupported retrieval_type {retrieval_type!r}. Choose 'sqlite_fts' or 'chroma'."
    )


def _to_corpus_documents(corpus: list[Any]) -> list[CorpusDocument]:
    docs = []
    for i, d in enumerate(corpus):
        if isinstance(d, CorpusDocument):
            docs.append(d)
        elif isinstance(d, dict):
            docs.append(
                CorpusDocument(
                    doc_id=str(d.get("doc_id", i)),
                    title=str(d.get("title", d.get("doc_id", f"doc_{i}"))),
                    text=d["text"],
                    metadata=d.get("metadata", {}),
                )
            )
        else:
            raise TypeError(
                f"corpus entries must be dict or CorpusDocument, got {type(d).__name__}"
            )
    return docs


def _normalize_dataset(train_dataset: Any, column_mapping: dict[str, str] | None) -> Dataset:
    rows = train_dataset.to_list() if isinstance(train_dataset, Dataset) else list(train_dataset)
    mapping = {**_COLUMN_ALIASES, **(column_mapping or {})}
    normalized = []
    for row in rows:
        row = dict(row)
        for src, dst in mapping.items():
            if src in row and dst not in row:
                row[dst] = row.pop(src)
        if "prompt" not in row:
            raise ValueError(
                f"Row missing a 'prompt' column after applying column_mapping={mapping}: {row}"
            )
        normalized.append(row)
    return Dataset.from_list(normalized)


def create_rag_trainer(
    model: str,
    train_dataset: Any,
    *,
    corpus: list[Any] | None = None,
    retrieval_backend: SearchBackend | None = None,
    retrieval_type: str = "sqlite_fts",
    db_path: str | None = None,
    match_all: bool = False,
    top_k: int = 5,
    chunk_size: int = 400,
    chunk_overlap: int = 0,
    column_mapping: dict[str, str] | None = None,
    algorithm: str = "grpo",
    backend: str = "auto",
    reward_funcs: Any = None,
    reward_weights: list[float] | None = None,
    system_prompt: str | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Build a search-augmented agentic RAG trainer in one call.

    Handles retrieval-backend construction, corpus indexing, search-tool wiring,
    and dataset/reward/system-prompt defaults; everything else (GRPO/DPO/PPO/RLOO/BCO
    knobs, LoRA, backend selection, Hub push) is forwarded verbatim to
    ``create_agentic_trainer``.

    Parameters
    ----------
    model : HF repo id or local path for the policy model.
    train_dataset : list[dict] | datasets.Dataset
        QA rows. Needs a `prompt` (or `question`/`query`) column and an `answer`
        (or `gold_answer`) column; any other column (e.g. `gold_chunk_ids`) passes
        through untouched.
    corpus : list[dict | CorpusDocument], optional
        Raw documents (`doc_id`, `title`, `text`) to chunk + index into a fresh
        retrieval backend. Ignored if `retrieval_backend` is given.
    retrieval_backend : an already-built-and-indexed SearchBackend to reuse
        instead of building one from `corpus`.
    retrieval_type : "sqlite_fts" (default, lexical/BM25) | "chroma" (vector).
    db_path : where to persist a newly-built backend's index (temp path if omitted).
    match_all : SQLiteFTSBackend query semantics — AND (True) vs OR+bm25 (False,
        default; required for natural-language-ish queries).
    top_k : passages returned per search call.
    chunk_size / chunk_overlap : corpus chunking knobs (only used with `corpus=`).
    column_mapping : extra dataset column renames layered on top of the built-in
        question->prompt / query->prompt / gold_answer->answer aliases.
    algorithm / backend : forwarded to create_agentic_trainer (default "grpo"/"auto").
    reward_funcs / reward_weights : any callable(s), `REWARD_REGISTRY` string name(s), or
        a mix — passed straight through to `create_agentic_trainer`/`combine_rewards`,
        exactly like calling `create_agentic_trainer` directly. Not limited to built-ins;
        pass your own scoring function(s) (`def my_reward(prompts, completions, **kw)`)
        freely. Defaults to
        `["search_grounding_reward", "format_reward", "answer_correctness_reward"]`
        with weights `[1.0, 0.2, 1.0]` only when both are omitted.
    system_prompt : defaults to a RAG-flavored prompt matching the default rewards'
        expectations (search before answering, cite chunk_id, <answer> tags). Override
        freely if you bring your own reward_funcs with different conventions.
    tools : any additional tool(s) — a `BaseTool` instance or plain Python callable,
        your own custom tools included — alongside the auto-built `search_corpus` tool.
    **kwargs : forwarded verbatim to create_agentic_trainer (max_steps, peft_config,
        num_generations, output_dir, ...).

    Returns
    -------
    Whatever create_agentic_trainer returns (has `.train()`, hub-push helpers).
    The retrieval backend and search tool are attached as `trainer.retrieval_backend`
    / `trainer.search_tool` for inspection/reuse.
    """
    # Imported lazily so `import agenttune.rag` (and pure retrieval usage — building/
    # querying a SearchBackend, no training) never pulls in the TRL/torch training
    # stack that `create_agentic_trainer` needs.
    from ..core.backend_factory import create_agentic_trainer

    if retrieval_backend is not None:
        search_backend = retrieval_backend
    else:
        if not corpus:
            raise ValueError(
                "create_rag_trainer needs either `corpus=` (to build+index a new "
                "retrieval backend) or an already-indexed `retrieval_backend=`."
            )
        search_backend = _build_backend(retrieval_type, db_path, match_all)
        build_index(
            search_backend,
            _to_corpus_documents(corpus),
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

    search_tool = SearchCorpusTool(search_backend, top_k=top_k)
    all_tools = [search_tool, *(tools or [])]

    ds = _normalize_dataset(train_dataset, column_mapping)

    trainer = create_agentic_trainer(
        algorithm,
        backend=backend,
        model=model,
        train_dataset=ds,
        tools=all_tools,
        reward_funcs=reward_funcs if reward_funcs is not None else list(_DEFAULT_REWARD_FUNCS),
        reward_weights=(
            reward_weights if reward_weights is not None else list(_DEFAULT_REWARD_WEIGHTS)
        ),
        system_prompt=system_prompt or _DEFAULT_SYSTEM_PROMPT,
        **kwargs,
    )
    trainer.retrieval_backend = search_backend
    trainer.search_tool = search_tool
    return trainer
