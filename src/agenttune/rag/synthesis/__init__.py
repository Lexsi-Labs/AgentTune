"""
agenttune.rag.synthesis — synthetic multi-hop RAG QA dataset generation.

Self-contained use-case package under `agenttune.rag`: docs in → multi-hop
question/answer dataset out, ready to feed `rag.scripts.train_grpo`.

Reuses the rag package's existing primitives (langchain chunker, SearchBackend,
SQuAD metrics, LLMJudge/Groq pattern) and adapts GRADE + RAGAS ideas (typed
graph, shortest-path sampling, 2D difficulty matrix) plus MHTS answer-first
generation and a novel closed-loop chain-dependency verifier (the Min et al.
2019 fix). Nothing here modifies agenttune's core.

See `synthetic_qa_plan.md` and the per-module docstrings for the design.
"""

from .append_dataset import accumulate_and_append, load_existing_questions
from .build_dataset import run_pipeline
from .curriculum import CurriculumSampler, build_curriculum, easy_to_hard_dataset
from .difficulty import balance_by_matrix, label_difficulty, reassign_difficulty_cells
from .embeddings import BGEM3Embedder, Embedder, FakeEmbedder, Qwen3Embedder, Qwen3Embedding06B
from .generate import _ANSWER_FIRST_PROMPT_LEGAL as ANSWER_FIRST_PROMPT_LEGAL
from .generate import generate_batch, generate_qa
from .graph import annotate_chunks, build_graph, chunk_documents, embed_chunks
from .io_utils import dump_table, extract_json
from .llm_client import FakeLLMClient, GroqLLMClient, LLMClient, OpenAICompatLLMClient
from .paths import sample_paths
from .requires_search_filter import filter_to_requires_search, label_requires_search
from .schema import Chunk, GraphEdge, LLMCallRecord, QASample, ReasoningPath
from .solve_difficulty import filter_contaminated, probe_solve_difficulty
from .split import split_by_gold_chunks, write_split_artifacts
from .verify import verify_batch, verify_sample

__all__ = [
    "Chunk",
    "GraphEdge",
    "ReasoningPath",
    "QASample",
    "LLMCallRecord",
    "LLMClient",
    "GroqLLMClient",
    "OpenAICompatLLMClient",
    "FakeLLMClient",
    "Embedder",
    "Qwen3Embedder",
    "BGEM3Embedder",
    "FakeEmbedder",
    "chunk_documents",
    "annotate_chunks",
    "embed_chunks",
    "build_graph",
    "sample_paths",
    "generate_qa",
    "generate_batch",
    "verify_sample",
    "verify_batch",
    "label_difficulty",
    "reassign_difficulty_cells",
    "balance_by_matrix",
    "split_by_gold_chunks",
    "write_split_artifacts",
    "accumulate_and_append",
    "load_existing_questions",
    "run_pipeline",
    "dump_table",
    "extract_json",
    "probe_solve_difficulty",
    "filter_contaminated",
    "label_requires_search",
    "filter_to_requires_search",
    "build_curriculum",
    "CurriculumSampler",
    "easy_to_hard_dataset",
]
