"""
agenttune.rag — Agentic RAG use-case package.

A self-contained application built on top of agenttune's agentic training
framework: a document search environment (multiple retrieval backends),
search tools, RAG-specific rewards, and HotpotQA data/training/eval scripts.

Nothing here modifies agenttune's core training logic — everything is
consumed through public entry points (`create_agentic_trainer`,
`create_rollout_fn`, `BaseTool`, `combine_rewards`, `LLMJudge`).
"""

from .trainer import create_rag_trainer  # noqa: F401

__all__ = ["create_rag_trainer"]
