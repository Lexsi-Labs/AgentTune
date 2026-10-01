"""HotpotQA dataset loading and formatting."""

from .hotpotqa import (
    DEFAULT_SYSTEM_PROMPT,
    build_corpus_from_hotpotqa,
    load_hotpotqa_splits,
    to_grpo_dataset,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "build_corpus_from_hotpotqa",
    "load_hotpotqa_splits",
    "to_grpo_dataset",
]
