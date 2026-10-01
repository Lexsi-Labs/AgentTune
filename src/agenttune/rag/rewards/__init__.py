"""Phase 1 reward functions, T3 (necessity+frugality) rewards, and QA metrics."""

from .finder_rewards import (
    conciseness_reward,
    get_logged_finder_reward,
    golden_chunk_recall_reward,
    numeric_correctness_reward,
)
from .judge_eval import build_groq_judge, score_groundedness
from .phase1_rewards import (
    format_reward,
    get_training_reward,
    rag_correctness_reward,
    search_usage_reward,
)
from .qa_metrics import exact_match_score, extract_answer_tag, f1_score
from .t3_rewards import (
    frugality_reward,
    get_combined_reward,
    get_m2_reward,
    get_t3_reward,
    necessity_reward,
)

__all__ = [
    "exact_match_score",
    "extract_answer_tag",
    "f1_score",
    "format_reward",
    "get_training_reward",
    "rag_correctness_reward",
    "search_usage_reward",
    # T3 (Sprint 2)
    "necessity_reward",
    "frugality_reward",
    "get_t3_reward",
    "get_combined_reward",
    # M2 (Sprint 3)
    "get_m2_reward",
    "build_groq_judge",
    "score_groundedness",
    # FinDER (E1, FinNLP paper)
    "numeric_correctness_reward",
    "golden_chunk_recall_reward",
    "conciseness_reward",
    "get_logged_finder_reward",
]
