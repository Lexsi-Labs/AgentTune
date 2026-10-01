"""
T1 — Curriculum sampler for GRPO training.

Per rag_plan_s2.md §2/§6: R1-Searcher's two-stage curriculum pattern adapted
for our RAG training. Samples questions in easy→hard order, retires mastered
examples mid-training (contamination gate), and adjusts the difficulty mix
over training steps.

Two difficulty axes (keep both per plan §6):
  - retrieval_difficulty (GRADE, in difficulty.py): how hard to FIND evidence
  - solve_difficulty (R1-Searcher, in solve_difficulty.py): how hard to ANSWER

The curriculum sampler orders questions by combined difficulty and paces them
across training: start with easier questions, gradually introduce harder ones.
Mid-training, it re-probes the model and drops examples the model has mastered
(pass_rate >= threshold) — the contamination gate.

This module provides:
  - CurriculumSampler: wraps a dataset with difficulty-based pacing
  - build_curriculum: precompute difficulty labels + ordering
  - reprobe_and_filter: mid-training re-probe + contamination gate
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_curriculum(
    questions: list[dict[str, str]],
    solve_difficulty_labels: list[dict[str, Any]] | None = None,
    retrieval_difficulty_fn: callable | None = None,
) -> list[dict[str, Any]]:
    """Build a curriculum-ordered list of questions.

    Combines solve_difficulty (from T1 prober) and retrieval_difficulty (from
    GRADE scorer) into a single difficulty score, then sorts easy→hard.

    Args:
        questions: list of {"question": str, "answer": str}
        solve_difficulty_labels: output from probe_solve_difficulty (parallel to questions).
            If None, all questions get solve_difficulty=1.0 (assume hard).
        retrieval_difficulty_fn: callable(question) -> float 0..1. If None,
            all questions get retrieval_difficulty=0.5 (neutral).

    Returns:
        list of dicts with question, answer, solve_difficulty,
        retrieval_difficulty, combined_difficulty, curriculum_order (0=easiest)
    """
    if solve_difficulty_labels is None:
        solve_difficulty_labels = [{"solve_difficulty": 1.0} for _ in questions]

    curriculum = []
    for i, (q, sd) in enumerate(zip(questions, solve_difficulty_labels, strict=False)):
        solve_d = sd.get("solve_difficulty", 1.0)
        retr_d = retrieval_difficulty_fn(q["question"]) if retrieval_difficulty_fn else 0.5
        # Combined: weighted average (solve slightly more important for training signal)
        combined = 0.6 * solve_d + 0.4 * retr_d
        curriculum.append(
            {
                "question": q["question"],
                "answer": q["answer"],
                "solve_difficulty": solve_d,
                "retrieval_difficulty": retr_d,
                "combined_difficulty": combined,
                "original_index": i,
            }
        )

    # Sort easy→hard by combined difficulty
    curriculum.sort(key=lambda x: x["combined_difficulty"])
    for order, item in enumerate(curriculum):
        item["curriculum_order"] = order
    return curriculum


class CurriculumSampler:
    """Paces questions across training steps in easy→hard order.

    Usage:
        sampler = CurriculumSampler(curriculum, max_steps=200, batch_size=4)
        for step in range(max_steps):
            batch = sampler.sample(step)
            # train on batch...
            if step % 50 == 0 and step > 0:
                sampler.reprobe(model_path)  # mid-training re-probe
    """

    def __init__(
        self,
        curriculum: list[dict[str, Any]],
        max_steps: int = 200,
        batch_size: int = 4,
        warmup_ratio: float = 0.2,
        hard_ratio: float = 0.7,
    ):
        """
        Args:
            curriculum: output from build_curriculum (sorted easy→hard)
            max_steps: total training steps
            batch_size: questions per step
            warmup_ratio: fraction of steps that use only easy questions
            hard_ratio: fraction of steps after which all questions are available
        """
        self.curriculum = curriculum
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.full_steps = int(max_steps * hard_ratio)
        self.mastered: set = set()  # original_index of mastered questions
        self._step = 0

    def _available_pool(self, step: int) -> list[dict[str, Any]]:
        """Return the pool of questions available at this step."""
        if step < self.warmup_steps:
            # Warmup: only easiest 30%
            cutoff = max(1, int(len(self.curriculum) * 0.3))
            pool = self.curriculum[:cutoff]
        elif step < self.full_steps:
            # Ramp: progressively include harder questions
            progress = (step - self.warmup_steps) / max(1, self.full_steps - self.warmup_steps)
            cutoff = max(1, int(len(self.curriculum) * (0.3 + 0.7 * progress)))
            pool = self.curriculum[:cutoff]
        else:
            # Full: all questions available
            pool = self.curriculum

        # Filter out mastered questions (contamination gate)
        return [q for q in pool if q["original_index"] not in self.mastered]

    def sample(self, step: int) -> list[dict[str, str]]:
        """Sample a batch of questions for this training step.

        Returns a list of {"question": str, "answer": str} dicts.
        Sampling is deterministic within a step (uses step as seed) for reproducibility.
        """
        import random

        pool = self._available_pool(step)
        if not pool:
            pool = [q for q in self.curriculum if q["original_index"] not in self.mastered]
        if not pool:
            pool = self.curriculum  # fallback: all questions

        rng = random.Random(step)
        n = min(self.batch_size, len(pool))
        batch = rng.sample(pool, n)
        self._step = step
        return [{"question": q["question"], "answer": q["answer"]} for q in batch]

    def mark_mastered(self, question_indices: list[int]):
        """Mark questions as mastered (drop from future sampling)."""
        self.mastered.update(question_indices)

    def reprobe(
        self,
        model_path: str,
        questions: list[dict[str, str]] | None = None,
        k: int = 4,
        pass_threshold: float = 1.0,
        use_vllm: bool = True,
    ):
        """Mid-training re-probe: check which questions the model now solves.

        Marks questions with pass_rate >= threshold as mastered (contamination
        gate — drops them from future sampling). This is R1-Searcher's
        mid-training retirement of mastered examples.

        Args:
            model_path: current model (or adapter path) to probe
            questions: subset to probe (default: all non-mastered). If None,
                probes all questions in the curriculum.
            k: samples per question
            pass_threshold: drop if pass_rate >= this
            use_vllm: use vLLM for fast probing
        """
        from agenttune.rag.synthesis.solve_difficulty import probe_solve_difficulty

        if questions is None:
            questions = [
                {"question": q["question"], "answer": q["answer"]}
                for q in self.curriculum
                if q["original_index"] not in self.mastered
            ]

        if not questions:
            return []

        results = probe_solve_difficulty(questions, model_path, k=k, use_vllm=use_vllm)

        # Mark mastered questions
        newly_mastered = []
        for q, r in zip(questions, results, strict=False):
            if r["pass_rate"] >= pass_threshold:
                # Find original index
                for c in self.curriculum:
                    if c["question"] == q["question"]:
                        self.mastered.add(c["original_index"])
                        newly_mastered.append(c["original_index"])
                        break

        logger.info(
            f"[curriculum] re-probe at step {self._step}: "
            f"{len(newly_mastered)} newly mastered, "
            f"{len(self.mastered)} total mastered"
        )
        return newly_mastered

    def stats(self) -> dict[str, Any]:
        """Current curriculum state."""
        return {
            "total_questions": len(self.curriculum),
            "mastered": len(self.mastered),
            "available": len(
                [q for q in self.curriculum if q["original_index"] not in self.mastered]
            ),
            "current_step": self._step,
            "warmup_steps": self.warmup_steps,
            "full_steps": self.full_steps,
            "avg_difficulty": sum(q["combined_difficulty"] for q in self.curriculum)
            / len(self.curriculum),
        }


def easy_to_hard_dataset(
    curriculum: list[dict[str, Any]], split_ratio: float = 0.3
) -> tuple[list[dict], list[dict]]:
    """Split curriculum into easy (Stage 1) and full (Stage 2) sets.

    R1-Searcher's two-stage pattern: Stage 1 uses format-only reward on easy
    questions (350 Q), Stage 2 uses format+F1 on all questions (8148 Q).
    This function returns the split.
    """
    split_idx = int(len(curriculum) * split_ratio)
    easy = curriculum[:split_idx]
    full = curriculum
    return easy, full
