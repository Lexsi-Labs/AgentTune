"""
Stage 4 — 2D difficulty labeling + balanced sampling.

Labels every surviving sample with (hop_count, retrieval_difficulty) where
`retrieval_difficulty = 1 - power_mean_p(cosine_sim(question, each_supporting_chunk))`
(GRADE's formula; p=-3 power-mean emphasizes the WEAKEST/hardest-to-retrieve
chunk, so a question far from any single support scores high difficulty).

Then bins into a 2D matrix and resamples to balance cells (Know Your RAG:
naive generation produces imbalanced datasets; balance explicitly).
"""

from __future__ import annotations

import math
import random

from .schema import QASample

_POWER_P = -3.0  # GRADE: negative p emphasizes the min (weakest chunk match)


def _power_mean(values, p=_POWER_P):
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    try:
        if p == 0:
            return math.exp(sum(math.log(v) for v in vals) / len(vals))
        return (sum(v**p for v in vals) / len(vals)) ** (1.0 / p)
    except (OverflowError, ZeroDivisionError):
        return min(vals)


def _cosine(a, b):
    import numpy as np

    a, b = np.array(a), np.array(b)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def label_difficulty(
    samples: list[QASample], chunks_by_id, embedder, *, stage="stage4"
) -> list[QASample]:
    """Compute retrieval_difficulty for each sample (embeds the question once).

    Reuses the injected embedder (Qwen3 on A100 in prod; fake in tests).
    """
    if not samples:
        return samples
    q_texts = [s.question for s in samples]
    q_vecs = embedder.embed_queries(q_texts)
    for s, qv in zip(samples, q_vecs, strict=False):
        sims = []
        for cid in s.gold_chunk_ids:
            ch = chunks_by_id.get(cid)
            if ch and ch.embedding:
                sims.append(_cosine(qv, ch.embedding))
        pm = _power_mean(sims, _POWER_P) if sims else 0.0
        s.retrieval_difficulty = max(0.0, min(1.0, 1.0 - pm))
        # provisional cell (legacy 2-level split; run_pipeline re-buckets via
        # reassign_difficulty_cells once the run distribution is known)
        band = "hard" if s.retrieval_difficulty >= 0.5 else "easy"
        s.difficulty_cell = f"{s.hop_count}hop_{band}"
    return samples


def reassign_difficulty_cells(
    samples: list[QASample], *, quantiles=(1 / 3, 2 / 3)
) -> list[QASample]:
    """Re-bucket difficulty_cell into 3 levels calibrated to THIS run.

    The fixed 0.5 hard threshold is degenerate on legal text: questions are
    derived from their gold chunks, so embedding similarity stays high and
    almost everything lands "easy" (verified: mean retrieval_difficulty 0.26
    on the 4-contract smoke run, all cells easy). Instead, buckets are
    quantiles of the run's own retrieval_difficulty distribution:
      easy < q1, medium < q2, hard >= q2. Calibrated per corpus/run — the
    same formula, a threshold that means something local. Called by
    run_pipeline after label_difficulty; standalone callers (tests) that use
    label_difficulty directly keep the legacy 2-level behavior.
    """
    vals = sorted(s.retrieval_difficulty or 0.0 for s in samples)
    n = len(vals)
    if n == 0:
        return samples
    q1 = vals[min(n - 1, int(quantiles[0] * n))]
    q2 = vals[min(n - 1, int(quantiles[1] * n))]
    for s in samples:
        v = s.retrieval_difficulty or 0.0
        band = "easy" if v < q1 else ("medium" if v < q2 else "hard")
        s.difficulty_cell = f"{s.hop_count}hop_{band}"
    return samples


def balance_by_matrix(samples: list[QASample], *, target_per_cell=20, seed=42) -> list[QASample]:
    """Resample to balance the (hop x difficulty) matrix (Know Your RAG).

    Caps each cell at `target_per_cell` (downsamples over-represented cells)
    and returns the balanced subset. Doesn't upsample (no duplication) — just
    removes imbalance, keeping the pipeline honest.
    """
    rng = random.Random(seed)
    cells: dict = {}
    for s in samples:
        cells.setdefault(s.difficulty_cell or "unknown", []).append(s)
    out = []
    for _cell, group in cells.items():
        rng.shuffle(group)
        out.extend(group[:target_per_cell])
    return out
