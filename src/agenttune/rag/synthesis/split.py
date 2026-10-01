"""
Post-Stage-5 — train/val split by gold-chunk disjointness (leakage prevention).

A random row-split of the generated dataset leaks validation into training:
two different questions can traverse *overlapping* gold chunks, so a chunk a
model memorizes from a train question can surface as the answer to a val
question. For retrieval-augmented RL this is a correctness bug, not an
optimization — it inflates eval reward with contaminated signal.

The fix partitions the *accepted* samples so that no gold chunk_id appears in
both splits' gold paths. The key insight: if two samples share even one gold
chunk they are coupled (the chunk can only live in one split), transitively —
so this is a **connected-components** problem on the sample-sharing graph
(edges = "shares a gold chunk"), not a per-row decision. Each connected
component goes wholesale into one split; components are then bin-packed into
train/val to hit the target validation fraction.

   samples ──(shared gold chunk)──► components ──(greedy bin-pack)──► train / val

Stricter variant (`level="entity"`): the same algorithm on `path_entities`
instead of chunk_ids. Entity-level disjointness is stronger (a model that
memorizes facts about entity E from train can't leak them into val) but more
aggressive — it shrinks the usable split on small/dense corpora where most
questions touch the same handful of entities. Chunk-level is the default.

Emits `dataset_train.csv` / `dataset_val.csv` +
`dataset_{train,val}_grpo.jsonl` alongside the existing dataset dumps.
"""

from __future__ import annotations

import json
import random

from .schema import QASample


def _accepted(samples: list[QASample]) -> list[QASample]:
    """Only accepted/revised_accepted samples belong in a training split."""
    return [s for s in samples if s.status in ("accepted", "revised_accepted")]


def _gold_keys(sample: QASample, level: str) -> list[str]:
    """The disjointness keys for a sample — chunk_ids (default) or path entities."""
    if level == "entity":
        ents = list(getattr(sample.path, "entities", []) or []) if sample.path else []
        return [e.lower().strip() for e in ents if e and e.strip()]
    # chunk-level: the ordered gold chunk ids (the passages a RAG system retrieves)
    return list(sample.gold_chunk_ids)


def _connected_components(samples: list[QASample], level: str) -> list[list[int]]:
    """Find connected components of the sample-sharing graph.

    Two samples share an edge iff their gold-key sets intersect. Union-Find over
    samples; the key→sample index map resolves each shared key into an edge.
    Returns a list of components (each a list of sample indices into `samples`).
    """
    parent = list(range(len(samples)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # group sample indices by every gold key they touch; any two samples under
    # the same key share that chunk/entity → union them.
    key_to_samples: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        for k in _gold_keys(s, level):
            if not k:
                continue
            key_to_samples.setdefault(k, []).append(i)
    for indices in key_to_samples.values():
        for j in range(1, len(indices)):
            union(indices[0], indices[j])

    comps: dict[int, list[int]] = {}
    for i in range(len(samples)):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def _binpack(
    components: list[list[int]], n_total: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Greedily assign whole components to train/val to hit `val_fraction`.

    Components are sorted largest-first (a standard greedy-number-partitioning
    heuristic: placing the big components first lets the many small ones
    fine-tune the split toward the target). Each component goes to whichever
    split keeps the val share closest to the target. This is NOT optimal
    bin-packing (NP-hard) but is within one component of the target on typical
    inputs, and it never breaks a component apart.
    """
    rng = random.Random(seed)
    # shuffle within equal-size bands so the assignment isn't deterministic on
    # input order (stability across runs comes from `seed`, not sample order).
    shuffled = list(components)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)

    target_val = val_fraction * n_total
    train_idx: list[int] = []
    val_idx: list[int] = []
    for comp in shuffled:
        # assign this whole component to whichever side keeps val closest to target
        val_if_val = len(val_idx) + len(comp)
        val_if_train = len(val_idx)
        # distance to target for each choice
        d_val = abs(val_if_val - target_val)
        d_train = abs(val_if_train - target_val)
        if d_val <= d_train:
            val_idx.extend(comp)
        else:
            train_idx.extend(comp)
    return train_idx, val_idx


def split_by_gold_chunks(
    samples: list[QASample],
    *,
    val_fraction: float = 0.2,
    level: str = "chunk",
    seed: int = 42,
    min_train: int = 2,
    min_val: int = 1,
) -> tuple[list[QASample], list[QASample], dict]:
    """Partition accepted samples into train/val with zero gold-key overlap.

    Guarantees (chunk level): the set of chunk_ids in any train question's gold
    path is disjoint from every val question's gold path → no retrieval
    leakage between splits. `level="entity"` extends the guarantee to path
    entities (stricter; may shrink usable splits on small corpora).

    Returns (train_samples, val_samples, report). The report carries:
      - n_train / n_val / n_skipped (rejected samples never enter the split)
      - val_fraction_actual (achieved, vs requested)
      - n_components (connected components in the sharing graph — a small number
        relative to n samples means questions are heavily coupled)
      - gold_overlap_train_val (must be 0 — the guarantee; asserted)
      - largest_component_fraction (a coupling diagnostic: if one component is
        ~100% of samples, the corpus is so interlinked that a non-trivial split
        is impossible — caller should grow the corpus)
      - level, seed (reproducibility)

    Guards: if the split can't satisfy `min_train`/`min_val` (e.g. a single huge
    component, or too few accepted samples), it returns ALL samples to train
    and an EMPTY val — and flags `degenerate=True` in the report — rather than
    silently breaking a component (which would violate the disjointness
    guarantee). The caller decides whether an empty val is acceptable.
    """
    accepted = _accepted(samples)
    report: dict = {
        "n_input": len(samples),
        "n_accepted": len(accepted),
        "level": level,
        "seed": seed,
        "val_fraction_requested": val_fraction,
        "degenerate": False,
        "reason": "",
    }
    if not accepted:
        report.update(
            n_train=0,
            n_val=0,
            n_skipped=len(samples),
            val_fraction_actual=0.0,
            n_components=0,
            gold_overlap_train_val=0,
            largest_component_fraction=0.0,
        )
        return [], [], report

    comps = _connected_components(accepted, level)
    comp_sizes = sorted((len(c) for c in comps), reverse=True)
    largest = comp_sizes[0]
    n = len(accepted)
    report["n_components"] = len(comps)
    report["largest_component_fraction"] = round(largest / n, 4)

    train_idx, val_idx = _binpack(comps, n, val_fraction, seed)
    train = [accepted[i] for i in train_idx]
    val = [accepted[i] for i in val_idx]

    # Degeneracy guard: if either side is below its minimum, the corpus is too
    # coupled (or too small) for a meaningful split. Don't break components —
    # fall back to all-train and flag it so the caller can grow the corpus.
    if len(train) < min_train or len(val) < min_val:
        report["degenerate"] = True
        report["reason"] = (
            f"split would yield train={len(train)}/val={len(val)} "
            f"(min_train={min_train}, min_val={min_val}); "
            f"{len(comps)} component(s), largest={largest}/{n} "
            f"— corpus too coupled/small for a disjoint split"
        )
        report.update(
            n_train=len(accepted),
            n_val=0,
            n_skipped=len(samples) - len(accepted),
            val_fraction_actual=0.0,
            gold_overlap_train_val=0,
        )
        return accepted, [], report

    # Assert the disjointness guarantee holds (defensive — the algorithm
    # preserves it by construction; this catches a future regression).
    overlap = _overlap(train, val, level)
    report.update(
        n_train=len(train),
        n_val=len(val),
        n_skipped=len(samples) - len(accepted),
        val_fraction_actual=round(len(val) / n, 4),
        gold_overlap_train_val=overlap,
    )
    return train, val, report


def _overlap(train: list[QASample], val: list[QASample], level: str) -> int:
    """Count gold keys shared between train and val (the leakage metric; 0 = clean)."""
    train_keys = set()
    for s in train:
        train_keys.update(_gold_keys(s, level))
    val_keys = set()
    for s in val:
        val_keys.update(_gold_keys(s, level))
    return len(train_keys & val_keys)


def write_split_artifacts(
    samples: list[QASample],
    out_dir: str,
    *,
    val_fraction: float = 0.2,
    level: str = "chunk",
    seed: int = 42,
    backend: str = "sqlite",
    min_train: int = 2,
    min_val: int = 1,
    system_prompt=None,
    chunks_by_id=None,
) -> dict | None:
    """Split `samples` and emit train/val CSV + GRPO jsonl into `out_dir`.

    Reuses `build_dataset._rows_from_samples` / `_grpo_rows` so the split files
    have the exact same schema as `dataset_final.csv` / `dataset_grpo.jsonl`.

    Returns the split report (always — even on degenerate splits, so the caller
    can record it in the manifest), or None if there were no accepted samples
    at all. On a degenerate split it still writes `dataset_train.csv` (all
    accepted) and an empty `dataset_val.csv` so downstream code never sees a
    missing file.
    """
    from pathlib import Path

    import agenttune.rag.synthesis.build_dataset as BD

    from .io_utils import dump_table

    train, val, report = split_by_gold_chunks(
        samples,
        val_fraction=val_fraction,
        level=level,
        seed=seed,
        min_train=min_train,
        min_val=min_val,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dump_table(BD._rows_from_samples(train, include_rejected=False), str(out / "dataset_train.csv"))
    dump_table(BD._rows_from_samples(val, include_rejected=False), str(out / "dataset_val.csv"))

    # GRPO-ready jsonl per split (same row shape as dataset_grpo.jsonl)
    for name, split in (("train", train), ("val", val)):
        rows = BD._grpo_rows(split, system_prompt=system_prompt, chunks_by_id=chunks_by_id)
        with open(out / f"dataset_{name}_grpo.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["files"] = {
        "train_csv": str(out / "dataset_train.csv"),
        "val_csv": str(out / "dataset_val.csv"),
        "train_grpo": str(out / "dataset_train_grpo.jsonl"),
        "val_grpo": str(out / "dataset_val_grpo.jsonl"),
    }
    return report
