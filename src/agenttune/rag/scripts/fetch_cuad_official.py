"""
Download + verify the official CUAD-QA release, then optionally write the
official train/test JSONs for build_cuad_dataset.py.

The official release is the Atticus-Project/cuad GitHub `data.zip`:
  - CUADv1.json                 — all 510 contracts (SQuAD format, 1 paragraph
                                  per doc, context == full contract text,
                                  answer spans with doc-relative offsets)
  - train_separate_questions.json — official TRAIN split (408 contracts,
                                    22,450 questions)
  - test.json                   — official TEST split (102 contracts,
                                  4,182 questions)

Verified on 2026-08-13: the split is perfectly contract-disjoint, all
13,823 answer spans resolve exactly against their context, and all 41
clause categories are encoded in the question ids (`<title>__<Category>[_<n>]`).

This script re-runs those verification checks on every download (so an
updated release can't silently break the split), then extracts the JSONs.

Usage:
    python -m agenttune.rag.scripts.fetch_cuad_official \\
        --out_dir ~/cuad_data [--zip_url ...]
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile

from ..data.cuad import (
    OFFICIAL_TEST_CONTRACTS,
    OFFICIAL_TRAIN_CONTRACTS,
    OFFICIAL_ZIP_URL,
    category_from_qa_id,
)


def _load_data(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


def verify_official_release(json_dir: str) -> dict:
    """Re-run the release checks; returns a report dict. Raises on failure."""
    train_path = os.path.join(json_dir, "train_separate_questions.json")
    test_path = os.path.join(json_dir, "test.json")
    full_path = os.path.join(json_dir, "CUADv1.json")
    for p in (train_path, test_path, full_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing — run this script to download")

    train = _load_data(train_path)
    test = _load_data(test_path)
    full = _load_data(full_path)

    tr_titles = {d["title"] for d in train}
    te_titles = {d["title"] for d in test}
    full_titles = {d["title"] for d in full}
    overlap = tr_titles & te_titles
    if overlap:
        raise AssertionError(
            f"official split NOT contract-disjoint: "
            f"{len(overlap)} overlapping: {sorted(overlap)[:5]}"
        )
    if len(tr_titles) != OFFICIAL_TRAIN_CONTRACTS or len(te_titles) != OFFICIAL_TEST_CONTRACTS:
        raise AssertionError(
            f"official split sizes changed: train={len(tr_titles)} "
            f"(expected {OFFICIAL_TRAIN_CONTRACTS}), test={len(te_titles)} "
            f"(expected {OFFICIAL_TEST_CONTRACTS})"
        )
    if full_titles != tr_titles | te_titles:
        raise AssertionError("split titles don't cover exactly the CUADv1 set")

    # span integrity + category coverage on the full file
    n_spans = n_bad = 0
    categories = set()
    for doc in full:
        if len(doc["paragraphs"]) != 1:
            raise AssertionError(f"{doc['title']}: expected exactly 1 paragraph")
        ctx = doc["paragraphs"][0]["context"]
        for qa in doc["paragraphs"][0]["qas"]:
            categories.add(category_from_qa_id(qa["id"]))
            for ans in qa["answers"]:
                n_spans += 1
                s, t = ans["answer_start"], ans["text"]
                if ctx[s : s + len(t)] != t:
                    n_bad += 1
                    if n_bad <= 3:
                        print(f"  [warn] span mismatch in {doc['title']}: {t[:40]!r}")
    if n_bad:
        raise AssertionError(f"{n_bad}/{n_spans} answer spans do not resolve")
    return {
        "train_contracts": len(tr_titles),
        "test_contracts": len(te_titles),
        "train_questions": sum(len(p["qas"]) for d in train for p in d["paragraphs"]),
        "test_questions": sum(len(p["qas"]) for d in test for p in d["paragraphs"]),
        "span_overlap_train_test": 0,
        "answer_spans_checked": n_spans,
        "answer_spans_bad": n_bad,
        "categories": len(categories),
    }


def main():
    ap = argparse.ArgumentParser(description="Download + verify official CUAD-QA release")
    ap.add_argument("--out_dir", default="~/cuad_data")
    ap.add_argument("--zip_url", default=OFFICIAL_ZIP_URL)
    args = ap.parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    zip_path = os.path.join(out_dir, "data.zip")
    if not os.path.exists(zip_path):
        print(f"[download] {args.zip_url} -> {zip_path}")
        urllib.request.urlretrieve(args.zip_url, zip_path)
    else:
        print(f"[download] {zip_path} exists, skipping")

    print("[extract] unzipping...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    # the zip contains CUADv1.json / train_separate_questions.json / test.json
    # at its root — they may be nested under a subdir; move them up if so
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            if fn in ("CUADv1.json", "train_separate_questions.json", "test.json"):
                src = os.path.join(root, fn)
                dst = os.path.join(out_dir, fn)
                if src != dst and not os.path.exists(dst):
                    os.rename(src, dst)

    print("[verify] re-running release checks...")
    report = verify_official_release(out_dir)
    print(f"[verify] OK: {report}")
    with open(os.path.join(out_dir, "verification.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] official CUAD-QA data ready in {out_dir}")


if __name__ == "__main__":
    main()
