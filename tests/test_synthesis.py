"""CPU tests for the synthetic QA pipeline — fakes, no network/key/GPU.

Mirrors test_datagen.py's injectable-callable style: a scripted FakeLLMClient
and FakeEmbedder exercise every stage's logic deterministically. If these pass,
the only remaining risk on Colab is the real-LLM/model behavior, not our code.
"""

import json

from agenttune.rag.retrieval.corpus_loader import CorpusDocument
from agenttune.rag.synthesis import (
    Chunk,
    FakeEmbedder,
    FakeLLMClient,
    QASample,
    ReasoningPath,
    accumulate_and_append,
    annotate_chunks,
    balance_by_matrix,
    build_graph,
    chunk_documents,
    embed_chunks,
    generate_batch,
    label_difficulty,
    load_existing_questions,
    run_pipeline,
    sample_paths,
    split_by_gold_chunks,
    verify_batch,
    write_split_artifacts,
)


def _docs():
    # A small connected corpus so the graph has real paths.
    return [
        CorpusDocument(
            doc_id="A", title="History", text="Northwind Robotics was founded in 2015 by Elena Cho."
        ),
        CorpusDocument(
            doc_id="B", title="Moves", text="Elena Cho joined Solace AI as CTO in 2021."
        ),
        CorpusDocument(
            doc_id="C",
            title="Partners",
            text="Solace AI and Northwind Robotics co-developed Aria-2 in 2023.",
        ),
        CorpusDocument(
            doc_id="D", title="Decoy", text="Elena Marsh founded Marsh Dynamics in 2016."
        ),
    ]


def _scripted_llm():
    """FakeLLM whose response depends on the call purpose."""

    def responder(purpose, messages):
        if purpose == "extract_entities":
            txt = messages[0]["content"]
            if "Northwind" in txt and "founded" in txt:
                return json.dumps(
                    {
                        "entities": ["northwind robotics", "elena cho", "2015"],
                        "keyphrases": ["founded"],
                        "summary": "Northwind founded by Elena Cho.",
                    }
                )
            if "Solace AI as CTO" in txt:
                return json.dumps(
                    {
                        "entities": ["elena cho", "solace ai", "2021"],
                        "keyphrases": ["joined", "cto"],
                        "summary": "Elena Cho joined Solace AI.",
                    }
                )
            if "Aria-2" in txt:
                return json.dumps(
                    {
                        "entities": ["solace ai", "northwind robotics", "aria-2", "2023"],
                        "keyphrases": ["co-developed"],
                        "summary": "Solace and Northwind built Aria-2.",
                    }
                )
            return json.dumps(
                {
                    "entities": ["elena marsh", "marsh dynamics"],
                    "keyphrases": ["founded"],
                    "summary": "Elena Marsh founded Marsh Dynamics.",
                }
            )
        if purpose == "contextual_equiv":
            # Claim a cross-chunk coreference only where it's real (aria / aria-2)
            return json.dumps({"mappings": []})
        if purpose == "generate_answer_first":
            return json.dumps(
                {
                    "answer": "Aria-2",
                    "question": "What product did the company co-developed by its own CTO's prior firm release?",
                    "reasoning": "Cho founded Northwind; Cho joined Solace; Solace+Northwind built Aria-2.",
                }
            )
        if purpose.startswith("chain_dep_mask_hop"):
            return json.dumps(
                {"answer": "Aria-2"}
            )  # solver leaks the answer -> hop not load-bearing
        if purpose.startswith("solver") or purpose == "eval_answerability":
            return json.dumps({"answer": "Aria-2"})
        if purpose == "revise_question":
            return json.dumps(
                {"question": "Revised question requiring all three hops.", "answer": "Aria-2"}
            )
        return "FAKE_RESPONSE"

    return FakeLLMClient(responder=responder)


# ── Stage 0 ──────────────────────────────────────────────────────────────────


def test_chunk_documents_reuses_chunker():
    chunks = chunk_documents(_docs(), chunk_size=512, overlap=0)
    assert len(chunks) >= 4
    assert all(c.chunk_id and c.text for c in chunks)
    assert all(c.doc_id in {"A", "B", "C", "D"} for c in chunks)


def test_annotate_and_embed():
    chunks = chunk_documents(_docs(), chunk_size=512, overlap=0)
    chunks = annotate_chunks(chunks, _scripted_llm())
    chunks = embed_chunks(chunks, FakeEmbedder())
    # entity extraction populated; embeddings populated
    assert any("elena cho" in c.entities for c in chunks)
    assert all(c.embedding is not None and len(c.embedding) == 64 for c in chunks)


def test_build_graph_has_exact_edges():
    chunks = chunk_documents(_docs(), chunk_size=512, overlap=0)
    chunks = annotate_chunks(chunks, _scripted_llm())
    chunks = embed_chunks(chunks, FakeEmbedder())
    edges, G = build_graph(chunks, _scripted_llm())
    exact = [e for e in edges if e.type == "exact"]
    # "elena cho" links A-B; "solace ai"/"northwind robotics" link B-C and A-C
    assert len(exact) >= 2
    assert G.number_of_nodes() == len(chunks)


# ── Stage 1 ──────────────────────────────────────────────────────────────────


def test_sample_paths_returns_multi_hop():
    chunks = chunk_documents(_docs(), chunk_size=512, overlap=0)
    chunks = annotate_chunks(chunks, _scripted_llm())
    chunks = embed_chunks(chunks, FakeEmbedder())
    _, G = build_graph(chunks, _scripted_llm())
    by_id = {c.chunk_id: c for c in chunks}
    paths = sample_paths(G, by_id, per_hop=10)
    assert paths, "expected at least one sampled path"
    assert all(2 <= p.hop_count <= 5 for p in paths)
    assert all(len(p.chunk_ids) == p.hop_count + 1 for p in paths)


# ── Stage 2 ──────────────────────────────────────────────────────────────────


def test_generate_batch_answer_first():
    chunks = chunk_documents(_docs(), chunk_size=512, overlap=0)
    chunks = annotate_chunks(chunks, _scripted_llm())
    chunks = embed_chunks(chunks, FakeEmbedder())
    _, G = build_graph(chunks, _scripted_llm())
    by_id = {c.chunk_id: c for c in chunks}
    paths = sample_paths(G, by_id, per_hop=5)
    samples = generate_batch(paths, by_id, _scripted_llm())
    assert all(s.question and s.answer for s in samples)
    assert all(s.gold_chunk_ids == p.chunk_ids for s, p in zip(samples, paths, strict=False))
    # call metadata captured
    assert all(len(s.llm_calls) >= 1 for s in samples)


# ── Stage 3 ──────────────────────────────────────────────────────────────────


def test_verify_records_checks_and_retry():
    chunks = chunk_documents(_docs(), chunk_size=512, overlap=0)
    chunks = annotate_chunks(chunks, _scripted_llm())
    chunks = embed_chunks(chunks, FakeEmbedder())
    _, G = build_graph(chunks, _scripted_llm())
    by_id = {c.chunk_id: c for c in chunks}
    paths = sample_paths(G, by_id, per_hop=5)
    samples = generate_batch(paths, by_id, _scripted_llm())
    import os
    import tempfile

    from agenttune.rag.retrieval.corpus_loader import build_index as _bi
    from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend

    db = os.path.join(tempfile.mkdtemp(), "c.sqlite")
    sb = SQLiteFTSBackend(db)
    _bi(sb, [CorpusDocument(doc_id=c.doc_id, title=c.title, text=c.text) for c in chunks])
    verified = verify_batch(samples, by_id, sb, _scripted_llm())
    assert all(v.status in ("accepted", "revised_accepted", "rejected") for v in verified)
    assert all(v.retrieval_necessity is not None for v in verified)
    # at least one sample recorded a chain-dependency probe or skipped (2-hop)
    assert all(v.chain_dependency is not None for v in verified)


# ── Stage 4 ──────────────────────────────────────────────────────────────────


def test_difficulty_and_balance():
    samples = [
        QASample(
            sample_id=f"q{i}",
            question=f"q {i}",
            answer="Aria-2",
            gold_chunk_ids=[f"c{i}"],
            hop_count=2 + (i % 3),
        )
        for i in range(10)
    ]
    for _i, s in enumerate(samples):
        s.status = "accepted"
    embedder = FakeEmbedder()
    by_id = {
        f"c{i}": Chunk(
            chunk_id=f"c{i}",
            text=f"chunk {i}",
            embedding=embedder.embed_documents([f"chunk {i}"])[0],
        )
        for i in range(10)
    }
    samples = label_difficulty(samples, by_id, embedder)
    assert all(s.retrieval_difficulty is not None for s in samples)
    assert all(s.difficulty_cell for s in samples)
    balanced = balance_by_matrix(samples, target_per_cell=2)
    assert len(balanced) <= len(samples)


# ── Full pipeline (fake) ─────────────────────────────────────────────────────


def test_run_pipeline_end_to_end_fake(tmp_path):
    llm = _scripted_llm()
    embedder = FakeEmbedder()
    run_pipeline(
        docs=_docs(),
        llm=llm,
        embedder=embedder,
        out_dir=str(tmp_path / "out"),
        backend="sqlite",
        per_hop=5,
        target_per_cell=5,
    )
    # artifacts written
    for name in [
        "stage0_chunks.csv",
        "stage0_graph_edges.csv",
        "stage1_paths.csv",
        "stage2_generated.csv",
        "stage3_verified.csv",
        "stage4_balanced.csv",
        "dataset_final.csv",
        "dataset_grpo.jsonl",
        "metrics.json",
        "cost_summary.json",
        "manifest.json",
    ]:
        assert (tmp_path / "out" / name).exists(), f"missing {name}"
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert "multi_hop_necessity_rate" in metrics
    assert "total_cost_usd" in metrics
    # Task 1: split artifacts now written by the pipeline
    for name in [
        "dataset_train.csv",
        "dataset_val.csv",
        "dataset_train_grpo.jsonl",
        "dataset_val_grpo.jsonl",
    ]:
        assert (tmp_path / "out" / name).exists(), f"missing split artifact {name}"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert "split" in manifest, "manifest missing split report"
    assert manifest["split"]["gold_overlap_train_val"] == 0


# ── Task 1: train/val split by gold-chunk disjointness ───────────────────────


def _accepted_sample(sid, gold, hop=2, ents=None):
    """Helper: build an accepted QASample with given gold chunk ids + path entities."""
    s = QASample(
        sample_id=sid,
        question=f"q {sid}",
        answer="Aria-2",
        gold_chunk_ids=list(gold),
        hop_count=hop,
        status="accepted",
    )
    s.path = ReasoningPath(
        path_id=f"p_{sid}", chunk_ids=list(gold), hop_count=hop, entities=ents or []
    )
    return s


def test_split_zero_chunk_overlap():
    # Two clusters: {A,B,C} linked by shared chunks, {D,E} linked, disjoint.
    # Use enough samples that a non-trivial split is achievable (tiny N makes
    # the bin-pack degenerate regardless of correctness).
    samples = [
        _accepted_sample("q0", ["A::0", "B::0"]),
        _accepted_sample("q1", ["B::0", "C::0"]),  # shares B::0 with q0
        _accepted_sample("q2", ["D::0", "E::0"]),  # separate cluster
        _accepted_sample("q3", ["F::0", "G::0"]),
        _accepted_sample("q4", ["H::0", "I::0"]),
    ]
    train, val, report = split_by_gold_chunks(samples, val_fraction=0.4, seed=42)
    assert report["gold_overlap_train_val"] == 0, "chunks leaked across splits"
    # q0 and q1 share B::0 → must be in the same split (component preserved)
    ids = {s.sample_id for s in train}
    assert ({"q0", "q1"} <= ids) or (
        {"q0", "q1"}.isdisjoint(ids)
    ), "coupled samples were split apart"
    assert len(train) + len(val) == 5
    assert report["n_components"] == 4
    assert report["degenerate"] is False
    assert len(val) >= 1 and len(train) >= 2


def test_split_degenerate_on_single_component():
    # All samples share one chunk → one component → can't split without breaking it
    samples = [_accepted_sample(f"q{i}", ["X::0", f"C{i}::0"]) for i in range(5)]
    train, val, report = split_by_gold_chunks(samples, val_fraction=0.4, seed=1, min_val=1)
    assert report["degenerate"] is True, "expected degenerate (single component)"
    assert val == [], "degenerate split must not break the component"
    assert len(train) == 5  # all returned to train


def test_split_entity_level_strict():
    samples = [
        _accepted_sample("q0", ["A::0", "B::0"], ents=["elena cho", "northwind"]),
        _accepted_sample("q1", ["C::0", "D::0"], ents=["solace ai"]),
        _accepted_sample("q2", ["E::0", "F::0"], ents=["elena cho"]),  # shares entity w/ q0
    ]
    train, val, report = split_by_gold_chunks(samples, level="entity", val_fraction=0.34, seed=2)
    assert report["gold_overlap_train_val"] == 0
    # q0 and q2 share entity "elena cho" → same component at entity level
    assert ({"q0", "q2"} <= {s.sample_id for s in train}) or (
        {"q0", "q2"} <= {s.sample_id for s in val}
    )


def test_write_split_artifacts_dumps_files(tmp_path):
    samples = [
        _accepted_sample("q0", ["A::0", "B::0"]),
        _accepted_sample("q1", ["B::0", "C::0"]),
        _accepted_sample("q2", ["D::0", "E::0"]),
        _accepted_sample("q3", ["F::0", "G::0"]),
    ]
    report = write_split_artifacts(samples, str(tmp_path / "s"), val_fraction=0.5, seed=7)
    assert report["gold_overlap_train_val"] == 0
    d = tmp_path / "s"
    for name in [
        "dataset_train.csv",
        "dataset_val.csv",
        "dataset_train_grpo.jsonl",
        "dataset_val_grpo.jsonl",
    ]:
        assert (d / name).exists(), f"missing {name}"
    # val jsonl rows parse and match the grpo schema
    import json as _json

    rows = [_json.loads(l) for l in (d / "dataset_val_grpo.jsonl").read_text().splitlines() if l]
    assert all("prompt" in r and "gold_answer" in r for r in rows)
    assert (
        sum(1 for l in (d / "dataset_train_grpo.jsonl").read_text().splitlines() if l) + len(rows)
        == 4
    )


def test_split_skips_rejected():
    samples = [
        _accepted_sample("q0", ["A::0", "B::0"]),
        _accepted_sample("q1", ["C::0", "D::0"]),
        _accepted_sample(
            "q2",
            ["E::0", "F::0"],
        ),  # mark rejected
    ]
    samples[2].status = "rejected"
    train, val, report = split_by_gold_chunks(samples, val_fraction=0.5, seed=3)
    assert report["n_skipped"] == 1
    assert len(train) + len(val) == 2


# ── Task 2: static-corpus append (dedup by question against existing + new) ──


def _make_run_pipeline_fn(sample_banks):
    """Build a fake run_pipeline that returns a different bank of samples per
    call (simulating seed-varied generation). `sample_banks` is a list of
    lists; each call pops the next bank. Each bank is a list of QASample."""
    state = {"call": 0}

    def fake_run_pipeline(**kw):
        bank = sample_banks[min(state["call"], len(sample_banks) - 1)]
        state["call"] += 1
        # write a stage0 cache marker so the append loop reuses it
        out_dir = kw["out_dir"]
        import os

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "stage0_cache.pkl"), "wb") as f:
            f.write(b"fake")
        return list(bank)

    return fake_run_pipeline


def test_load_existing_questions_only_accepted():
    import csv
    import os

    d = os.path.join(os.path.dirname(__file__), "_tmp_existing")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "dataset_all.csv")
    rows = [
        {
            "sample_id": "e0",
            "question": "Existing One",
            "answer": "A",
            "gold_chunk_ids": '["X::0"]',
            "hop_count": 2,
            "question_type": "bridge",
            "specificity": "specific",
            "status": "accepted",
            "retrieval_difficulty": "",
            "difficulty_cell": "2hop_easy",
            "answerability_f1": "1.0",
            "faithfulness_score": "",
        },
        {
            "sample_id": "e1",
            "question": "Rejected one",
            "answer": "B",
            "gold_chunk_ids": '["Y::0"]',
            "hop_count": 2,
            "question_type": "bridge",
            "specificity": "specific",
            "status": "rejected",
            "retrieval_difficulty": "",
            "difficulty_cell": "",
            "answerability_f1": "",
            "faithfulness_score": "",
        },
    ]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    seen, accepted_rows = load_existing_questions(p)
    assert "existing one" in seen
    assert "rejected one" not in seen  # rejected never counts as seen
    assert len(accepted_rows) == 2  # all rows returned; dedup is by status
    import shutil

    shutil.rmtree(d)


def test_accumulate_appends_dedup_against_existing(tmp_path):
    # Existing dataset has 1 accepted sample (q="Existing").
    import csv

    existing_csv = str(tmp_path / "dataset_all.csv")
    with open(existing_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "question",
                "answer",
                "gold_chunk_ids",
                "hop_count",
                "question_type",
                "specificity",
                "status",
                "retrieval_difficulty",
                "difficulty_cell",
                "answerability_f1",
                "faithfulness_score",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "sample_id": "e0",
                "question": "Existing",
                "answer": "A",
                "gold_chunk_ids": '["X::0","Y::0"]',
                "hop_count": 2,
                "question_type": "bridge",
                "specificity": "specific",
                "status": "accepted",
                "retrieval_difficulty": "0.3",
                "difficulty_cell": "2hop_easy",
                "answerability_f1": "1.0",
                "faithfulness_score": "",
            }
        )

    # Fake generation: iter0 returns 2 NEW unique + 1 dup of existing;
    # iter1 returns 1 more new. Target_total=4 → existing(1)+new(3)=4.
    bank0 = [
        _accepted_sample("n0", ["A::0", "B::0"]),  # new
        _accepted_sample("n1", ["C::0", "D::0"]),  # new
        _accepted_sample("dup0", ["X::0", "Y::0"], hop=2),  # dup of existing
    ]
    dup0 = _accepted_sample("dup0", ["X::0", "Y::0"], hop=2)
    dup0.question = "Existing"  # same question text → must dedup
    bank0[2] = dup0
    bank1 = [_accepted_sample("n2", ["E::0", "F::0"])]  # new
    rpf = _make_run_pipeline_fn([bank0, bank1])

    report = accumulate_and_append(
        run_pipeline_fn=rpf,
        docs=[],
        llm=None,
        embedder=None,
        out_dir=str(tmp_path / "out"),
        existing_csv=existing_csv,
        target_total=4,
        max_iters=5,
        per_hop=5,
        target_per_cell=5,
        base_seed=42,
    )

    assert report["n_existing_accepted"] == 1
    assert report["n_new_unique"] == 3  # n0, n1, n2 (dup dropped)
    assert report["n_total_after"] == 4
    # combined dataset_final.csv has 4 unique accepted
    rows = list(csv.DictReader(open(tmp_path / "out" / "dataset_final.csv")))
    questions = {r["question"].strip().lower() for r in rows}
    assert "existing" in questions
    assert len(rows) == 4
    # the duplicate did not sneak in
    assert sum(1 for q in questions if q == "existing") == 1
    # split over the combined set
    assert (tmp_path / "out" / "dataset_train.csv").exists()
    assert (tmp_path / "out" / "dataset_val.csv").exists()


def test_accumulate_coverage_delta_reported(tmp_path):
    # Existing covers chunks X,Y; new run covers X,Y,Z → report shows +1 new chunk
    existing_csv = str(tmp_path / "dataset_all.csv")
    import csv

    with open(existing_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "question",
                "answer",
                "gold_chunk_ids",
                "hop_count",
                "question_type",
                "specificity",
                "status",
                "retrieval_difficulty",
                "difficulty_cell",
                "answerability_f1",
                "faithfulness_score",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "sample_id": "e0",
                "question": "Existing",
                "answer": "A",
                "gold_chunk_ids": '["X::0","Y::0"]',
                "hop_count": 2,
                "question_type": "bridge",
                "specificity": "specific",
                "status": "accepted",
                "retrieval_difficulty": "0.3",
                "difficulty_cell": "2hop_easy",
                "answerability_f1": "",
                "faithfulness_score": "",
            }
        )
    bank0 = [_accepted_sample("n0", ["Y::0", "Z::0"])]  # adds Z, reuses Y
    rpf = _make_run_pipeline_fn([bank0])
    report = accumulate_and_append(
        run_pipeline_fn=rpf,
        docs=[],
        llm=None,
        embedder=None,
        out_dir=str(tmp_path / "out"),
        existing_csv=existing_csv,
        target_total=2,
        max_iters=1,
        per_hop=5,
        target_per_cell=5,
    )
    assert report["coverage_chunks_before"] == 2  # X, Y
    assert report["coverage_chunks_after"] == 3  # X, Y, Z
    assert report["n_new_chunks_covered"] == 1  # Z is new


def test_accumulate_missing_existing_starts_fresh(tmp_path):
    # No existing csv → starts from zero, just accumulates new
    bank0 = [_accepted_sample("n0", ["A::0", "B::0"]), _accepted_sample("n1", ["C::0", "D::0"])]
    rpf = _make_run_pipeline_fn([bank0])
    report = accumulate_and_append(
        run_pipeline_fn=rpf,
        docs=[],
        llm=None,
        embedder=None,
        out_dir=str(tmp_path / "out"),
        existing_csv=None,
        target_total=2,
        max_iters=3,
        per_hop=5,
        target_per_cell=5,
    )
    assert report["n_existing_accepted"] == 0
    assert report["n_new_unique"] == 2
    assert report["n_total_after"] == 2


# ── Task 3: Stage 0 caching control (only Stage 0 is cached, by design) ─────


def test_use_cache_false_ignores_existing_stage0_cache(tmp_path, caplog):
    """use_cache=False must rebuild Stage 0 even when stage0_cache.pkl exists.

    This is the staleness escape hatch: after a model/chunk-size change a stale
    cache would serve the old graph. Only Stage 0 is cached (Stages 1-5 are
    always fresh by design — see build_dataset.run_pipeline's docstring), so
    this test is the one caching-behavior check.
    """
    llm = _scripted_llm()
    embedder = FakeEmbedder()
    out = tmp_path / "out"
    # First run: builds + caches Stage 0
    run_pipeline(
        docs=_docs(),
        llm=llm,
        embedder=embedder,
        out_dir=str(out),
        backend="sqlite",
        per_hop=5,
        target_per_cell=5,
        use_cache=True,
    )
    cache = out / "stage0_cache.pkl"
    assert cache.exists(), "Stage 0 cache should be written"

    # Corrupt the cache so a cache-HIT would produce wrong chunk count
    import pickle

    with open(cache, "wb") as f:
        pickle.dump({"chunks": [], "edges": [], "graph": None}, f)

    # use_cache=False → must IGNORE the corrupted cache and rebuild.
    # run_pipeline logs via the `logging` module (logger.info), not print(),
    # so this must be captured with caplog rather than stdout redirection.
    with caplog.at_level("INFO"):
        run_pipeline(
            docs=_docs(),
            llm=llm,
            embedder=embedder,
            out_dir=str(out),
            backend="sqlite",
            per_hop=5,
            target_per_cell=5,
            use_cache=False,
        )
    log = caplog.text
    assert "built + cached" in log, "use_cache=False must rebuild Stage 0"
    assert "loaded from cache" not in log, "use_cache=False must not load the cache"

    # use_cache=True (default) → loads the (now-good, rebuilt) cache
    caplog.clear()
    run_pipeline(
        docs=_docs(),
        llm=llm,
        embedder=embedder,
        out_dir=str(out),
        backend="sqlite",
        per_hop=5,
        target_per_cell=5,
        use_cache=True,
    )
    # rebuilt cache on the False run is valid → this True run loads it
    chunks_after = out / "stage0_chunks.csv"
    assert chunks_after.exists()
