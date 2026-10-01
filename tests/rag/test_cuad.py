"""CPU tests for the CUAD loader and the same-document graph guard.

Fakes only — no network/key. Mirrors test_synthesis.py's style.
"""

import os
import tempfile

from agenttune.rag.data.cuad import (
    build_corpus_from_cuad,
    load_cuad_contracts,
    split_contracts,
)
from agenttune.rag.synthesis import (
    Chunk,
    FakeLLMClient,
    build_graph,
)


def _write_fake_contracts(tmpdir):
    # Two unrelated contracts (different "companies"), each with internal
    # cross-references (a defined term reused later in the same doc).
    os.makedirs(os.path.join(tmpdir, "Part_I"))
    with open(os.path.join(tmpdir, "Part_I", "AcmeCorp_ServicesAgreement.txt"), "w") as f:
        f.write(
            'This Services Agreement is between Acme Corp (the "Provider") '
            'and Globex Inc (the "Recipient"). Provider shall deliver the '
            "Services described in Exhibit A. "
            + ("filler text. " * 40)
            + "Any dispute regarding the Services shall be resolved under "
            "Section 9, referencing the obligations defined above."
        )
    with open(os.path.join(tmpdir, "Part_I", "Umbrella_LLC_LicenseAgreement.txt"), "w") as f:
        f.write(
            'This License Agreement is between Umbrella LLC (the "Licensor") '
            'and Initech Ltd (the "Licensee"). Licensor grants a license to '
            "the Product. "
            + ("filler text. " * 40)
            + "Termination of this license follows the notice period defined "
            "for the Product license above."
        )
    return tmpdir


def test_load_cuad_contracts_reads_all_txt_recursively():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fake_contracts(tmp)
        contracts = load_cuad_contracts(tmp)
        assert len(contracts) == 2
        ids = {c["contract_id"] for c in contracts}
        assert ids == {"AcmeCorp_ServicesAgreement", "Umbrella_LLC_LicenseAgreement"}
        assert all(c["text"].strip() for c in contracts)


def test_split_contracts_is_deterministic_and_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fake_contracts(tmp)
        contracts = load_cuad_contracts(tmp)
        train1, test1 = split_contracts(contracts, val_fraction=0.5, seed=0)
        train2, test2 = split_contracts(contracts, val_fraction=0.5, seed=0)
        assert [c["contract_id"] for c in train1] == [c["contract_id"] for c in train2]
        assert [c["contract_id"] for c in test1] == [c["contract_id"] for c in test2]
        train_ids = {c["contract_id"] for c in train1}
        test_ids = {c["contract_id"] for c in test1}
        assert not (train_ids & test_ids), "split must be contract-disjoint"


def test_build_corpus_from_cuad_doc_ids_match_contract_ids():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fake_contracts(tmp)
        contracts = load_cuad_contracts(tmp)
        docs = build_corpus_from_cuad(contracts)
        assert {d.doc_id for d in docs} == {c["contract_id"] for c in contracts}


# ── The regression test that matters most: same_doc_only really blocks
#    cross-document edges, even when two chunks from DIFFERENT documents
#    share an entity (the exact failure mode verified as real on CUAD in
#    NLLP_SynthData.md §A0 — coincidental entity overlap across unrelated
#    contracts must never become an edge). ──────────────────────────────────


def _cross_doc_chunks_with_shared_entity():
    # Two chunks, two DIFFERENT doc_ids, but they share an entity ("delaware")
    # the way two unrelated contracts both mentioning "Delaware" would.
    a = Chunk(
        chunk_id="docA::0",
        doc_id="docA",
        title="A",
        text="Governed by the laws of Delaware.",
        chunk_index=0,
        entities=["delaware", "governing law"],
    )
    b = Chunk(
        chunk_id="docB::0",
        doc_id="docB",
        title="B",
        text="This agreement is construed under Delaware law.",
        chunk_index=0,
        entities=["delaware", "governing law"],
    )
    # plus one within-document pair (docA has 2 chunks sharing an entity)
    c = Chunk(
        chunk_id="docA::1",
        doc_id="docA",
        title="A",
        text="Delaware courts have exclusive jurisdiction.",
        chunk_index=1,
        entities=["delaware", "jurisdiction"],
    )
    for ch in (a, b, c):
        ch.embedding = [0.1, 0.2, 0.3]
    return [a, b, c]


def test_same_doc_only_blocks_cross_document_exact_edges():
    chunks = _cross_doc_chunks_with_shared_entity()
    llm = FakeLLMClient(responder=lambda purpose, messages: '{"mappings": []}')
    edges, G = build_graph(chunks, llm, same_doc_only=True)
    cross_doc = [
        e
        for e in edges
        if next(c for c in chunks if c.chunk_id == e.source).doc_id
        != next(c for c in chunks if c.chunk_id == e.target).doc_id
    ]
    assert not cross_doc, f"same_doc_only=True must block cross-doc edges, found: {cross_doc}"
    # the within-document pair (docA::0 <-> docA::1) must still get an edge
    within_doc = [e for e in edges if e.source == "docA::0" and e.target == "docA::1"]
    assert within_doc, "within-document edges must still be built"


def test_same_doc_only_false_preserves_existing_behavior():
    # Default (same_doc_only=False, used by HotpotQA/FinDER) must NOT be
    # restricted — the cross-document edge (docA::0 <-> docB::0) should exist.
    chunks = _cross_doc_chunks_with_shared_entity()
    llm = FakeLLMClient(responder=lambda purpose, messages: '{"mappings": []}')
    edges, G = build_graph(chunks, llm, same_doc_only=False)
    cross_doc = [e for e in edges if e.source == "docA::0" and e.target == "docB::0"]
    assert cross_doc, "default behavior must be unchanged — cross-doc edges still form"


# ── Per-document entity-frequency filtering (the fix for the real over-
#    connection found on CUAD: a contract's own party name appears in most
#    of its chunks and otherwise creates a near-complete subgraph). ────────


def test_ubiquitous_entity_does_not_dominate_same_doc_graph():
    from agenttune.rag.synthesis.graph import build_graph

    # 20 chunks in ONE document (large enough that the min-count floor
    # doesn't exempt it, matching the real CUAD scale where this bug
    # appeared on 100-250 chunk contracts). 18/20 chunks share "acmecorp"
    # (the party name — ubiquitous, should be dropped for edge purposes,
    # 90% frequency and well above the 5-chunk floor). Only chunks 0/1
    # share a SPECIFIC, rare term ("net 30 payment terms", 2/20 = 10%,
    # below the floor) that should survive and produce the edge.
    chunks = []
    for i in range(20):
        ents = []
        if i < 18:
            ents.append("acmecorp")
        if i in (0, 1):
            ents.append("net 30 payment terms")
        chunks.append(
            Chunk(
                chunk_id=f"doc::{i}",
                doc_id="doc",
                title="doc",
                text=f"passage {i}",
                chunk_index=i,
                entities=ents,
            )
        )
    llm = FakeLLMClient(responder=lambda purpose, messages: '{"mappings": []}')
    edges, G = build_graph(chunks, llm, same_doc_only=True, max_entity_doc_frequency=0.3)
    exact = [e for e in edges if e.type == "exact"]
    # only the 0-1 pair (via the rare term) should get an edge; the
    # ubiquitous "acmecorp" (18/20 = 90% > 0.3 threshold, cnt=18 >= floor=5)
    # must not connect every pair among the 18 chunks that have it.
    assert (
        len(exact) == 1
    ), f"expected exactly 1 edge (the rare-term pair), got {len(exact)}: {exact}"
    assert {exact[0].source, exact[0].target} == {"doc::0", "doc::1"}


def test_max_entity_doc_frequency_is_noop_when_same_doc_only_false():
    from agenttune.rag.synthesis.graph import build_graph

    # Same setup, but same_doc_only=False (default/HotpotQA/FinDER behavior)
    # must NOT apply the per-document frequency filter.
    chunks = []
    for i in range(6):
        chunks.append(
            Chunk(
                chunk_id=f"doc::{i}",
                doc_id="doc",
                title="doc",
                text=f"passage {i}",
                chunk_index=i,
                entities=["acmecorp"],
            )
        )
    llm = FakeLLMClient(responder=lambda purpose, messages: '{"mappings": []}')
    edges, G = build_graph(chunks, llm, same_doc_only=False)
    exact = [e for e in edges if e.type == "exact"]
    # C(6,2) = 15 pairs, all sharing "acmecorp" -> all 15 edges, unfiltered
    assert len(exact) == 15


# ── sample_paths must not starve on one dense endpoint pair (the real bug
#    found on CUAD: a 202-node densely-connected component inside one
#    contract burned the entire path budget on 1-2 pairs, leaving only 6
#    total paths across 14 contracts / 984 chunks). ─────────────────────────


def test_sample_paths_does_not_starve_on_one_dense_pair():
    import networkx as nx

    from agenttune.rag.synthesis.paths import sample_paths
    from agenttune.rag.synthesis.schema import Chunk as _Chunk

    # A dense complete-ish graph between node 0 and node 9 (many alternate
    # routings of length 2-5), PLUS a handful of other simple 2-3 hop pairs
    # elsewhere in the same graph. Without the per-pair cap, enumerating all
    # routes between 0 and 9 alone can exhaust `max_paths_considered` before
    # the other pairs are ever tried.
    G = nx.complete_graph(10)  # nodes 0..9, all pairs directly connected +
    # every intermediate routing exists
    G.add_edge(9, 10)
    G.add_edge(10, 11)  # a distinct 2-hop pair (9,11) outside the dense core
    chunks_by_id = {
        i: _Chunk(chunk_id=i, doc_id="doc", title="doc", text=f"chunk {i}", chunk_index=i)
        for i in G.nodes
    }
    paths = sample_paths(G, chunks_by_id, per_hop=5, seed=0, max_paths_considered=200)
    endpoint_pairs = {(p.chunk_ids[0], p.chunk_ids[-1]) for p in paths}
    # with the fix, we should see paths from MORE than just one endpoint pair
    assert len(endpoint_pairs) > 1, (
        f"path budget starved on a single endpoint pair — got paths only "
        f"between: {endpoint_pairs}"
    )


# ── Official CUAD-QA JSON loader + split (NLLP_SynthData.md A8 step 1) ─────
# The official release ships SQuAD-format JSONs (CUADv1.json /
# train_separate_questions.json / test.json). Verified against the real
# release on 2026-08-13: 1 paragraph per doc (context == full contract),
# answer spans resolve exactly, 41 categories encoded as
# `<title>__<Category>[_<n>]`, and the official split is contract-disjoint
# (408 train / 102 test, zero overlap).


def _write_squad_json(tmpdir, docs, name="cuad.json"):
    """docs: list of {"title", "context", "qas": [{"id", "question", "answers", "is_impossible"}]}"""
    import json as _json

    path = tmpdir / name
    path.write_text(_json.dumps({"version": "aok_v1.0", "data": docs}))
    return str(path)


def _squad_doc(title, context, qas):
    return {"title": title, "paragraphs": [{"context": context, "qas": qas}]}


def test_load_cuad_squad_json_records_and_categories(tmp_path):
    from agenttune.rag.data.cuad import load_cuad_squad_json

    ctx = (
        "This Services Agreement is between Acme Corp and Globex Inc. "
        "Acme shall indemnify Globex for breaches. Either party may "
        "terminate for convenience on 30 days written notice."
    )
    qas = [
        {
            "id": "ACME_SERVICES__Parties_0",
            "question": "q1",
            "answers": [{"text": "Acme Corp", "answer_start": 44}],
            "is_impossible": False,
        },
        {
            "id": "ACME_SERVICES__Termination For Convenience_1",
            "question": "q2",
            "answers": [{"text": "30 days", "answer_start": 148}],
            "is_impossible": False,
        },
    ]
    path = _write_squad_json(tmp_path, [_squad_doc("ACME_SERVICES", ctx, qas)])
    records = load_cuad_squad_json(path)
    assert len(records) == 1
    r = records[0]
    assert r["contract_id"] == "ACME_SERVICES"
    assert r["text"] == ctx
    assert [q["category"] for q in r["qas"]] == ["Parties", "Termination For Convenience"]
    assert r["qas"][0]["answers"][0] == {"text": "Acme Corp", "answer_start": 44}


def test_load_cuad_squad_json_rejects_multi_paragraph(tmp_path):
    import pytest

    from agenttune.rag.data.cuad import load_cuad_squad_json

    doc = {"title": "T", "paragraphs": [{"context": "a", "qas": []}, {"context": "b", "qas": []}]}
    path = _write_squad_json(tmp_path, [doc])
    with pytest.raises(ValueError):
        load_cuad_squad_json(path)


def test_category_from_qa_id_handles_both_formats():
    from agenttune.rag.data.cuad import category_from_qa_id

    assert category_from_qa_id("LIMEENERGYCO_1999__Parties") == "Parties"
    assert category_from_qa_id("LIMEENERGYCO_1999__Parties_3") == "Parties"
    assert category_from_qa_id("X__No-Solicit Of Employees_14") == "No-Solicit Of Employees"
    assert category_from_qa_id("no_separator") == ""


def test_official_split_contracts_disjoint_and_counted(tmp_path, monkeypatch):
    from agenttune.rag.data.cuad import (
        official_split_contracts,
    )

    # 3 train + 2 test contracts, disjoint by construction; monkeypatch the
    # expected counts to the fixture sizes so the real assert logic runs.
    monkeypatch.setattr("agenttune.rag.data.cuad.OFFICIAL_TRAIN_CONTRACTS", 3)
    monkeypatch.setattr("agenttune.rag.data.cuad.OFFICIAL_TEST_CONTRACTS", 2)
    tr = _write_squad_json(
        tmp_path,
        [
            _squad_doc(
                "T1",
                "text one",
                [
                    {
                        "id": "T1__Parties",
                        "question": "q",
                        "answers": [{"text": "x", "answer_start": 0}],
                        "is_impossible": False,
                    }
                ],
            ),
            _squad_doc("T2", "text two", []),
            _squad_doc("T3", "text three", []),
        ],
        name="train.json",
    )
    te = _write_squad_json(
        tmp_path,
        [_squad_doc("E1", "text four", []), _squad_doc("E2", "text five", [])],
        name="test.json",
    )
    tr_ids, te_ids = official_split_contracts(tr, te)
    assert tr_ids == ["T1", "T2", "T3"]
    assert te_ids == ["E1", "E2"]
    assert not (set(tr_ids) & set(te_ids))


def test_official_split_contracts_fails_on_overlap(tmp_path, monkeypatch):
    import pytest

    from agenttune.rag.data.cuad import official_split_contracts

    monkeypatch.setattr("agenttune.rag.data.cuad.OFFICIAL_TRAIN_CONTRACTS", 1)
    monkeypatch.setattr("agenttune.rag.data.cuad.OFFICIAL_TEST_CONTRACTS", 1)
    doc = _squad_doc(
        "SAME",
        "text",
        [
            {
                "id": "SAME__Parties",
                "question": "q",
                "answers": [{"text": "x", "answer_start": 0}],
                "is_impossible": False,
            }
        ],
    )
    tr = _write_squad_json(tmp_path, [doc])
    te = _write_squad_json(tmp_path, [doc])
    with pytest.raises(AssertionError):
        official_split_contracts(tr, te)


# ── Category tagging (NLLP_SynthData.md §A2, metadata) ─────────────────────


def test_tag_chunks_with_categories_overlap_semantics(tmp_path):
    from agenttune.rag.data.cuad import (
        gold_chunk_categories,
        labeled_spans_from_records,
        load_cuad_squad_json,
        tag_chunks_with_categories,
    )
    from agenttune.rag.retrieval.corpus_loader import CorpusDocument
    from agenttune.rag.synthesis import chunk_documents

    text = (
        "SECTION 1. Parties. Acme Corp and Globex Inc are the parties. "
        "SECTION 2. Termination. Either party may terminate for "
        "convenience on 30 days notice. SECTION 3. Governing Law. "
        "This agreement is governed by Delaware law."
    )
    qas = [
        {
            "id": "DOC__Parties_0",
            "question": "q",
            "answers": [{"text": "Acme Corp", "answer_start": text.index("Acme Corp")}],
            "is_impossible": False,
        },
        {
            "id": "DOC__Termination For Convenience_0",
            "question": "q",
            "answers": [
                {
                    "text": "terminate for convenience",
                    "answer_start": text.index("terminate for convenience"),
                }
            ],
            "is_impossible": False,
        },
    ]
    # build records through the real loader so categories come from the qa ids
    # (same path the CLI uses — hand-built dicts would skip this and break)
    records = load_cuad_squad_json(_write_squad_json(tmp_path, [_squad_doc("DOC", text, qas)]))
    spans = labeled_spans_from_records(records)
    # chunk with offsets; the splitter may produce 2-3 pieces — don't assert a
    # fixed count, assert the SEMANTICS: whichever chunk holds the Parties
    # span text carries "Parties", whichever holds the Termination span text
    # carries "Termination For Convenience"
    doc = CorpusDocument(doc_id="DOC", title="DOC", text=text)
    chunks = chunk_documents([doc], chunk_size=len(text) // 2, overlap=0, with_offsets=True)
    tag_chunks_with_categories(chunks, spans)
    party_chunk = next(c for c in chunks if "Acme Corp" in c.text)
    term_chunk = next(c for c in chunks if "terminate for convenience" in c.text)
    assert "Parties" in party_chunk.metadata["cuad_categories"]
    assert "Termination For Convenience" in term_chunk.metadata["cuad_categories"]
    # gold_chunk_categories: union across a question's gold chunks
    by_id = {c.chunk_id: c for c in chunks}
    assert gold_chunk_categories(by_id, [party_chunk.chunk_id, term_chunk.chunk_id]) == sorted(
        set(party_chunk.metadata["cuad_categories"]) | set(term_chunk.metadata["cuad_categories"])
    )


def test_tag_chunks_no_offsets_gets_empty_tags():
    from agenttune.rag.data.cuad import tag_chunks_with_categories
    from agenttune.rag.synthesis import Chunk

    c = Chunk(chunk_id="doc::0", doc_id="doc", text="no offsets here")
    tag_chunks_with_categories(
        [c],
        [{"contract_id": "doc", "category": "Parties", "start": 0, "end": 10, "text": "no offset"}],
    )
    assert c.metadata["cuad_categories"] == []


# ── Difficulty: 3-level quantile buckets (NLLP_SynthData.md A3/Stage 4) ─────


def test_reassign_difficulty_cells_three_levels():
    from agenttune.rag.synthesis import QASample, reassign_difficulty_cells

    samples = []
    for i, v in enumerate([0.1, 0.15, 0.2, 0.5, 0.55, 0.9]):
        s = QASample(sample_id=f"s{i}", hop_count=3)
        s.retrieval_difficulty = v
        s.difficulty_cell = f"3hop_{'hard' if v >= 0.5 else 'easy'}"
        samples.append(s)
    reassign_difficulty_cells(samples)
    cells = {s.sample_id: s.difficulty_cell for s in samples}
    assert cells["s0"] == "3hop_easy" and cells["s1"] == "3hop_easy"
    assert cells["s2"] == "3hop_medium" and cells["s3"] == "3hop_medium"
    assert cells["s4"] == "3hop_hard" and cells["s5"] == "3hop_hard"
    # all three levels present (the fixed-0.5 split could never do this)
    assert {c.split("_")[1] for c in cells.values()} == {"easy", "medium", "hard"}


# ── Relaxed answerability scorer + gate ────────────────────────────────────


def test_relaxed_f1_handles_legal_entity_phrasing():
    from agenttune.rag.rewards.qa_metrics import f1_score, relaxed_f1_score

    # Note: SQuAD's own normalization already strips articles and punctuation,
    # so strict F1 on these pairs is 0.8-1.0, NOT 0. The relaxed scorer's real
    # edge over strict is suffix/alias tokens ("Corporation" vs "Corp") — the
    # case where a solver's correct-but-differently-phrased answer would still
    # lose credit under strict scoring.
    cases = [
        ("The ARC Group, Inc.", "ARC Group"),  # leading article + suffix
        ("Tel-Aviv, Israel", "Tel Aviv"),  # hyphen + punctuation
        ("arc group", "ARC Group"),  # case
        ("Provider", "the Provider"),  # article
    ]
    for pred, gold in cases:
        assert relaxed_f1_score(pred, gold) >= 0.8, (pred, gold)
    # suffix tokens: strict loses credit (0.5), relaxed gives full (1.0)
    assert f1_score("Arizona Corporation", "Arizona Corp") == 0.5
    assert relaxed_f1_score("Arizona Corporation", "Arizona Corp") == 1.0
    # genuinely different answers still score ~0 under BOTH scorers
    assert relaxed_f1_score("Kansas", "Delaware") == 0.0
    assert f1_score("Kansas", "Delaware") == 0.0


def test_answerability_sets_answerable_gate():
    from agenttune.rag.synthesis import Chunk, FakeLLMClient, QASample
    from agenttune.rag.synthesis.evaluate import answerability

    # solver answers correctly but with paraphrased phrasing — strict F1 low,
    # relaxed high → answerable must be True. Genuinely wrong → False.
    chunks = [
        Chunk(
            chunk_id="doc::0",
            doc_id="doc",
            text="The ARC Group, Inc. shall indemnify Globex for "
            "losses arising from a breach of representations.",
        )
    ]
    answers = {
        "Which party indemnifies Globex?": "The ARC Group, Inc.",
        "Is the indemnity capped?": "Yes",
        "Which state law governs?": "Delaware",
    }
    golds = {
        "Which party indemnifies Globex?": "ARC Group",
        "Is the indemnity capped?": "Yes",
        "Which state law governs?": "Kansas",
    }
    samples = []
    for q in answers:
        s = QASample(sample_id=f"s-{q}", gold_chunk_ids=["doc::0"])
        s.question = q  # the solver prompt needs the actual question
        s.answer = golds[q]
        samples.append(s)

    def responder(purpose, messages):
        import json as _json

        q = messages[-1]["content"].split("Question: ")[1].split("\n")[0].strip()
        return _json.dumps({"answer": answers.get(q, "UNANSWERABLE")})

    llm = FakeLLMClient(responder=responder)
    metrics = answerability(samples, {c.chunk_id: c for c in chunks}, llm)
    # ARC Group (solver used the suffix form): relaxed gives full credit,
    # strict loses some → answerable must be True on the relaxed gate
    assert samples[0].answerability_f1_relaxed >= 0.8
    assert samples[0].answerable is True
    # Yes: both high
    assert samples[1].answerable is True
    # Delaware vs Kansas: genuinely wrong → not answerable
    assert samples[2].answerable is False
    assert metrics["answerable_rate"] == 2 / 3
    assert metrics["answerability_pass_rate_relaxed"] == 2 / 3
