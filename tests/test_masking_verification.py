"""
Masking-verification tests.

The pure-logic helpers (word overlap scoring) are tested here without a GPU.
The full `run_masking_check` (Phase 0's actual proof — loads a real model,
runs a rollout, decodes env_mask spans) requires a GPU-capable environment
with `transformers`/`torch` and is marked `gpu` (see pytest.ini's registered
`gpu` marker) — it runs as part of the Colab experiment matrix, not in the
local/CPU test pass.
"""

import pytest

from agenttune.rag.scripts.verify_masking import _overlap_ratio, _word_set


def test_word_set_lowercases_and_tokenizes():
    assert _word_set("Paris, France!") == {"paris", "france"}


def test_overlap_ratio_full_overlap():
    assert _overlap_ratio("the eiffel tower is in paris", "paris") == 1.0


def test_overlap_ratio_no_overlap():
    assert _overlap_ratio("completely unrelated text", "paris") == 0.0


def test_overlap_ratio_empty_reference_is_zero():
    assert _overlap_ratio("some text", "") == 0.0


def test_overlap_ratio_partial():
    ratio = _overlap_ratio("paris is nice", "paris and london")
    assert 0.0 < ratio < 1.0


@pytest.mark.gpu
def test_run_masking_check_on_real_model(tmp_path):
    """Runs on Colab as part of the experiment matrix — requires a GPU-capable
    environment with transformers/torch and a built index."""
    import os

    from agenttune.rag.retrieval.corpus_loader import CorpusDocument, build_index
    from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
    from agenttune.rag.scripts.verify_masking import run_masking_check

    backend = SQLiteFTSBackend(str(tmp_path / "corpus.db"))
    build_index(
        backend,
        [
            # Invented facts the model can't know from pretraining — forces
            # it to actually rely on search_corpus rather than answering from
            # parametric memory (which would report zero tool calls and make
            # this test unable to verify masking). Multiple Q/A pairs reduce
            # the chance a single-question sample happens to skip tool use.
            CorpusDocument(
                doc_id="d1",
                title="d1",
                text="The Zibbendorf Tower is a 214-meter lattice structure located in the fictional city of Quorvane.",
            ),
            CorpusDocument(
                doc_id="d2",
                title="d2",
                text="The Blurnak river flows through the invented region of Sepwick and empties into Lake Fendrow.",
            ),
            CorpusDocument(
                doc_id="d3",
                title="d3",
                text="Ambassador Torvel Quinnick negotiated the fictional Treaty of Marrowfen in the year 1847.",
            ),
        ],
    )
    report = run_masking_check(
        model_path=os.environ.get("RAG_TEST_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        backend=backend,
        questions=[
            "Where is the Zibbendorf Tower located, and how tall is it?",
            "What lake does the Blurnak river empty into?",
            "Who negotiated the Treaty of Marrowfen, and in what year?",
        ],
        max_steps=4,
    )
    assert report["spot_check_passed"]
