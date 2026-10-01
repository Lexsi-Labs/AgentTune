"""VectorMemory with a *real* embedding model — genuine semantic recall.

`VectorMemory(embed=...)` ranks stored items by cosine similarity of their embeddings to the
query. The GPU-free case study (`memory_recall.py`) and the service's `/memory_demo` pass a
**bag-of-words** `embed` (a fixed vocab count vector) so they can run offline — which means they
only match on *shared words*, not meaning. This example plugs in a real sentence-transformer
(`all-MiniLM-L6-v2`, 384-d) as the `embed` callable and shows it recalls by *meaning*:

  * the query shares **no content words** with the correct item, yet is retrieved first;
  * a lexical-overlap baseline (Jaccard over word sets) picks the wrong item on the same data.

That contrast is the whole point — with a real embedder VectorMemory does something the
bag-of-words stand-in cannot. Same `VectorMemory` class, same `write`/`read` calls; only the
`embed` function is real.

Requires: sentence-transformers and the model in the local HF cache
(`sentence-transformers/all-MiniLM-L6-v2`). Runs on CPU or GPU; no network.

    python examples/vector_memory_real.py
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from agenttune.agentic import MemoryItem, VectorMemory

# Support notes an agent has seen before. The key pair: the query below is a paraphrase of the
# "won't power on" note but shares no salient word with it, while a distractor note shares words
# with neither — so lexical overlap has nothing to grab onto.
NOTES = [
    "Customer's laptop will not switch on at all after the latest update.",
    "The invoice total does not match the sum of the line items on the order.",
    "Shipment was delivered to the wrong address in a different city.",
    "User cannot log in; the password reset email never arrives.",
]

# Paraphrase of note 0 with deliberately disjoint vocabulary ("dead / boot / pressing the button"
# vs "will not switch on / power"). A word-overlap matcher scores 0 against every note.
QUERY = "My machine is completely dead and nothing happens when I press the power button."
EXPECTED = NOTES[0]


def _jaccard(a: str, b: str) -> float:
    """Lexical-overlap baseline: Jaccard similarity over lowercased word sets."""
    wa = {w.strip(".,;:'\"").lower() for w in a.split()}
    wb = {w.strip(".,;:'\"").lower() for w in b.split()}
    return len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0


def main() -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    print(f"[embed]   sentence-transformers/all-MiniLM-L6-v2  ({dim}-d, real embeddings)")

    # The real embedder IS the VectorMemory embed callable.
    def embed(text) -> list[float]:
        return model.encode(str(text), normalize_embeddings=True).tolist()

    mem = VectorMemory(embed=embed)
    for note in NOTES:
        mem.write(MemoryItem(content=note))
    print(f"[write]   stored {len(NOTES)} support notes")

    # --- real semantic recall ---
    top = mem.read(QUERY, k=1)[0].content
    ranked = mem.read(QUERY, k=len(NOTES))
    print(f"[query]   {QUERY!r}")
    print(f"[recall]  top-1 (real embeddings) -> {top!r}")
    print("[rank]    full order (semantic):")
    for i, item in enumerate(ranked, 1):
        print(f"            {i}. {item.content!r}")

    # --- lexical baseline on the identical data ---
    lex = sorted(NOTES, key=lambda n: _jaccard(QUERY, n), reverse=True)
    lex_top = lex[0]
    lex_score = _jaccard(QUERY, EXPECTED)
    print(f"[lexical] Jaccard top-1 -> {lex_top!r}  (overlap with correct note = {lex_score:.2f})")

    semantic_ok = top == EXPECTED
    lexical_ok = lex_top == EXPECTED
    print(
        f"[verdict] real embedder retrieves the paraphrase: {semantic_ok}; "
        f"lexical baseline gets it: {lexical_ok} "
        f"-> semantic recall is doing real work: {semantic_ok and not lexical_ok}"
    )


if __name__ == "__main__":
    main()
