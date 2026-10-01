"""
Case study 4 — Memory design: semantic vs relational recall.
============================================================

The memory pillar ships three retrieval paradigms. This shows the two that differ most, and
why an agentic-RAG app would pick one over the other:

  VectorMemory (semantic)   — ranks by cosine similarity to the query embedding.
  GraphMemory  (relational) — traverses typed edges; recalls what is *connected*, not similar.

Both are model-free here: the embedder is a stdlib bag-of-words, and the graph is a few
linked entities. Swap in a real embedder / entity extractor unchanged.

Run:  python examples/memory_recall.py
"""

from agenttune.agentic import GraphMemory, MemoryItem, VectorMemory


def semantic_recall() -> None:
    # A 3-word vocabulary bag-of-words "embedder" — no model needed.
    vocab = ["cat", "dog", "car"]

    def embed(x):
        return [float(str(x).lower().count(w)) for w in vocab]

    m = VectorMemory(embed=embed)
    for content in ["cat cat", "dog dog", "car"]:
        m.write(MemoryItem(content=content))

    recalled = [i.content for i in m.read("cat", k=2)]
    print(f"[vector]  query 'cat' -> {recalled}   (semantic: nearest by cosine)")


def relational_recall() -> None:
    m = GraphMemory()
    alice = m.write(MemoryItem(content="Alice"))
    bob = m.write(MemoryItem(content="Bob"))
    m.write(MemoryItem(content="Dave"))  # present but unconnected
    m.link(alice, bob, "friend")

    recalled = [i.content for i in m.read("Alice", k=5)]
    print(f"[graph]   query 'Alice' -> {recalled}   (relational: Alice's neighbours, not 'Dave')")


def main() -> None:
    semantic_recall()
    relational_recall()


if __name__ == "__main__":
    main()
