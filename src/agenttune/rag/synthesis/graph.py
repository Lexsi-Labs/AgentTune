"""
Stage 0 — Corpus → typed knowledge graph.

Chunk documents (reuse our `retrieval/chunker.py` — langchain
RecursiveCharacterTextSplitter), extract entities + a short summary per chunk
(via the injected LLM), embed every chunk (injected embedder), then build a
typed-edge graph (GRADE exact/contextual + RAGAS abstract):

  exact      — two chunks share a named entity (string overlap on entity sets)
  contextual — an LLM resolves that two entities are the same concept across
               chunks where exact string match fails ("Aria" ≈ "Aria-2")
  abstract   — summary cosine similarity above a threshold (RAGAS MultiHopAbstract)

This is the one-time, amortized corpus-level step (Castform's "corpus profile"
principle): the graph is built once and sampled from repeatedly.

Edges are returned as `GraphEdge` records + a networkx graph (reused library,
the same one GRADE's path finder uses). The chunk dict is returned for the
path sampler and the index builder.
"""

from __future__ import annotations

from .io_utils import extract_json
from .schema import Chunk, GraphEdge

# A prompt that extracts entities (for exact edges), keyphrases (for diversity),
# and a one-line summary (for abstract edges) in one call per chunk.
#
# The "specific named entities" instruction alone is not enough on contract
# text: a first pass on real CUAD contracts (NLLP_SynthData.md validation
# run) showed the model over-extracting generic contract vocabulary —
# "agreement" (180 chunks), "party"/"parties" (95), "company" (78),
# "product" (83), even filing artifacts like "8-k" and "securities and
# exchange commission" — as if they were entities. A term appearing in most
# chunks of a document creates a near-complete subgraph on that document
# (every pair of chunks "shares an entity"), which drowns out genuine
# bridges and made path sampling blow up combinatorially on a single dense
# 239-chunk contract. The explicit exclusion list below is the fix, plus a
# deterministic stop-list filter in `annotate_chunks` as defense-in-depth
# (LLMs are not perfectly reliable instruction-followers at scale).
_EXTRACT_PROMPT = """You are extracting structured facts from a document passage for a knowledge graph.

Passage:
\"\"\"{text}\"\"\"

Extract:
1. `entities`: a list of PLAIN STRINGS (not objects) — proper nouns and specific named
   entities (people, orgs, products, places, dates, domain terms), each lowercase.
   Example: ["northwind robotics", "elena cho", "2015"]. Max 12.
   Do NOT include generic contract vocabulary even if capitalized in the source —
   "agreement", "party", "parties", "company", "product", "affiliate(s)",
   "recipient", "provider", "contractor" and similar role/structural words are
   NOT entities unless part of a more specific compound (e.g. "acme services
   agreement" is fine, bare "agreement" is not). Do NOT include filing
   metadata (form types like "8-k", regulator names like "securities and
   exchange commission") — that is not part of the contract's content.
2. `keyphrases`: 3-6 distinctive noun phrases as plain strings.
3. `summary`: ONE sentence (max ~30 words) capturing the passage's core topic.

Respond ONLY with JSON of this exact shape:
{{"entities": ["string", "string"], "keyphrases": ["string"], "summary": "string"}}"""

# Deterministic stop-list — applied after parsing, regardless of what the
# LLM returns. Catches the exact terms observed polluting the graph on real
# CUAD contracts, plus obvious generalizations of the same failure mode.
# Domain-agnostic callers (HotpotQA/FinDER) are unaffected: these are
# contract- and SEC-filing-specific terms that don't occur in that text.
_GENERIC_ENTITY_STOPLIST = {
    "agreement",
    "party",
    "parties",
    "company",
    "companies",
    "product",
    "products",
    "affiliate",
    "affiliates",
    "recipient",
    "provider",
    "contractor",
    "vendor",
    "customer",
    "customers",
    "service",
    "services",
    "8-k",
    "10-k",
    "10-q",
    "s-1",
    "exhibit",
    "securities and exchange commission",
    "sec",
    "inc",
    "llc",
    "ltd",
    "corp",
    "corporation",
}


def _filter_generic_entities(entities):
    """Drop stop-list terms and single-character/empty noise. Keeps
    multi-word compounds even if they contain a stop-list word (e.g.
    "acme services agreement" survives; bare "agreement" does not)."""
    return [e for e in entities if e and e not in _GENERIC_ENTITY_STOPLIST]


_CONTEXTUAL_PROMPT = """You are deciding whether entities from different passages refer to the same real-world thing.

Passage A entities: {a_ents}
Passage B entities: {b_ents}

Return JSON mapping each A-entity to the B-entity it refers to (if any).
Context-dependent equivalence counts (e.g. "Aria" and "Aria-2" = same product line;
"USA" and "United States" = always same). Only map genuine coreferences.
Respond ONLY with JSON: {{"mappings": [{{"a": "...", "b": "...", "type": "context"}}]}}
If none, return {{"mappings": []}}."""


def chunk_documents(docs, chunk_size=512, overlap=64, with_offsets=False) -> list[Chunk]:
    """Reuse the package's langchain chunker + CorpusDocument shape.

    `docs`: list of CorpusDocument (doc_id, title, text) or dicts with those keys.
    `with_offsets`: if True, each chunk's character range in its source
    document is recorded at `chunk.metadata["start"]`/`["end"]` — needed to
    align externally-labeled spans (e.g. CUAD's clause categories) to
    chunks. Off by default; existing HotpotQA/FinDER callers are unaffected.
    """
    from ..retrieval.chunker import chunk_text, chunk_text_with_offsets
    from ..retrieval.corpus_loader import CorpusDocument

    chunks: list[Chunk] = []
    for d in docs:
        if not isinstance(d, CorpusDocument):
            d = CorpusDocument(
                doc_id=d["doc_id"],
                title=d.get("title", ""),
                text=d["text"],
                metadata=d.get("metadata", {}),
            )
        if with_offsets:
            pieces = chunk_text_with_offsets(d.text, chunk_size=chunk_size, overlap=overlap)
            for idx, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        chunk_id=f"{d.doc_id}::{idx}",
                        text=piece["text"],
                        doc_id=d.doc_id,
                        title=d.title,
                        chunk_index=idx,
                        metadata={"start": piece["start"], "end": piece["end"]},
                    )
                )
        else:
            pieces = chunk_text(d.text, chunk_size=chunk_size, overlap=overlap)
            for idx, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        chunk_id=f"{d.doc_id}::{idx}",
                        text=piece,
                        doc_id=d.doc_id,
                        title=d.title,
                        chunk_index=idx,
                    )
                )
    return chunks


def annotate_chunks(
    chunks: list[Chunk], llm, *, stage="stage0", max_chunks=None, max_workers=4
) -> list[Chunk]:
    """Extract entities/keyphrases/summary per chunk via the injected LLM.

    Mutates each chunk's `entities`, `keyphrases`, and `metadata.summary`.
    Records every call on the returned chunks (callers aggregate into the dump).

    Runs the per-chunk extraction calls `max_workers`-way concurrent (LLM
    calls are I/O-bound). The old hardcoded 4 matched a limited endpoint;
    DeepSeek's official API allows far more — pass the run's budget (e.g. 64).
    """
    from .llm_client import parallel_chat

    n = len(chunks) if max_chunks is None else min(max_chunks, len(chunks))
    targets = chunks[:n] if max_chunks else chunks
    messages_list = [
        [{"role": "user", "content": _EXTRACT_PROMPT.format(text=c.text)}] for c in targets
    ]
    results = parallel_chat(
        llm,
        messages_list,
        max_workers=max_workers,
        stage=stage,
        purpose="extract_entities",
        temperature=0.0,
        max_tokens=12288,
    )
    for c, (text, rec) in zip(targets, results, strict=False):
        c.metadata["llm_calls"] = c.metadata.get("llm_calls", []) + [
            rec.__dict__ if hasattr(rec, "__dict__") else rec
        ]
        try:
            data = extract_json(text)
            # normalize entities: model may return list[str] or list[{"value":...}]
            raw_ents = data.get("entities", [])
            ents = []
            for e in raw_ents:
                if isinstance(e, str):
                    ents.append(e.lower().strip())
                elif isinstance(e, dict):
                    v = e.get("value") or e.get("name") or e.get("entity") or ""
                    if v:
                        ents.append(str(v).lower().strip())
            c.entities = _filter_generic_entities(ents)[:12]
            c.keyphrases = [
                str(k).lower().strip() for k in data.get("keyphrases", []) if isinstance(k, str)
            ][:6]
            c.metadata["summary"] = data.get("summary", "")
        except Exception:
            c.metadata["summary"] = ""
    return chunks


def embed_chunks(chunks: list[Chunk], embedder) -> list[Chunk]:
    """Embed each chunk's text (document side, no query prefix). Mutates + returns."""
    if not chunks:
        return chunks
    vecs = embedder.embed_documents([c.text for c in chunks])
    for c, v in zip(chunks, vecs, strict=False):
        c.embedding = v
    return chunks


def _drop_high_doc_frequency_entities(
    chunks: list[Chunk],
    max_doc_frequency: float,
    same_doc_only: bool,
    min_doc_frequency_count: int = 5,
) -> dict[str, set]:
    """Per-chunk entity sets with near-ubiquitous-within-their-document
    entities removed, for edge-building purposes only (does not mutate
    `chunk.entities`, which callers may still want intact for display).

    Verified need (NLLP_SynthData.md validation run): on real CUAD contracts,
    the entities that dominate a document are usually the contract's OWN
    party names — "aimmune" in 99/239 chunks of the Aimmune contract, entirely
    expected, since a party is referenced throughout its own contract. A
    global stop-list can't catch this (the same word is fine in a document
    where it's rare). But an entity present in most of one document's chunks
    provides ~zero discriminating signal for *which* chunks are meaningfully
    connected — every pair "shares" it, producing a near-complete subgraph
    that drowns out genuine bridges (rare, specific shared terms) and made
    path sampling combinatorially blow up on a 239-chunk contract in
    practice. This is TF-IDF's idea applied per-document instead of
    per-corpus: down-weight (here, drop) terms too common in their local
    context to discriminate.

    Only meaningful when `same_doc_only=True` (frequency is computed within
    each document's own chunk set) — a no-op passthrough otherwise, since
    HotpotQA/FinDER's cross-document design has a different frequency
    profile this wasn't validated against.

    `min_doc_frequency_count`: an entity is only eligible to be dropped if
    it appears in at least this many chunks of its document, REGARDLESS of
    fraction. Needed because fraction alone breaks on small documents — a
    2-chunk document where both chunks share one entity is 100% "frequency"
    but that's just a 2-hop bridge doing its job, not ubiquity. The count
    floor exempts small documents from this filter entirely (they can't
    reach it) while still catching the real pathology (an entity in 90+
    chunks of a 200+ chunk contract).
    """
    if not same_doc_only:
        return {c.chunk_id: set(c.entities) for c in chunks}
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    result: dict[str, set] = {}
    for _doc_id, doc_chunks in by_doc.items():
        n = len(doc_chunks)
        freq: dict[str, int] = {}
        for c in doc_chunks:
            for e in set(c.entities):
                freq[e] = freq.get(e, 0) + 1
        too_common = {
            e
            for e, cnt in freq.items()
            if n > 0 and cnt / n > max_doc_frequency and cnt >= min_doc_frequency_count
        }
        for c in doc_chunks:
            result[c.chunk_id] = set(c.entities) - too_common
    return result


def build_graph(
    chunks: list[Chunk],
    llm,
    *,
    abstract_threshold=0.75,
    exact_min_overlap=1,
    contextual_threshold=0.70,
    max_contextual_probes=60,
    stage="stage0",
    same_doc_only=False,
    max_entity_doc_frequency=0.3,
    max_workers=4,
) -> tuple[list[GraphEdge], object]:
    """Build typed edges over chunks.

    Returns (edges, networkx_graph). Uses exact entity overlap + LLM contextual
    resolution + cosine abstract similarity. networkx is the same library
    GRADE's path finder uses.

    `abstract_threshold` default 0.75 (strict): summary cosine similarity must be
    high for an abstract edge — a loose threshold (e.g. 0.55) links nearly every
    chunk to every other, making the graph near-complete and path sampling
    meaningless. Multi-hop questions need *genuine* semantic bridges, not noise.

    `same_doc_only=True` restricts every edge type (exact/contextual/abstract)
    to chunk pairs sharing a `doc_id`. Verified need (NLLP_SynthData.md §A0):
    on corpora of independent documents (e.g. CUAD contracts — 190/200 sampled
    are unrelated companies), a cross-document entity/embedding match is very
    likely coincidental, not a genuine reasoning chain. This guard makes that
    impossible at the source rather than filtering it after the fact, and as a
    side effect turns the O(n^2) pairwise cost into a sum of small per-document
    O(k^2) costs. Off by default (HotpotQA/FinDER rely on cross-document edges
    by design).

    `max_entity_doc_frequency` (default 0.3, only active when same_doc_only=True):
    drops an entity from edge consideration within a document if it appears
    in more than this fraction of that document's own chunks — see
    `_drop_high_doc_frequency_entities`. Without this, a contract's own party
    name (mentioned in most of its chunks) creates a near-complete subgraph.
    """
    import networkx as nx
    import numpy as np

    G = nx.Graph()
    for c in chunks:
        G.add_node(
            c.chunk_id,
            text=c.text,
            entities=c.entities,
            summary=c.metadata.get("summary", ""),
            doc_id=c.doc_id,
        )

    edge_entities = _drop_high_doc_frequency_entities(
        chunks, max_entity_doc_frequency, same_doc_only
    )

    edges: list[GraphEdge] = []
    seen: set = set()

    # --- exact edges: shared entities ---
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            a, b = chunks[i], chunks[j]
            if same_doc_only and a.doc_id != b.doc_id:
                continue
            shared = edge_entities[a.chunk_id] & edge_entities[b.chunk_id]
            if len(shared) >= exact_min_overlap:
                key = (a.chunk_id, b.chunk_id, "exact")
                if key in seen:
                    continue
                seen.add(key)
                e = GraphEdge(
                    a.chunk_id,
                    b.chunk_id,
                    "exact",
                    weight=len(shared),
                    evidence=";".join(sorted(shared)),
                )
                edges.append(e)
                G.add_edge(
                    a.chunk_id, b.chunk_id, type="exact", weight=len(shared), evidence=e.evidence
                )

    # --- contextual edges: LLM resolves cross-chunk coreference ---
    # Cost guard: only probe pairs that are (a) not already exact-linked AND
    # (b) embedding-similar above a STRICT threshold (0.70) — contextual
    # equivalence implies near-identical meaning, so a loose threshold (0.45)
    # probes ~900 pairs for 62 chunks = ~80 min on a reasoning model. 0.70 +
    # a hard cap (`max_contextual_probes`) keeps this bounded and fast. Probes
    # run 4-way concurrent to match the endpoint's 4-call limit.
    embs = np.array([c.embedding for c in chunks if c.embedding is not None])
    emb_idx = [k for k, c in enumerate(chunks) if c.embedding is not None]
    if len(embs) >= 2:
        sim_mat = embs @ embs.T  # normalized embeddings
    else:
        sim_mat = None

    # collect candidate pairs (strict threshold + cap), then probe in parallel
    candidates = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            a, b = chunks[i], chunks[j]
            if same_doc_only and a.doc_id != b.doc_id:
                continue
            if G.has_edge(a.chunk_id, b.chunk_id):
                continue
            if not a.entities or not b.entities:
                continue
            if sim_mat is not None and i in emb_idx and j in emb_idx:
                si, sj = emb_idx.index(i), emb_idx.index(j)
                if sim_mat[si, sj] < contextual_threshold:
                    continue
            candidates.append((i, j))
    candidates = candidates[:max_contextual_probes]

    if candidates:
        from .llm_client import parallel_chat

        msgs = [
            [
                {
                    "role": "user",
                    "content": _CONTEXTUAL_PROMPT.format(
                        a_ents=chunks[i].entities, b_ents=chunks[j].entities
                    ),
                }
            ]
            for i, j in candidates
        ]
        results = parallel_chat(
            llm,
            msgs,
            max_workers=max_workers,
            stage=stage,
            purpose="contextual_equiv",
            temperature=0.0,
            max_tokens=12288,
        )
        for (i, j), (text, _rec) in zip(candidates, results, strict=False):
            try:
                maps = extract_json(text).get("mappings", [])
            except Exception:
                maps = []
            if maps:
                a, b = chunks[i], chunks[j]
                ev = "; ".join(f"{m['a']}≈{m['b']}" for m in maps)
                e = GraphEdge(a.chunk_id, b.chunk_id, "contextual", weight=len(maps), evidence=ev)
                edges.append(e)
                G.add_edge(a.chunk_id, b.chunk_id, type="contextual", weight=len(maps), evidence=ev)

    # --- abstract edges: summary cosine similarity (RAGAS MultiHopAbstract) ---
    summaries = [c.metadata.get("summary", "") or c.text[:60] for c in chunks]
    if any(summaries):
        vecs = np.array([c.embedding for c in chunks if c.embedding is not None])
        idx_map = [k for k, c in enumerate(chunks) if c.embedding is not None]
        if len(vecs) >= 2:
            sim = vecs @ vecs.T  # embeddings are normalized in the embedder
            for ii in range(len(vecs)):
                for jj in range(ii + 1, len(vecs)):
                    if sim[ii, jj] >= abstract_threshold:
                        a, b = chunks[idx_map[ii]], chunks[idx_map[jj]]
                        if same_doc_only and a.doc_id != b.doc_id:
                            continue
                        if G.has_edge(a.chunk_id, b.chunk_id):
                            continue
                        e = GraphEdge(
                            a.chunk_id,
                            b.chunk_id,
                            "abstract",
                            weight=float(sim[ii, jj]),
                            evidence=f"summary_sim={sim[ii,jj]:.3f}",
                        )
                        edges.append(e)
                        G.add_edge(
                            a.chunk_id,
                            b.chunk_id,
                            type="abstract",
                            weight=float(sim[ii, jj]),
                            evidence=e.evidence,
                        )

    if same_doc_only:
        by_id = {c.chunk_id: c for c in chunks}
        violations = [e for e in edges if by_id[e.source].doc_id != by_id[e.target].doc_id]
        assert not violations, (
            f"same_doc_only=True but {len(violations)} cross-document edge(s) "
            f"were built — this is a bug, not a warning: {violations[:3]}"
        )
    return edges, G
