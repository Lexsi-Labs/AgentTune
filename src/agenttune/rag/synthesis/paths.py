"""
Stage 1 — Sample typed multi-hop paths over the chunk graph.

Adapts GRADE's path sampling (`graph/find_DAG_path_shortest.py` +
`sampling/sample_path.py`): find all simple paths of length 2–5 over the
networkx graph, dedup by (start, end) chunk so different internal routings
to the same endpoints aren't double-counted, then sample N per hop band.

Tags each path with a question type (2WikiMultiHopQA typology) and
specificity (RAGAS specific vs abstract), inferred from the edge types
along the path:
  - all-exact edges         → bridge (entity-anchored, specific)
  - comparison (two paths sharing a common node) → comparison
  - includes contextual/abstract edges → compositional / abstract
"""

from __future__ import annotations

import random

from .schema import ReasoningPath

MAX_HOP = 5
MIN_HOP = 2


def _infer_type(edges) -> tuple[str, str]:
    """Infer (question_type, specificity) from path edge types."""
    types = {e.type for e in edges}
    if types <= {"exact"}:
        return "bridge", "specific"
    if "abstract" in types:
        return "compositional", "abstract"
    return "compositional", "specific"


def sample_paths(
    G,
    chunks_by_id,
    *,
    per_hop=50,
    max_hop=MAX_HOP,
    min_hop=MIN_HOP,
    seed=42,
    max_paths_considered=20000,
    max_paths_per_endpoints=3,
) -> list[ReasoningPath]:
    """Sample reasoning paths from the graph (GRADE shortest-path + dedup).

    `G`: networkx graph with chunk_id nodes and typed/weighted edges.
    `chunks_by_id`: {chunk_id: Chunk} for entity/summary lookups.

    Bounded path collection: `all_simple_paths` explodes combinatorially on a
    dense graph (20k+ paths from 3 sources on a 53-node component), so we
    enumerate source-target pairs in random order and STOP once we've
    collected `max_paths_considered` total paths. Dedup keeps up to
    `max_paths_per_endpoints` distinct paths per (start,end) pair — keeping
    ONLY the shortest per pair (the old behavior) starves a small dense graph
    to a handful of paths even when hundreds of distinct routings exist, so we
    retain a few per pair to preserve path diversity. GRADE's shortest-path
    dedup is the degenerate case (max_paths_per_endpoints=1).

    Pairs are restricted to same-connected-component up front (computed once
    via `nx.connected_components`), not the full N^2 node pairs. Two nodes in
    different components can never have a path, so trying them is always
    wasted work — this is a correctness-neutral speedup on any graph, but it
    matters a lot on a graph built with `same_doc_only=True` (verified on a
    real 984-chunk/14-contract CUAD run): with the naive N^2 enumeration,
    ~88% of the ~968k pairs are cross-document and trivially path-less, so
    most of the shuffle+iterate budget was spent confirming the obvious
    before ever reaching a same-document pair.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes)
    if len(nodes) < 2:
        return []

    cap = max_paths_considered
    by_endpoints: dict = {}
    import networkx as nx

    pairs = []
    for component in nx.connected_components(G):
        comp_nodes = list(component)
        if len(comp_nodes) < 2:
            continue
        pairs.extend((s, t) for s in comp_nodes for t in comp_nodes if s != t)
    rng.shuffle(pairs)
    collected = 0
    for src, tgt in pairs:
        if collected >= cap:
            break
        # Cheap pre-filter: if no simple path of length <= max_hop exists,
        # all_simple_paths exhaustively searches the (often dense) component and
        # finds nothing — the dominant cost on dense same-doc graphs (measured:
        # a 46k-edge graph hung for minutes without this). Such pairs contribute
        # zero paths, so skipping them is output-identical.
        if _shortest_path_length(G, src, tgt, max_hop) is None:
            continue
        try:
            for path in _bounded_paths(G, src, tgt, max_hop, min_hop, max_paths_per_endpoints * 3):
                n_edges = len(path) - 1
                if min_hop <= n_edges <= max_hop:
                    key = (path[0], path[-1])
                    bucket = by_endpoints.setdefault(key, [])
                    # Stop enumerating routes for THIS (src,tgt) pair once we
                    # have enough — a dense subgraph (verified on real CUAD
                    # contracts: a 202-node connected component within one
                    # contract, avg degree ~23) can have tens of thousands of
                    # distinct simple paths of length <=5 between a single
                    # pair. Without this the inner loop burns the entire
                    # `max_paths_considered` budget on one or two pairs before
                    # any other (src,tgt) pair is ever tried, starving path
                    # diversity across the rest of the graph (observed: 984
                    # chunks, 5878 edges, 156s, only 6 final paths). We keep
                    # slightly more than max_paths_per_endpoints here (the
                    # final trim below still sorts shortest-first) so the
                    # shortest-first preference isn't biased by enumeration
                    # order within this small over-collection.
                    if len(bucket) >= max_paths_per_endpoints * 3:
                        break
                    # keep a few distinct routings per endpoint pair (dedup by
                    # the full node sequence, not just endpoints)
                    psig = tuple(path)
                    if psig in bucket:
                        continue
                    bucket.append(psig)
                    collected += 1
                    if collected >= cap:
                        break
        except Exception:
            continue
    # trim each endpoint bucket to max_paths_per_endpoints, shortest-first
    # (shorter paths = tighter reasoning chains, preferred)
    deduped = []
    for bucket in by_endpoints.values():
        bucket.sort(key=len)
        deduped.extend(bucket[:max_paths_per_endpoints])
    rng.shuffle(deduped)

    # Bin by hop count, sample per_hop each
    out: list[ReasoningPath] = []
    for hop in range(min_hop, max_hop + 1):
        band = [p for p in deduped if len(p) - 1 == hop]
        rng.shuffle(band)
        for path_nodes in band[:per_hop]:
            edges = []
            ents = set()
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                edata = G.get_edge_data(u, v) or {}
                from .schema import GraphEdge

                edges.append(
                    GraphEdge(
                        source=u,
                        target=v,
                        type=edata.get("type", "exact"),
                        weight=edata.get("weight", 1.0),
                        evidence=edata.get("evidence", ""),
                    )
                )
                ch = chunks_by_id.get(u)
                if ch:
                    ents.update(ch.entities)
            qtype, spec = _infer_type(edges)
            last = chunks_by_id.get(path_nodes[-1])
            if last:
                ents.update(last.entities)
            out.append(
                ReasoningPath(
                    path_id=f"path_{hop}hop_{len(out)}",
                    chunk_ids=list(path_nodes),
                    edges=edges,
                    hop_count=hop,
                    question_type=qtype,
                    specificity=spec,
                    entities=sorted(ents)[:20],
                )
            )
    return out


def networkx_all_simple_paths(G, src, tgt, cutoff):
    """Thin wrapper over networkx.all_simple_paths (imported lazily)."""
    import networkx as nx

    return nx.all_simple_paths(G, source=src, target=tgt, cutoff=cutoff)


def _shortest_path_length(G, src, tgt, cutoff):
    """Distance (edges) from src to tgt up to `cutoff`, or None if unreachable.

    `single_source_shortest_path_length` with a cutoff only explores up to
    `cutoff` layers — pairs whose target isn't reached are exactly the pairs
    whose all_simple_paths enumeration would exhaust the (often dense)
    component for nothing. Skipping them is output-identical and orders of
    magnitude faster on dense same-doc graphs (see sample_paths). Returns None
    (never raises) so the caller's pre-filter is a clean `is None` check."""
    import networkx as nx

    try:
        dists = nx.single_source_shortest_path_length(G, src, cutoff=cutoff)
        return dists.get(tgt)
    except Exception:
        return None


# Per-pair DFS budget (node-visits) — caps the cost of enumerating paths for a
# single (src, tgt) pair on dense graphs. nx.all_simple_paths exhaustively
# explores the whole component to yield even one path for pairs whose short
# routes are few (measured: <500 pairs processed in 25s on a 46k-edge graph);
# a fixed budget makes every pair O(budget) worst-case instead.
_BOUNDED_BUDGET = 5000


def _bounded_paths(G, src, tgt, max_hop, min_hop, max_paths, budget=_BOUNDED_BUDGET):
    """Yield simple paths src→tgt with length in [min_hop, max_hop], up to
    `max_paths`, bounded by `budget` node-visits per pair.

    DFS with an explicit visited-set, stopping once `max_paths` are found or
    `budget` neighbor-visits are consumed — guaranteed termination on dense
    graphs, unlike nx.all_simple_paths which explores the whole component.
    Output is a subset of what nx.all_simple_paths would yield (the first
    `max_paths` short paths), which is exactly what the sampler keeps anyway.
    """
    found = 0
    state = {"work": 0}

    def dfs(path, visited):
        nonlocal found
        last = path[-1]
        n = len(path) - 1
        if last == tgt:
            if n >= min_hop:
                found += 1
                yield tuple(path)
                if found >= max_paths:
                    return
            return  # never extend past the target
        if n >= max_hop:
            return
        for nb in G.neighbors(last):
            if nb in visited:
                continue
            state["work"] += 1
            if state["work"] > budget:
                return
            visited.add(nb)
            yield from dfs(path + (nb,), visited)
            visited.discard(nb)
            if found >= max_paths:
                return

    yield from dfs((src,), {src})
