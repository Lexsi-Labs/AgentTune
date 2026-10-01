"""
BFSI case study — Compliance QA over a regulation graph.
========================================================

Compliance knowledge is relational: a control implements a regulation, which cites a clause,
which supersedes an older one. A vector store recalls text that *looks* similar; a graph
recalls what is actually *connected*. This seeds a small regulation graph and shows an agent
recalling the controls linked to a regulation — the retrieval a compliance-QA agent needs.
GPU-free (no embedder, no model).

Run:  python examples/compliance_graph_memory.py
"""

from agenttune.agentic import GraphMemory, MemoryItem


def main():
    g = GraphMemory()

    # Nodes: two regulations, their controls, and an unrelated one.
    aml = g.write(MemoryItem(content="AML-4: ongoing transaction monitoring"))
    ctrl_velocity = g.write(MemoryItem(content="Control: velocity thresholds"))
    ctrl_sanctions = g.write(MemoryItem(content="Control: daily sanctions screening"))
    kyc = g.write(MemoryItem(content="KYC-1: customer due diligence"))
    ctrl_idv = g.write(MemoryItem(content="Control: identity verification"))
    g.write(MemoryItem(content="Control: office fire drill"))  # present, unlinked

    # Edges: which control implements which regulation.
    g.link(aml, ctrl_velocity, "implemented_by")
    g.link(aml, ctrl_sanctions, "implemented_by")
    g.link(kyc, ctrl_idv, "implemented_by")

    # An auditor asks: which controls cover AML-4?
    recalled = [i.content for i in g.read("AML-4: ongoing transaction monitoring", k=5)]
    print("[query]  controls covering 'AML-4':")
    for c in recalled:
        print(f"           - {c}")
    print(
        f"[check]  KYC control surfaced? {'Control: identity verification' in recalled}  "
        f"(should be False — different regulation)"
    )
    print(
        f"[check]  fire-drill surfaced? {'Control: office fire drill' in recalled}  "
        f"(should be False — unlinked)"
    )


if __name__ == "__main__":
    main()
