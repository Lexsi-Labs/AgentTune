"""
Legal case study — Precedent retrieval over a citation graph.
=============================================================

Case law is a graph: a case cites an earlier case, and a later ruling can overrule it. A vector
store recalls opinions that *read* alike; a citation graph recalls what a case actually *relies
on* — and never surfaces an unrelated matter just because the language is similar. This seeds a
small precedent graph and shows an agent recalling the authorities a case leans on. GPU-free
(no embedder, no model).

Illustrative only — not legal advice.

Run:  python examples/precedent_graph_memory.py
"""

from agenttune.agentic import GraphMemory, MemoryItem


def main():
    g = GraphMemory()

    # Nodes: the case at hand, the authorities it relies on, and an unrelated matter.
    subject = g.write(MemoryItem(content="Acme v. Byte (2024): trade-secret misappropriation"))
    pepsi = g.write(MemoryItem(content="PepsiCo v. Redmond (1995): inevitable disclosure"))
    uniform = g.write(MemoryItem(content="Uniform Trade Secrets Act §1: definition of a secret"))
    g.write(MemoryItem(content="Doe v. City (2019): municipal zoning dispute"))

    # Edges: what the subject case cites as authority.
    g.link(subject, pepsi, "cites")
    g.link(subject, uniform, "cites")

    # A litigator asks: what authority does Acme v. Byte rely on?
    recalled = [
        i.content for i in g.read("Acme v. Byte (2024): trade-secret misappropriation", k=5)
    ]
    print("[query]  authorities relied on by 'Acme v. Byte':")
    for c in recalled:
        print(f"           - {c}")
    print(
        f"[check]  zoning matter surfaced? {'Doe v. City (2019): municipal zoning dispute' in recalled}  "
        f"(should be False — uncited, unrelated)"
    )


if __name__ == "__main__":
    main()
