"""
Example 2: Full KYC Triage Pipeline

Runs the bfsi/kyc_triage template with sample customer data,
inspects the audit trail, and shows how to extract DPO training pairs.

Usage:
    python examples/bfsi_kyc.py

Requirements:
    - Copy config.example.yaml to config.yaml and fill in API keys
    - OR run with mock mode: set AGENTTUNE_MOCK=1
"""

import asyncio
import json
import os
from pathlib import Path

MOCK_MODE = os.environ.get("AGENTTUNE_MOCK", "1") == "1"

# Sample customer inputs for KYC triage
SAMPLE_CUSTOMERS = [
    {
        "id": "cust_001",
        "text": (
            "Customer: Alice Smith. Date of Birth: January 15, 1990. "
            "Income: $75,000/year. Employment: Senior Engineer at TechCorp (5 years). "
            "SSN: 123-45-6789. Address: 456 Oak Street, Austin TX 78701. "
            "No prior bankruptcies or fraud flags."
        ),
        "expected": "APPROVE",
    },
    {
        "id": "cust_002",
        "text": (
            "Customer: Bob Jones. Date of Birth: July 22, 1982. "
            "Income: $28,000/year. Employment: Freelance consultant (variable). "
            "Address: 789 Pine Ave, Miami FL 33101. "
            "Previous address unknown. No tax returns on file."
        ),
        "expected": "REVIEW",
    },
    {
        "id": "cust_003",
        "text": "Customer document is incomplete. No income information, no DOB found.",
        "expected": "DENY",
    },
]

CONFIG_PATH = "./config.yaml"
TEMPLATE_ID = "bfsi/kyc_triage"
AUDIT_PATH = "./audit_kyc.jsonl"
DECISIONS_PATH = "./decisions_kyc.jsonl"


def print_separator(title: str = ""):
    width = 60
    print(f"\n{'─' * width}")
    if title:
        print(f"  {title}")
        print(f"{'─' * width}")


def mock_run_pipeline(customer: dict) -> dict:
    """Return a synthetic pipeline result without calling real LLMs."""
    verdict_map = {"APPROVE": 8, "REVIEW": 5, "DENY": 2}
    expected = customer["expected"]
    return {
        "verdict": expected,
        "verdict_label": expected.lower(),
        "confidence": verdict_map[expected],
        "reason": f"Mock decision for {customer['id']}",
        "step_count": 4,
        "elapsed_seconds": 1.2,
        "pipeline_id": f"mock-{customer['id']}",
    }


async def run_real_pipeline(customer: dict) -> dict:
    """Run the actual Decide pipeline for a customer."""
    from agenttune.decide.graph_runner import GraphRunner

    runner = GraphRunner.from_template(TEMPLATE_ID, CONFIG_PATH)
    state = await runner.run(customer["text"])
    return {
        "verdict": state.verdict,
        "verdict_label": state.verdict_label,
        "confidence": state.confidence,
        "reason": state.reason,
        "step_count": state.step_count,
        "elapsed_seconds": state.elapsed_seconds,
        "pipeline_id": state.pipeline_id,
    }


def run_kyc_pipeline(customer: dict) -> dict:
    if MOCK_MODE:
        return mock_run_pipeline(customer)
    return asyncio.run(run_real_pipeline(customer))


def show_audit_summary():
    """Read audit.jsonl and show stage-level summary."""
    audit_path = Path(AUDIT_PATH)
    if not audit_path.exists():
        print("  No audit log found (mock mode skips writing).")
        return

    stage_counts: dict = {}
    total_cost = 0.0
    total_latency_ms = 0

    with open(audit_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "stage_id" in entry:
                    stage_id = entry["stage_id"]
                    stage_counts[stage_id] = stage_counts.get(stage_id, 0) + 1
                    total_cost += entry.get("cost_usd", 0.0)
                    total_latency_ms += entry.get("latency_ms", 0)
            except json.JSONDecodeError:
                continue

    print(f"  Stages executed: {stage_counts}")
    print(f"  Total cost:      ${total_cost:.4f}")
    print(f"  Total latency:   {total_latency_ms}ms")


def extract_training_pairs():
    """Extract DPO pairs from the audit log for fine-tuning."""
    from agenttune.decide.training_bridge import DecideToTrainerBridge

    audit_path = Path(AUDIT_PATH)
    if not audit_path.exists():
        print("  No audit log found — run with real pipelines first.")
        return

    bridge = DecideToTrainerBridge(AUDIT_PATH)
    pairs = bridge.extract_dpo_pairs("decision_judge")
    labels = bridge.extract_bco_labels("approve")

    print(f"  DPO pairs extracted:  {len(pairs)}")
    print(f"  BCO labels extracted: {len(labels)}")

    if pairs:
        print("\n  Sample DPO pair:")
        pair = pairs[0]
        print(f"    Prompt:   {str(pair.get('input', ''))[:80]}...")
        print(f"    Rejected: {str(pair.get('rejected_output', ''))[:60]}")
        print(f"    Chosen:   {str(pair.get('chosen_output', ''))[:60]}")


def main():
    print("=" * 60)
    print("  AgentTune Decide — KYC Triage Example")
    print("=" * 60)
    print(f"  Mode: {'MOCK (no LLM calls)' if MOCK_MODE else 'REAL (requires config.yaml)'}")
    print(f"  Template: {TEMPLATE_ID}")

    results = []

    print_separator("Running KYC Triage for Sample Customers")
    for customer in SAMPLE_CUSTOMERS:
        print(f"\n  Customer: {customer['id']}")
        print(f"  Input:    {customer['text'][:80]}...")

        result = run_kyc_pipeline(customer)
        results.append({"customer_id": customer["id"], **result})

        verdict = result["verdict"]
        expected = customer["expected"]
        match = "✓" if verdict == expected else "✗"

        print(f"  Verdict:  {verdict} {match} (expected: {expected})")
        print(f"  Score:    {result['confidence']}/10")
        print(f"  Steps:    {result['step_count']}")
        print(f"  Time:     {result['elapsed_seconds']:.2f}s")

    print_separator("Summary")
    approved = sum(1 for r in results if r["verdict"] == "APPROVE")
    denied = sum(1 for r in results if r["verdict"] == "DENY")
    review = sum(1 for r in results if r["verdict"] == "REVIEW")
    print(f"  APPROVE: {approved} | REVIEW: {review} | DENY: {denied}")

    print_separator("Audit Trail Summary")
    show_audit_summary()

    print_separator("Training Data Extraction")
    extract_training_pairs()

    print_separator()
    print("  Done! Next steps:")
    print("  1. Review audit_kyc.jsonl for full execution traces")
    print("  2. Add human corrections to create DPO training pairs")
    print("  3. Run: python examples/training_pipeline.py to fine-tune")
    print()


if __name__ == "__main__":
    main()
