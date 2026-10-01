#!/usr/bin/env python3
"""
Basic usage example for AgentTune Decide.

This example demonstrates:
1. Loading a configuration
2. Creating a GraphRunner from a template
3. Running a simple pipeline
4. Inspecting results
5. Reviewing audit logs
"""

import asyncio
import json
from pathlib import Path

import yaml

from agenttune.decide.audit import AuditReader
from agenttune.decide.graph_runner import GraphRunner


async def main():
    """Run the basic usage example."""

    # Configuration paths
    config_path = "config.yaml"
    audit_path = "audit.jsonl"
    output_file = "decision_result.json"

    print("=" * 70)
    print("AgentTune Decide - Basic Usage Example")
    print("=" * 70)

    # Step 1: Check configuration
    print("\n[Step 1] Checking configuration...")
    if not Path(config_path).exists():
        print(f"❌ Config file not found at {config_path}")
        print("   Please create config.yaml from config.example.yaml and set API keys")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"✅ Config loaded from {config_path}")
    print(f"   Default model: {config.get('default_model')}")
    print(f"   Audit path: {config.get('audit', {}).get('path')}")

    # Step 2: Create GraphRunner from template
    print("\n[Step 2] Creating GraphRunner from template...")
    template_id = "generic/text_classify"

    try:
        runner = GraphRunner.from_template(template_id, config_path)
        print(f"✅ GraphRunner created for template: {template_id}")
        print(f"   Template name: {runner.config.get('name')}")
        print(f"   Version: {runner.config.get('version')}")
        print(f"   Stages: {len(runner.config.get('stages', []))}")

        # Show stage pipeline
        print("   Pipeline stages:")
        for i, stage in enumerate(runner.config.get("stages", []), 1):
            print(f"     {i}. {stage['id']} ({stage['type']})")

    except Exception as e:
        print(f"❌ Error creating GraphRunner: {e}")
        return

    # Step 3: Prepare sample input
    print("\n[Step 3] Preparing sample input...")
    sample_input = """
    Customer Support Ticket:
    Subject: Product Quality Issue
    Description: The product arrived damaged and is not functioning as advertised.
    Sentiment: Negative
    Priority: High

    The customer is very upset and demanding a full refund or replacement.
    They have been a loyal customer for 2 years.
    """.strip()

    print("Sample input prepared:")
    print(f"  Length: {len(sample_input)} characters")
    print(f"  Preview: {sample_input[:100]}...")

    # Step 4: Run the pipeline
    print("\n[Step 4] Running pipeline...")
    print("⏳ This may take a minute or more (depending on model latency)...\n")

    try:
        final_state = await runner.run(sample_input)

        print("✅ Pipeline execution completed")
        print(f"   Pipeline ID: {final_state.pipeline_id}")
        print(f"   Steps executed: {final_state.step_count}")
        print(f"   Elapsed time: {final_state.elapsed_seconds:.2f}s")
        print(f"   Complete: {final_state.is_complete}")

        if final_state.error:
            print(f"   ⚠️  Error: {final_state.error}")

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 5: Inspect stage outputs
    print("\n[Step 5] Inspecting stage outputs...")
    print(f"Total stages executed: {len(final_state.stage_outputs)}")

    for stage_id, output in final_state.stage_outputs.items():
        print(f"\n  Stage: {stage_id}")
        if isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, str) and len(value) > 80:
                    print(f"    {key}: {value[:80]}...")
                else:
                    print(f"    {key}: {value}")
        else:
            print(f"    Output: {output}")

    # Step 6: Review audit trail
    print("\n[Step 6] Reviewing audit trail...")
    if Path(audit_path).exists():
        try:
            audit_reader = AuditReader(audit_path)
            all_entries = audit_reader.read_all()

            # Filter for this pipeline
            pipeline_entries = [
                e for e in all_entries if e.get("pipeline_id") == final_state.pipeline_id
            ]

            print(f"✅ Audit log contains {len(pipeline_entries)} entries for this pipeline")

            # Show stage execution timeline
            print("\n  Execution Timeline:")
            for i, entry in enumerate(pipeline_entries, 1):
                stage_id = entry.get("stage_id", "unknown")
                timestamp = entry.get("timestamp", "unknown")
                latency = entry.get("latency_ms", "?")
                print(f"    {i}. {stage_id:20s} @ {timestamp} ({latency}ms)")

        except Exception as e:
            print(f"⚠️  Error reading audit: {e}")
    else:
        print(f"ℹ️  Audit file not found at {audit_path}")

    # Step 7: Save results
    print("\n[Step 7] Saving results...")
    result_dict = {
        "pipeline_id": final_state.pipeline_id,
        "template_id": final_state.template_id,
        "template_version": final_state.template_version,
        "verdict": final_state.verdict,
        "verdict_label": final_state.verdict_label,
        "step_count": final_state.step_count,
        "elapsed_seconds": round(final_state.elapsed_seconds, 2),
        "is_complete": final_state.is_complete,
        "error": final_state.error,
        "stage_outputs": final_state.stage_outputs,
    }

    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    print(f"✅ Results saved to {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review audit.jsonl for detailed execution logs")
    print(f"  2. Check {output_file} for the final verdict")
    print("  3. Run another pipeline with different input")
    print("  4. Try a different template from the 22 available")
    print()


if __name__ == "__main__":
    asyncio.run(main())
