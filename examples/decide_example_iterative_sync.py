"""
Example 3: Iterative Refinement (Synchronous)
Generate → Evaluate → [Loop if score < 7] → Output
Uses: Qwen/Qwen2.5-1.5B-Instruct
"""

from agenttune.decide import GraphRunner


def main():
    print("=" * 70)
    print("DECIDE Example 3: Iterative Refinement (Sync)")
    print("=" * 70)

    runner = GraphRunner.from_template(
        "generic/iterative_refinement", config_path="./config/config.yaml"
    )

    prompts = [
        "Write a haiku about spring",
        "Explain quantum entanglement in 2 sentences",
        "Create a business plan summary",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Task {i}: {prompt} ---")

        state = runner.run_sync(prompt)

        if state.error:
            print(f"❌ ERROR: {state.error}")
            continue

        iterations = len([s for s in state.step_history if s == "generate"])
        print(f"Iterations: {iterations}")
        print(f"Final Quality Score: {state.confidence}/10")

        # Show generated output
        if "generate" in state.stage_outputs:
            output = state.stage_outputs["generate"].get("output", {}).get("output", "")
            print(f"Generated:\n{output[:150]}...")

        print("✅ PASS")

    print("\n" + "=" * 70)
    print("Iterative Refinement Sync Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
