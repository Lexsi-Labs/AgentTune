"""
Example 1: Text Classification (Asynchronous)
Uses: Qwen/Qwen2.5-1.5B-Instruct
"""

import asyncio

from agenttune.decide import GraphRunner


async def main():
    print("=" * 70)
    print("DECIDE Example 1: Text Classification (Async)")
    print("=" * 70)

    runner = GraphRunner.from_template("generic/text_classify", config_path="./config/config.yaml")

    test_cases = [
        ("I absolutely love this product! Best purchase ever!", "POSITIVE"),
        ("This is terrible, waste of money and time.", "NEGATIVE"),
        ("It's okay, nothing special.", "NEUTRAL"),
    ]

    # Run all in parallel
    print("\nRunning all tests in parallel...")
    tasks = [runner.run(text) for text, _ in test_cases]
    states = await asyncio.gather(*tasks)

    for i, ((text, expected), state) in enumerate(zip(test_cases, states, strict=False), 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {text}")
        print(f"Expected: {expected}")

        if state.error:
            print(f"❌ ERROR: {state.error}")
            continue

        print(f"Verdict: {state.verdict}")
        print(f"Confidence: {state.confidence}/10")

        # Verify
        if state.verdict == expected or state.verdict.upper() == expected.upper():
            print("✅ CORRECT")
        else:
            print(f"⚠️  Expected {expected}, got {state.verdict}")

    print("\n" + "=" * 70)
    print("Async Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
