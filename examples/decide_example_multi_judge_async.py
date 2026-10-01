"""
Example 2: Multi-Judge Consensus Scoring (Asynchronous)
Uses: Qwen/Qwen2.5-1.5B-Instruct
"""

import asyncio

from agenttune.decide import GraphRunner


async def main():
    print("=" * 70)
    print("DECIDE Example 2: Multi-Judge Consensus Scoring (Async)")
    print("=" * 70)

    runner = GraphRunner.from_template(
        "generic/multi_judge_score", config_path="./config/config.yaml"
    )

    test_texts = [
        "The research clearly shows evidence with multiple datasets.",
        "This paper lacks proper citations and makes unsupported claims.",
        "The methodology is sound but conclusions overstep the data.",
    ]

    # Run all in parallel
    print("\nRunning all evaluations in parallel...")
    tasks = [runner.run(text) for text in test_texts]
    states = await asyncio.gather(*tasks)

    for i, (text, state) in enumerate(zip(test_texts, states, strict=False), 1):
        print(f"\n--- Sample {i} ---")
        print(f"Text: {text[:60]}...")

        if state.error:
            print(f"❌ ERROR: {state.error}")
            continue

        print(f"Consensus Verdict: {state.verdict}")
        print(f"Consensus Score: {state.confidence:.1f}/10")

        # Extract individual judge scores
        if "judge_1" in state.stage_outputs:
            j1_score = state.stage_outputs["judge_1"].get("output", {}).get("score", "?")
            print(f"  Judge 1 score: {j1_score}")

        if "judge_2" in state.stage_outputs:
            j2_score = state.stage_outputs["judge_2"].get("output", {}).get("score", "?")
            print(f"  Judge 2 score: {j2_score}")

        if "judge_3" in state.stage_outputs:
            j3_score = state.stage_outputs["judge_3"].get("output", {}).get("score", "?")
            print(f"  Judge 3 score: {j3_score}")

        print("✅ PASS")

    print("\n" + "=" * 70)
    print("Multi-Judge Async Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
