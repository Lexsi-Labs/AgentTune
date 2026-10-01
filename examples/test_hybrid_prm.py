import os

from dotenv import load_dotenv

load_dotenv()

# We need to set PYTHONPATH=src before importing
import sys

sys.path.insert(0, os.path.abspath("src"))

from agenttune.agentic.rewards.builtin_rewards.hybrid_prm import hybrid_prm_reward


def main():
    print("Testing Hybrid PRM Reward Function...")
    prompts = [
        "System: You are an agent.\nUser: what is the weather in SF?",
        "System: You are an agent.\nUser: tell me a joke",
        "System: You are an agent.\nUser: what is the weather in NY?",
    ]

    # 1. A perfectly valid tool call
    comp1 = 'Let me check that for you.\n<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}}</tool_call>'

    # 2. A malformed tool call (Missing closing brace)
    comp2 = 'Sure.\n<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}</tool_call>'

    # 3. An empty/useless tool call (API redundancy penalty usually targets these if repeated)
    comp3 = "<tool_call>{}</tool_call>"

    completions = [comp1, comp2, comp3]

    print("\n--- Deterministic PRM Mode (Fast) ---")
    rewards = hybrid_prm_reward(prompts, completions, use_llm_judge=False)

    for i, r in enumerate(rewards):
        print(f"Completion {i+1} Reward: {r:.4f}")

    print("\nHybrid PRM test completed successfully!")


if __name__ == "__main__":
    main()
