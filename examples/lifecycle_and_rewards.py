"""
Lifecycle + rewards — `collect`, `evaluate_agentic`, and reward shaping.
========================================================================

The other case studies use `infer` / `collect_rollout` / `evaluate`. This one closes the loop on
the remaining lifecycle surface and the reward pillar that GRPO training consumes:

  - `Project.collect(tasks)`       -> run a batch of tasks, keep every EventLog
  - `Project.evaluate_agentic()`   -> the agentic metrics report (tac/ter/arr/scsr/rad/lcf)
  - `answer_match(log, expected)`  -> a spine-native reward straight off an EventLog
  - `REWARD_REGISTRY` + `combine_rewards(...)` -> a weighted TRL-style reward_func the GRPO
    trainer would call on candidate completions

GPU-free; the reward functions are the exact ones a real GRPO run would score with.

Run:  python examples/lifecycle_and_rewards.py
"""

from agenttune.agentic import (
    REWARD_REGISTRY,
    DictToolHarness,
    Project,
    ReActStrategy,
    answer_match,
    combine_rewards,
)


def calc(x=0, y=0):
    return {"result": x + y}


def policy(state):
    if state.step == 0:
        return {"name": "calc", "arguments": {"x": 19, "y": 23}, "thought": "add the operands"}
    return {"name": "finish", "arguments": {"answer": "42"}, "thought": "report the sum"}


def main():
    proj = Project(
        strategy=ReActStrategy(policy, max_steps=4),
        harness=DictToolHarness({"calc": calc}, max_steps=4),
    )

    # 1) collect() — run a batch of tasks and keep every EventLog.
    logs = proj.collect(["what is 19 + 23?", "what is 19 + 23?"])
    print(f"[collect]  {len(logs)} episodes collected, tiers={[l.tier for l in logs]}")

    # 2) evaluate_agentic() — the agentic metrics report over a task set.
    report = proj.evaluate_agentic(["what is 19 + 23?"])
    print(f"[evaluate] n={report['n']}  metrics -> {report['metrics']}")

    # 3) answer_match — a reward computed directly off a spine EventLog.
    print(f"[reward]   answer_match(log, '42') -> {answer_match(logs[0], '42')}")

    # 4) Reward shaping for GRPO: pick funcs from the registry, weight them, and score
    #    candidate completions exactly as the trainer's rollout would.
    print(
        f"[registry] {len(REWARD_REGISTRY)} built-in reward funcs, e.g. "
        f"{sorted(REWARD_REGISTRY)[:3]}"
    )
    reward_fn = combine_rewards(
        [REWARD_REGISTRY["reward_correct_answer"], REWARD_REGISTRY["reward_concise_answer"]],
        weights=[0.7, 0.3],
    )
    candidates = [
        [{"role": "assistant", "content": "42"}],  # correct + concise
        [
            {"role": "assistant", "content": "the answer is 99 " + "and " * 90 + "done"}
        ],  # wrong + rambling
    ]
    scores = reward_fn(candidates, answer=["42", "42"])
    print(
        f"[shape]    weighted reward over 2 candidates -> {[round(s, 2) for s in scores]} "
        f"(correct+concise beats wrong+long)"
    )


if __name__ == "__main__":
    main()
