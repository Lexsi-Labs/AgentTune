"""_wrap_reward_fn: what DPO/BCO rollout reward functions receive."""

from agenttune.agentic.rollout_engines.rollout_factory import _wrap_reward_fn
from agenttune.agentic.trajectory.dataset import Trajectory

PROMPT = [
    {"role": "system", "content": "Answer in <answer></answer> tags."},
    {"role": "user", "content": "2+3?"},
]


def _traj(conversation):
    return Trajectory(
        task="2+3?", steps=[], final_response="5", metadata={"conversation": conversation}
    )


def test_reward_sees_generated_turns_not_the_prompt():
    seen = []

    def reward(completions, **kw):
        seen.extend(completions)
        return [0.0] * len(completions)

    multi = _traj(PROMPT + [{"role": "assistant", "content": "5"}])
    single = _traj(PROMPT)  # the single-turn path records only the prompt
    _wrap_reward_fn(reward)(["5", "5"], ["2+3?", "2+3?"], [multi, single])
    assert seen == [[{"role": "assistant", "content": "5"}], "5"]


def test_pre_11_reward_styles_still_work():
    trajs = [_traj(PROMPT + [{"role": "assistant", "content": "5"}])]

    def responses_and_prompts(responses, prompts):
        assert responses == ["5"] and prompts == ["2+3?"]
        return [1.0]

    def responses_only(responses):
        assert responses == ["5"]
        return [2.0]

    assert _wrap_reward_fn(responses_and_prompts)(["5"], ["2+3?"], trajs) == [1.0]
    assert _wrap_reward_fn(responses_only)(["5"], ["2+3?"], trajs) == [2.0]
    assert _wrap_reward_fn(lambda t: 0.5)(["5"], ["2+3?"], trajs) == [0.5]
    # Dict prompts (dataset rows) still reach old-style rewards as prompt text.
    row = {"prompt": "2+3?", "answer": "5"}
    assert _wrap_reward_fn(responses_and_prompts)(["5"], [row], trajs) == [1.0]
