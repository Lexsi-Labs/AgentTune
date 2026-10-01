from agenttune.rag.rewards.phase1_rewards import (
    format_reward,
    get_last_component_scores,
    get_logged_training_reward,
    get_training_reward,
    rag_correctness_reward,
    search_usage_reward,
)
from agenttune.rag.rewards.qa_metrics import exact_match_score, extract_answer_tag, f1_score


def test_exact_match_score_identical():
    assert exact_match_score("Paris", "Paris") == 1.0


def test_exact_match_score_case_and_punct_normalized():
    assert exact_match_score("paris.", "Paris") == 1.0


def test_exact_match_score_different():
    assert exact_match_score("London", "Paris") == 0.0


def test_f1_score_partial_overlap():
    score = f1_score("The capital is Paris", "Paris")
    assert 0.0 < score <= 1.0


def test_f1_score_no_overlap():
    assert f1_score("London", "Paris") == 0.0


def test_extract_answer_tag_present():
    assert extract_answer_tag("some reasoning <answer>Paris</answer> done") == "Paris"


def test_extract_answer_tag_absent_falls_back_to_full_text():
    assert extract_answer_tag("Paris") == "Paris"


def test_extract_answer_tag_handles_message_list_completion():
    # Agentic-mode completions are list[dict]; existing convention is str(completion)
    # then regex — extract_answer_tag must not crash on non-string input.
    completion = [{"role": "assistant", "content": "<answer>Paris</answer>"}]
    assert extract_answer_tag(completion) == "Paris"


def test_rag_correctness_reward_scores_against_gold():
    completions = ["<answer>Paris</answer>", "<answer>London</answer>"]
    gold = ["Paris", "Paris"]
    scores = rag_correctness_reward(prompts=None, completions=completions, gold_answer=gold)
    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_rag_correctness_reward_no_gold_returns_zero():
    completions = ["<answer>Paris</answer>"]
    scores = rag_correctness_reward(prompts=None, completions=completions, gold_answer=None)
    assert scores == [0.0]


def test_search_usage_reward_tiers():
    # NOTE: search_grounding_reward's own docstring claims "3+ calls -> 0.4",
    # but its implementation does tiers.get(min(c, 2), 0.4) — since 2 is a
    # valid key, the min() cap means 0.4 is never actually reachable. This is
    # a pre-existing quirk in agenttune's own builtin reward, not something
    # we can fix here (no edits to agentic/ files) — asserting real behavior.
    completions = ["a", "b", "c", "d"]
    scores = search_usage_reward(
        prompts=None, completions=completions, tool_call_counts=[0, 1, 2, 3]
    )
    assert scores == [0.0, 0.2, 0.3, 0.3]


def test_format_reward_detects_answer_tags():
    scores = format_reward(prompts=None, completions=["<answer>x</answer>", "no tags here"])
    assert scores == [0.1, 0.0]


def test_get_training_reward_combines_components():
    reward_fn = get_training_reward()
    completions = ["<answer>Paris</answer>"]
    score = reward_fn(
        completions=completions,
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[2],
    )
    assert len(score) == 1
    assert score[0] > 0.0


def test_get_training_reward_custom_weights():
    # Zero out everything except format (incl. the new termination component)
    # -> score == format reward's own value (0.1, normalized weight=1.0)
    reward_fn = get_training_reward(
        weights={"format": 1.0, "termination": 0.0, "search_usage": 0.0, "correctness": 0.0}
    )
    completions = ["<answer>x</answer>"]
    score = reward_fn(
        completions=completions, prompts=[None], gold_answer=["y"], tool_call_counts=[0]
    )
    assert score[0] == 0.1


def test_termination_reward_rewards_answer_tag():
    from agenttune.rag.rewards.phase1_rewards import termination_reward

    # has answer tag → +1.0
    scores = termination_reward(
        prompts=None, completions=["<answer>Paris</answer>"], tool_call_counts=[3]
    )
    assert scores == [1.0]


def test_termination_reward_penalizes_oversearch_without_answer():
    from agenttune.rag.rewards.phase1_rewards import termination_reward

    # no answer, 5 searches → negative penalty
    scores = termination_reward(
        prompts=None, completions=["just a tool call"], tool_call_counts=[5]
    )
    assert scores[0] < 0.0


def test_termination_reward_neutral_early_exploration():
    from agenttune.rag.rewards.phase1_rewards import termination_reward

    # no answer, 1-2 searches → neutral (0.0)
    scores = termination_reward(prompts=None, completions=["searching..."], tool_call_counts=[1])
    assert scores == [0.0]


def test_training_reward_answering_beats_looping():
    # The key Sprint-2 property: a rollout that answers (even wrong) must score
    # higher than one that loops on searches without answering. This is what
    # gives GRPO the variance to learn to stop searching and answer.
    reward_fn = get_training_reward()
    # Looping: 6 searches, no answer tag
    looping = reward_fn(
        completions=["<function=search_corpus><parameter=query>foo</parameter></function>"],
        prompts=[None],
        gold_answer=["bar"],
        tool_call_counts=[6],
    )[0]
    # Answering: 2 searches, answer tag (wrong answer)
    answering = reward_fn(
        completions=["<answer>wrong</answer>"],
        prompts=[None],
        gold_answer=["bar"],
        tool_call_counts=[2],
    )[0]
    assert answering > looping, f"answering={answering} should beat looping={looping}"


def test_logged_reward_captures_components():
    # The logged reward fn should stash per-component scores accessible via
    # get_last_component_scores(), for the trace + tensorboard.
    reward_fn = get_logged_training_reward()
    score = reward_fn(
        completions=["<answer>Paris</answer>", "<answer>London</answer>"],
        prompts=[None, None],
        gold_answer=["Paris", "Paris"],
        tool_call_counts=[2, 5],
    )
    assert len(score) == 2
    assert score[0] > score[1]  # correct+fewer-searches beats wrong+more-searches
    comp = get_last_component_scores()
    assert "format" in comp and "correctness" in comp
    assert "termination" in comp and "search_usage" in comp
    assert len(comp["correctness"]) == 2
    # rollout 0 correct, rollout 1 wrong
    assert comp["correctness"][0] == 1.0
    assert comp["correctness"][1] == 0.0
    # rollout 0 has answer tag -> termination 1.0; rollout 1 also has tag -> 1.0
    assert comp["termination"][0] == 1.0
