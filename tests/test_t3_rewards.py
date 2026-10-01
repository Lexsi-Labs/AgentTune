"""
Unit tests for T3 necessity + frugality rewards.

Hand-built episodes per rag_plan_s2.md §3-B1:
  - 0 searches when search was needed (under-search)
  - exact minimum searches (perfect timing)
  - 2× minimum searches (over-search)
  - search when NOT needed (necessity violation)
  - answer directly when not needed (necessity ideal)
"""

from agenttune.rag.rewards.t3_rewards import (
    frugality_reward,
    get_combined_reward,
    get_last_combined_component_scores,
    get_last_m2_component_scores,
    get_logged_combined_reward,
    get_logged_m2_reward,
    get_m2_reward,
    get_t3_reward,
    necessity_reward,
)

# ── Necessity (IKEA r_kb) ────────────────────────────────────────────────────


def test_necessity_needs_search_correct_few_searches_high_reward():
    # Needs retrieval, correct answer, 1 search → high reward (1 - 1/6)
    scores = necessity_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[1],
        gold_answer=["Paris"],
        requires_search=[True],
        max_searches=6,
    )
    assert scores == [1.0 - 1.0 / 6.0]


def test_necessity_needs_search_wrong_answer_zero():
    # Needs retrieval, wrong answer → 0 regardless of search count (r_ans gates r_kb)
    scores = necessity_reward(
        prompts=None,
        completions=["<answer>London</answer>"],
        tool_call_counts=[2],
        gold_answer=["Paris"],
        requires_search=[True],
    )
    assert scores == [0.0]


def test_necessity_needs_search_more_searches_lower_reward():
    # Same correct answer, more searches → lower reward (fewer-is-better)
    s1 = necessity_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[1],
        gold_answer=["Paris"],
        requires_search=[True],
        max_searches=6,
    )[0]
    s3 = necessity_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[3],
        gold_answer=["Paris"],
        requires_search=[True],
        max_searches=6,
    )[0]
    assert s1 > s3


def test_necessity_no_search_needed_answer_from_memory_ideal():
    # Doesn't need retrieval, correct, 0 searches → max reward (1.0)
    scores = necessity_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[0],
        gold_answer=["Paris"],
        requires_search=[False],
    )
    assert scores == [1.0]


def test_necessity_no_search_needed_but_searched_penalised():
    # Doesn't need retrieval, correct, but searched anyway → partial, decaying
    scores = necessity_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[3],
        gold_answer=["Paris"],
        requires_search=[False],
        max_searches=6,
    )
    assert 0.0 < scores[0] < 1.0
    assert scores[0] == 1.0 - 3.0 / 6.0


def test_necessity_default_requires_search_true_when_absent():
    # When requires_search is None, default to True (conservative — never punish
    # a correct search-then-answer).
    scores = necessity_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[1],
        gold_answer=["Paris"],
        requires_search=None,
    )
    assert scores[0] > 0.0  # correct + searched + (default) needs-search


def test_necessity_no_gold_returns_zero():
    scores = necessity_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[1],
        gold_answer=None,
    )
    assert scores == [0.0]


# ── Frugality (FrugalRAG) ─────────────────────────────────────────────────────


def test_frugality_exact_optimal_gets_hardness_bonus():
    # optimal=2, searched 2 → 1.0 + bonus
    scores = frugality_reward(
        prompts=None,
        completions=["x"],
        tool_call_counts=[2],
        requires_search=[True],
        optimal_search_count=[2],
    )
    assert scores == [1.2]


def test_frugality_under_search_partial():
    # optimal=2, searched 1 → 0.5 * (1/2) = 0.25
    scores = frugality_reward(
        prompts=None,
        completions=["x"],
        tool_call_counts=[1],
        requires_search=[True],
        optimal_search_count=[2],
    )
    assert scores == [0.25]


def test_frugality_over_search_log_penalty_capped_at_zero():
    # optimal=1, searched 6 (way over) → should be heavily penalised, >= 0
    scores = frugality_reward(
        prompts=None,
        completions=["x"],
        tool_call_counts=[6],
        requires_search=[True],
        optimal_search_count=[1],
        max_searches=6,
    )
    assert 0.0 <= scores[0] < 0.5  # over-search → low reward


def test_frugality_more_oversearch_lower_reward():
    # Monotonic: more over-search → lower (or equal) reward
    s2 = frugality_reward(
        prompts=None,
        completions=["x"],
        tool_call_counts=[2],
        requires_search=[True],
        optimal_search_count=[1],
        max_searches=6,
    )[0]
    s4 = frugality_reward(
        prompts=None,
        completions=["x"],
        tool_call_counts=[4],
        requires_search=[True],
        optimal_search_count=[1],
        max_searches=6,
    )[0]
    assert s2 >= s4


def test_frugality_not_needed_returns_zero():
    # requires_search=False → frugality vacuous (necessity handles it)
    scores = frugality_reward(
        prompts=None,
        completions=["x"],
        tool_call_counts=[0],
        requires_search=[False],
        optimal_search_count=[0],
    )
    assert scores == [0.0]


def test_frugality_heuristic_optimal_when_absent():
    # No optimal_search_count → heuristic: 2 for needs-search, 0 otherwise
    scores = frugality_reward(
        prompts=None,
        completions=["x"],
        tool_call_counts=[2],
        requires_search=[True],
        optimal_search_count=None,
    )
    # heuristic optimal=2, searched 2 → perfect timing + bonus
    assert scores == [1.2]


# ── Composed T3 stack ─────────────────────────────────────────────────────────


def test_get_t3_reward_combines_components():
    reward_fn = get_t3_reward()
    score = reward_fn(
        completions=["<answer>Paris</answer>"],
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[2],
        requires_search=[True],
        optimal_search_count=[2],
    )
    assert len(score) == 1
    # Correct + perfect-timing frugality + necessity + format → strongly positive
    assert score[0] > 0.0


def test_get_t3_reward_custom_weights():
    # Zero out everything except format → score == format reward (0.1, normalised to 1.0)
    reward_fn = get_t3_reward(
        weights={"format": 1.0, "correctness": 0.0, "necessity": 0.0, "frugality": 0.0}
    )
    score = reward_fn(
        completions=["<answer>x</answer>"],
        prompts=[None],
        gold_answer=["y"],
        tool_call_counts=[0],
    )
    assert score[0] == 0.1


def test_get_t3_reward_wrong_answer_low_score():
    # Wrong answer, no tags, over-searched → should be near-zero
    reward_fn = get_t3_reward()
    score = reward_fn(
        completions=["no tags here"],
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[6],
        requires_search=[True],
        optimal_search_count=[1],
    )
    assert score[0] < 0.2


# ── Combined stack (fixes the m1_t3_long collapse: EM=0.000, no answer tag) ──
# T3's default weights (termination=0.2) let the m1_t3_long run (150 steps,
# 40 questions) collapse into looping-without-answering — necessity+frugality
# award search-count-based credit independent of ever emitting <answer>. The
# combined stack restores phase-1's termination dominance (format+termination
# =0.4, the ratio that fixed the original loop failure — see
# sprint2_results/README.md §2/§5) while keeping necessity/frugality small.


def test_combined_looping_without_answer_scores_low():
    # No answer tag, 6 searches → termination penalty dominates, low score
    reward_fn = get_combined_reward()
    looping_score = reward_fn(
        completions=["still searching, no answer tag here"],
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[6],
        requires_search=[True],
        optimal_search_count=[2],
    )[0]
    answering_score = reward_fn(
        completions=["<answer>Paris</answer>"],
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[2],
        requires_search=[True],
        optimal_search_count=[2],
    )[0]
    assert answering_score > looping_score


def test_combined_termination_weight_matches_phase1_ratio():
    # format(0.1) + termination(0.3) = 0.4, same ratio as the phase-1 fix.
    # Isolate: zero out correctness/necessity/frugality, answer-tag present.
    # termination fires fully (has_answer=True -> 1.0); format may be partial
    # depending on the builtin's structure checks, so just assert termination's
    # larger share dominates a wrong-answer, tag-present completion.
    reward_fn = get_combined_reward(
        weights={
            "format": 0.1,
            "termination": 0.3,
            "correctness": 0.0,
            "necessity": 0.0,
            "frugality": 0.0,
        }
    )
    score = reward_fn(
        completions=["<answer>x</answer>"],
        prompts=[None],
        gold_answer=["y"],  # wrong answer, but termination only cares about the tag
        tool_call_counts=[1],
    )
    # norm_w: format=0.25, termination=0.75 of the active weight. termination
    # alone (has_answer=1.0) contributes >= 0.75.
    assert score[0] >= 0.75


def test_combined_over_search_without_answer_penalised_more_than_early_explore():
    reward_fn = get_combined_reward()
    early = reward_fn(
        completions=["no answer yet"],
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[1],
        requires_search=[True],
        optimal_search_count=[2],
    )[0]
    over = reward_fn(
        completions=["no answer yet"],
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[6],
        requires_search=[True],
        optimal_search_count=[2],
    )[0]
    assert early > over


def test_logged_combined_reward_captures_components():
    reward_fn = get_logged_combined_reward()
    reward_fn(
        completions=["<answer>Paris</answer>"],
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[2],
        requires_search=[True],
        optimal_search_count=[2],
    )
    comp = get_last_combined_component_scores()
    assert set(comp.keys()) == {"format", "termination", "correctness", "necessity", "frugality"}
    assert comp["termination"] == [1.0]
    assert comp["correctness"][0] > 0.0


# ── M2 (combined stack + decision-token bonus) ──────────────────────────────
# get_m2_reward builds on get_combined_reward (not raw T3) so M2 doesn't
# reintroduce the termination-dilution mode collapse combined already fixed.
# decision_reward is a small additive bonus (per its own docstring) — these
# tests check it nudges the score without dominating the outcome signal.


def test_m2_no_decision_token_scores_like_combined():
    # No <decision:...> token emitted -> decision component contributes 0,
    # so an m2 completion with no token scores lower than an otherwise
    # identical completion that does emit one.
    combined_fn = get_combined_reward()
    m2_fn = get_m2_reward()
    base_kwargs = {
        "prompts": [None],
        "gold_answer": ["Paris"],
        "tool_call_counts": [2],
        "requires_search": [True],
        "optimal_search_count": [2],
    }
    combined_score = combined_fn(completions=["<answer>Paris</answer>"], **base_kwargs)[0]
    m2_score_no_token = m2_fn(completions=["<answer>Paris</answer>"], **base_kwargs)[0]
    m2_score_with_token = m2_fn(
        completions=["<decision:compress:answer-now> <answer>Paris</answer>"], **base_kwargs
    )[0]
    assert m2_score_with_token > m2_score_no_token
    # combined and m2-without-token differ only by decision's (zero) contribution
    # diluting the other weights slightly — should stay close, not diverge.
    assert abs(combined_score - m2_score_no_token) < 0.1


def test_m2_decision_bonus_is_small_not_dominant():
    # A wrong answer with a decision token should not outscore a correct
    # answer with no token — the outcome signal (correctness/termination)
    # must still dominate the small decision bonus.
    m2_fn = get_m2_reward()
    base_kwargs = {
        "prompts": [None],
        "tool_call_counts": [2],
        "requires_search": [True],
        "optimal_search_count": [2],
    }
    wrong_with_token = m2_fn(
        completions=["<decision:compress:answer-now> <answer>WrongAnswer</answer>"],
        gold_answer=["Paris"],
        **base_kwargs,
    )[0]
    correct_no_token = m2_fn(
        completions=["<answer>Paris</answer>"],
        gold_answer=["Paris"],
        **base_kwargs,
    )[0]
    assert correct_no_token > wrong_with_token


def test_logged_m2_reward_captures_components():
    reward_fn = get_logged_m2_reward()
    reward_fn(
        completions=["<decision:compress:answer-now> <answer>Paris</answer>"],
        prompts=[None],
        gold_answer=["Paris"],
        tool_call_counts=[2],
        requires_search=[True],
        optimal_search_count=[2],
    )
    comp = get_last_m2_component_scores()
    assert set(comp.keys()) == {
        "format",
        "termination",
        "correctness",
        "necessity",
        "frugality",
        "decision",
    }
    assert comp["decision"][0] > 0.0
    assert comp["termination"] == [1.0]
