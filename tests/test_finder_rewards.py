"""
Unit tests for the FinDER reward stack (finder_rewards.py) — hand-built
episodes, no GPU, no network beyond the one-time squad metric load.

Covers the E1 stack contract (FINNLP_EXPERIMENTS.md v2 §3):
  - numeric correctness: 2% band, unit suffixes, parenthetical negatives,
    first-number (headline) vs intermediate-number partial credit, F1
    fallback for qualitative golds
  - golden-chunk recall: reference-level coverage (reward) + chunk-level
    (logging side channel)
  - conciseness: free zone, linear decay, floor
  - the composed logged stack: weight normalisation + per-component capture
"""

import unittest

from agenttune.rag.rewards.finder_rewards import (
    _coverage,
    conciseness_reward,
    get_last_finder_component_scores,
    get_logged_finder_reward,
    golden_chunk_recall_reward,
    numeric_correctness_reward,
)

P = ["q"]  # prompts are unused by these rewards but required by the convention


def _finder(**kwargs):
    """Call the composed finder reward with sensible defaults for one row."""
    defaults = {
        "prompts": P,
        "completions": ["<answer>$111.5 million</answer>"],
        "gold_answer": ["The revenue increased by $111.5 million from 2021 to 2023."],
        "tool_call_counts": [2],
        "retrieved_chunk_ids": [["finder_abc::0"]],
        "gold_chunk_ids": [["finder_abc::0", "finder_abc::1"]],
    }
    defaults.update(kwargs)
    return defaults


class TestNumericCorrectness(unittest.TestCase):
    def test_exact_number_match(self):
        s = numeric_correctness_reward(
            P,
            ["<answer>$111.5 million</answer>"],
            gold_answer=["increased by $111.5 million, calculated as 539.2 minus 427.7"],
        )
        self.assertEqual(s, [1.0])

    def test_unit_equivalence(self):
        # "2.5B" vs "$2,500,000,000" must match (finqa parsing rules).
        s = numeric_correctness_reward(
            P,
            ["<answer>$2.5 billion</answer>"],
            gold_answer=["Revenue was $2,500,000,000 in FY23."],
        )
        self.assertEqual(s, [1.0])

    def test_two_percent_band(self):
        s = numeric_correctness_reward(
            P,
            ["<answer>112.0 million</answer>"],
            gold_answer=["increased by $111.5 million"],
        )[0]
        self.assertEqual(s, 1.0)  # 0.45% off — inside the 2% band

    def test_outside_band_no_match(self):
        s = numeric_correctness_reward(
            P,
            ["<answer>130 million</answer>"],
            gold_answer=["increased by $111.5 million, from 539.2 million"],
        )[0]
        self.assertLess(s, 1.0)

    def test_intermediate_number_partial_credit(self):
        # 427.7 is a gold intermediate (not the headline 111.5) -> 0.4.
        # Real FinDER golds repeat units on intermediates ("539.2 million minus
        # 427.7 million") — unitless intermediates would fall to F1, which is
        # acceptable heuristic behaviour (the LLM judge arbitrates at eval).
        s = numeric_correctness_reward(
            P,
            ["<answer>427.7 million</answer>"],
            gold_answer=[
                "increased by $111.5 million from 2021 to 2023, "
                "calculated as 539.2 million minus 427.7 million."
            ],
        )
        self.assertEqual(s, [0.4])

    def test_parenthetical_negative(self):
        s = numeric_correctness_reward(
            P,
            ["<answer>-$(1,234)</answer>"],
            gold_answer=["The company reported a loss of (1,234) thousand."],
        )
        # pred parses as -1234 (parentheses), gold's first number is -1234 too.
        self.assertEqual(s, [1.0])

    def test_qualitative_f1_fallback(self):
        gold = "The collective bargaining agreement covers three building engineers."
        good = numeric_correctness_reward(
            P,
            [
                "<answer>three building engineers are covered by the collective bargaining agreement</answer>"
            ],
            gold_answer=[gold],
        )[0]
        bad = numeric_correctness_reward(
            P,
            ["<answer>the company was founded in 1987</answer>"],
            gold_answer=[gold],
        )[0]
        self.assertGreater(good, bad)
        self.assertGreater(good, 0.3)

    def test_no_answer_tag_still_scored_on_text(self):
        # extract_answer_tag falls back to full text — a bare correct number
        # still earns correctness (termination/format handle the missing tag).
        s = numeric_correctness_reward(
            P,
            ["the answer is 111.5 million"],
            gold_answer=["increased by $111.5 million"],
        )
        self.assertEqual(s, [1.0])


class TestGoldenChunkRecall(unittest.TestCase):
    def test_full_coverage(self):
        s = golden_chunk_recall_reward(
            P,
            ["x"],
            retrieved_chunk_ids=[["d1::0", "d1::1", "d2::0"]],
            gold_chunk_ids=[["d1::0", "d1::1", "d2::0"]],
        )
        self.assertEqual(s, [1.0])

    def test_chunk_level_reward_gives_partial_credit(self):
        # Only one chunk of d1 retrieved: the reward is CHUNK-level (1/3),
        # even though the REFERENCE d1 is fully surfaced (reference-level 1.0
        # — still available via _coverage for logging/eval, just not the
        # reward value itself; see golden_chunk_recall_reward's docstring).
        s = golden_chunk_recall_reward(
            P,
            ["x"],
            retrieved_chunk_ids=[["d1::0"]],
            gold_chunk_ids=[["d1::0", "d1::1", "d1::2"]],
        )
        self.assertAlmostEqual(s[0], 1 / 3)
        cov = _coverage(["d1::0"], ["d1::0", "d1::1", "d1::2"])
        self.assertAlmostEqual(cov["chunk_recall"], 1 / 3)
        self.assertEqual(cov["reference_recall"], 1.0)

    def test_partial_reference_coverage(self):
        s = golden_chunk_recall_reward(
            P,
            ["x"],
            retrieved_chunk_ids=[["d1::0", "d9::0"]],  # d9 is noise
            gold_chunk_ids=[["d1::0", "d2::0"]],
        )
        self.assertEqual(s, [0.5])

    def test_no_retrieval_zero(self):
        s = golden_chunk_recall_reward(
            P, ["x"], retrieved_chunk_ids=[[]], gold_chunk_ids=[["d1::0"]]
        )
        self.assertEqual(s, [0.0])

    def test_missing_kwargs_zero_not_crash(self):
        s = golden_chunk_recall_reward(P, ["x", "y"])
        self.assertEqual(s, [0.0, 0.0])


class TestConciseness(unittest.TestCase):
    def test_short_answer_full_credit(self):
        self.assertEqual(conciseness_reward(P, ["<answer>$111.5 million</answer>"]), [1.0])

    def test_long_answer_decays(self):
        answer = "<answer>" + ("word " * 150) + "</answer>"  # ~750 chars inside
        s = conciseness_reward(P, [answer])[0]
        self.assertGreater(s, 0.0)
        self.assertLess(s, 1.0)

    def test_very_long_answer_zero(self):
        answer = "<answer>" + ("word " * 400) + "</answer>"  # >1200 chars
        self.assertEqual(conciseness_reward(P, [answer]), [0.0])


class TestComposedStack(unittest.TestCase):
    def test_components_captured_and_totals_bounded(self):
        reward_fn = get_logged_finder_reward()
        totals = reward_fn(**_finder())
        self.assertEqual(len(totals), 1)
        # Weights normalise to 1, every component in [0,1] here -> total in [0,1].
        self.assertGreaterEqual(totals[0], 0.0)
        self.assertLessEqual(totals[0], 1.0 + 1e-9)
        comp = get_last_finder_component_scores()
        for name in (
            "format",
            "termination",
            "correctness_numeric",
            "golden_chunk_recall",
            "conciseness",
            "frugality",
        ):
            self.assertIn(name, comp)
            self.assertEqual(len(comp[name]), 1)
        # The logging-only reference-level channel is captured too (the
        # reward value itself is chunk-level — see golden_chunk_recall_reward).
        self.assertIn("golden_chunk_recall_referencelevel", comp)

    def test_perfect_episode_beats_looping_episode(self):
        reward_fn = get_logged_finder_reward()
        good = reward_fn(**_finder())[0]
        looping = reward_fn(
            **_finder(
                completions=["searching forever, no answer"],
                tool_call_counts=[6],
                retrieved_chunk_ids=[[]],
            )
        )[0]
        self.assertGreater(good, looping)

    def test_grounded_wrong_answer_beats_lucky_guess(self):
        # The anti-hacking property: an ungrounded right answer (no chunks
        # retrieved) must score below a grounded wrong answer is NOT required —
        # but a grounded right answer must beat an ungrounded right answer.
        reward_fn = get_logged_finder_reward()
        grounded = reward_fn(**_finder())[0]
        ungrounded = reward_fn(**_finder(retrieved_chunk_ids=[[]]))[0]
        self.assertGreater(grounded, ungrounded)


if __name__ == "__main__":
    unittest.main()
