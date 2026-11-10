"""LLMJudge reply parsing: only a finite score is trusted."""

import asyncio

import pytest

from agenttune.agentic.rewards.llm_judge import LLMJudge, _parse_absolute, _parse_relative

NON_FINITE = [
    '{"score": NaN}',
    '{"score": "NaN"}',
    '{"score": Infinity}',
    '{"score": -Infinity}',
    '{"score": NaN, "explanation": "failed at step 1"}',
]


@pytest.mark.parametrize("reply", NON_FINITE)
def test_absolute_non_finite_score_is_zero(reply):
    assert _parse_absolute(reply).score == 0.0


@pytest.mark.parametrize("value", ["NaN", "Infinity"])
def test_relative_non_finite_score_is_zero(value):
    reply = f'{{"scores": [{{"id": "1", "score": {value}}}, {{"id": "2", "score": 0.2}}]}}'
    assert [s.score for s in _parse_relative(reply, 2)] == [0.0, 0.2]


@pytest.mark.parametrize("raw,expected", [(0.8, 0.8), (1.7, 1.0), (-0.2, 0.0)])
def test_finite_scores_are_still_clamped(raw, expected):
    assert _parse_absolute(f'{{"score": {raw}}}').score == expected
    reply = f'{{"scores": [{{"id": "1", "score": {raw}}}]}}'
    assert _parse_relative(reply, 1)[0].score == expected


def test_judge_does_not_reward_nan(monkeypatch):
    async def acall(self, messages):
        return '{"score": NaN}'

    monkeypatch.setattr(LLMJudge, "_call", lambda self, messages: '{"score": NaN}')
    monkeypatch.setattr(LLMJudge, "_acall", acall)
    judge = LLMJudge(rollout_engine=object())
    assert judge.evaluate_trajectory("task", "answer") == 0.0
    assert asyncio.run(judge.async_evaluate_trajectory("task", "other answer")) == 0.0
