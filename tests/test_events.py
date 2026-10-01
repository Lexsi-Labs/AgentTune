import pytest

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.trajectory.dataset import Step, Trajectory
from agenttune.decide.state import PipelineState

# ---------- Task 1: Event + EventKind ----------


def test_event_kinds_exist():
    for name in [
        "TEXT",
        "REASONING",
        "TOOL_CALL",
        "TOOL_RESULT",
        "OBSERVATION",
        "TURN_COMPLETE",
        "REWARD",
        "MEMORY_OP",
    ]:
        assert hasattr(EventKind, name)


def test_event_defaults_and_fields():
    e = Event(kind=EventKind.TEXT, payload={"text": "hi"})
    assert e.kind is EventKind.TEXT
    assert e.payload == {"text": "hi"}
    assert e.token_span is None
    assert e.logprobs is None
    assert e.scope is None


def test_reward_event_carries_scope():
    e = Event(kind=EventKind.REWARD, payload={"value": 1.0}, scope="episode")
    assert e.scope == "episode"
    assert e.payload["value"] == 1.0


# ---------- Task 2: EventLog container ----------


def test_eventlog_append_len_iter():
    log = EventLog()
    assert log.tier == "light"
    log.append(Event(EventKind.TEXT, {"text": "a"}))
    log.append(Event(EventKind.TEXT, {"text": "b"}))
    assert len(log) == 2
    assert [e.payload["text"] for e in log] == ["a", "b"]


def test_rewards_filters_by_scope():
    log = EventLog(
        events=[
            Event(EventKind.REWARD, {"value": 0.5}, scope="step"),
            Event(EventKind.REWARD, {"value": 1.0}, scope="episode"),
            Event(EventKind.REWARD, {"value": 0.2}, scope="step"),
        ]
    )
    assert log.rewards("step") == [0.5, 0.2]
    assert log.rewards("episode") == [1.0]


def test_masked_tokens_requires_full_tier():
    light = EventLog(tier="light")
    with pytest.raises(ValueError, match="full"):
        light.masked_tokens()


def test_as_dataset_rows_requires_full_tier():
    light = EventLog(tier="light")
    with pytest.raises(ValueError, match="full"):
        light.as_dataset_rows("sft")


# ---------- Task 3: from_trajectory (full tier) ----------


def _sample_trajectory():
    return Trajectory(
        task="find the file",
        steps=[
            Step(
                step_number=0,
                state="s0",
                action={"name": "search", "arguments": {"q": "x"}},
                observation="found 3 hits",
                thought="I should search",
                reward=0.5,
            ),
        ],
        reward=1.0,
        final_response="the answer is 42",
        logprobs=[-0.1, -0.2, -0.05],
    )


def _rollout_wrapper_trajectory():
    """Mirrors the REAL action shape rollouts record:
    ``step.action = {"tool_calls": [openai-function-call, ...]}`` — the shape
    ``rollout_factory`` stores and ``from_trajectory`` copies verbatim. The
    OpenAI-style call carries string-encoded ``arguments``, exactly as an API
    model returns it."""
    return Trajectory(
        task="find the file",
        steps=[
            Step(
                step_number=0,
                state="s0",
                action={
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call_abc",
                            "function": {"name": "search", "arguments": '{"q": "x"}'},
                        }
                    ]
                },
                observation="found 3 hits",
                thought="I should search",
                reward=0.5,
            ),
        ],
        reward=1.0,
        final_response="the answer is 42",
        logprobs=[-0.1, -0.2, -0.05],
    )


def _multi_call_wrapper_trajectory():
    return Trajectory(
        task="multi",
        steps=[
            Step(
                step_number=0,
                state="s0",
                action={
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "c1",
                            "function": {"name": "search", "arguments": '{"q": "a"}'},
                        },
                        {
                            "type": "function",
                            "id": "c2",
                            "function": {"name": "read", "arguments": '{"p": "f"}'},
                        },
                    ]
                },
                observation="done",
                thought="two at once",
                reward=0.5,
            ),
        ],
        reward=1.0,
        final_response="done",
    )


def test_from_trajectory_is_full_tier():
    log = EventLog.from_trajectory(_sample_trajectory())
    assert log.tier == "full"


def test_from_trajectory_projects_step_events():
    log = EventLog.from_trajectory(_sample_trajectory())
    kinds = [e.kind for e in log]
    assert EventKind.REASONING in kinds
    assert EventKind.TOOL_CALL in kinds
    assert EventKind.TOOL_RESULT in kinds
    assert log.rewards("step") == [0.5]
    assert log.rewards("episode") == [1.0]


def test_from_trajectory_carries_logprobs_and_span():
    log = EventLog.from_trajectory(_sample_trajectory())
    assert log.masked_tokens() == [(0, 3)]
    text_events = [e for e in log if e.kind is EventKind.TEXT and e.logprobs]
    assert text_events[0].logprobs == [-0.1, -0.2, -0.05]


# ---------- Task 4: from_pipeline_state (light tier) ----------


def _sample_pipeline_state():
    st = PipelineState(
        pipeline_id="p1",
        template_id="bfsi/kyc",
        template_version="1",
        input_text="applicant John",
        input_hash="abc",
    )
    st.step_history = ["extract", "judge"]
    st.stage_outputs = {"extract": {"name": "John"}, "judge": {"ok": True}}
    st.verdict = "APPROVE"
    st.reason = "documents valid"
    st.confidence = 8
    return st


def test_from_pipeline_state_is_light_tier():
    log = EventLog.from_pipeline_state(_sample_pipeline_state())
    assert log.tier == "light"


def test_from_pipeline_state_projects_stages_and_reason():
    log = EventLog.from_pipeline_state(_sample_pipeline_state())
    tool_calls = [e for e in log if e.kind is EventKind.TOOL_CALL]
    assert [e.payload["stage"] for e in tool_calls] == ["extract", "judge"]
    texts = [e.payload["text"] for e in log if e.kind is EventKind.TEXT]
    assert "documents valid" in texts
    assert log.rewards("episode") == [0.8]


def test_light_tier_cannot_mask():
    log = EventLog.from_pipeline_state(_sample_pipeline_state())
    with pytest.raises(ValueError, match="full"):
        log.masked_tokens()


# ---------- Task 5: from_eval_dict (light tier) ----------


def test_from_eval_dict_projects_calls_and_outputs():
    d = {
        "tool_calls": [
            {"name": "search", "arguments": {"q": "a"}},
            {"name": "read", "arguments": {"p": "f"}},
        ],
        "tool_outputs": ["hit", "contents"],
    }
    log = EventLog.from_eval_dict(d)
    assert log.tier == "light"
    calls = [e.payload["action"]["name"] for e in log if e.kind is EventKind.TOOL_CALL]
    outs = [e.payload["output"] for e in log if e.kind is EventKind.TOOL_RESULT]
    assert calls == ["search", "read"]
    assert outs == ["hit", "contents"]


def test_from_eval_dict_handles_missing_keys():
    log = EventLog.from_eval_dict({})
    assert len(log) == 0
    assert log.tier == "light"


# ---------- Task 6: as_dataset_rows (full tier) ----------


def test_as_dataset_rows_sft_from_full_log():
    log = EventLog.from_trajectory(_sample_trajectory())
    rows = log.as_dataset_rows("sft")
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    roles = [m["role"] for m in msgs]
    assert "assistant" in roles
    assert "tool" in roles
    assert any("the answer is 42" in m["content"] for m in msgs)


def test_as_dataset_rows_tool_call_is_valid_json():
    """Regression: the <tool_call> content must be JSON in the bare
    {"name", "arguments"} shape that rollout_factory._extract_tool_calls()'s
    json.loads() + top-level-"name" check can actually parse at inference time.

    Uses the REAL rollout wrapper shape ({"tool_calls": [...]}, OpenAI-style
    function calls with string-encoded arguments) — the hand-made bare
    {"name": ...} action in _sample_trajectory would not catch this, since the
    wrapper is what from_trajectory() actually receives from a live rollout."""
    import json

    from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

    for traj in (_rollout_wrapper_trajectory(), _multi_call_wrapper_trajectory()):
        log = EventLog.from_trajectory(traj)
        rows = log.as_dataset_rows("sft")
        tool_call_msgs = [m for m in rows[0]["messages"] if "<tool_call>" in m["content"]]
        assert tool_call_msgs, "expected at least one <tool_call> message"
        for m in tool_call_msgs:
            inner = m["content"].removeprefix("<tool_call>").removesuffix("</tool_call>")
            parsed = json.loads(inner)  # raises if not valid JSON
            assert parsed["name"] in ("search", "read")
            assert isinstance(parsed["arguments"], dict)
            # round-trip through the actual inference parser: must parse, not return None
            assert _extract_tool_calls(None, m["content"]) is not None


def test_as_dataset_rows_bare_action_still_serializes():
    """The bare {"name", "arguments"} shape (spine/eval trajectories) must keep
    working, one message per call."""
    import json

    log = EventLog.from_trajectory(_sample_trajectory())
    rows = log.as_dataset_rows("sft")
    tool_call_msgs = [m for m in rows[0]["messages"] if "<tool_call>" in m["content"]]
    assert len(tool_call_msgs) == 1
    inner = tool_call_msgs[0]["content"].removeprefix("<tool_call>").removesuffix("</tool_call>")
    parsed = json.loads(inner)
    assert parsed == {"name": "search", "arguments": {"q": "x"}}


# ---------- Task 6: inference parser tolerance for near-miss model output ----------


def _parse(raw_text: str):
    from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

    return _extract_tool_calls(None, raw_text)


def test_parser_accepts_no_underscore_toolcall_tag():
    """Small fine-tunes sometimes imitate <tool_call> as <toolcall> (no
    underscore) and drop the matching closing tag. A correct call wrapped
    that way must still execute."""
    result = _parse(
        '<tool_call>\n<toolcall>\n{"name": "extract_table", '
        '"arguments": {"doc_id": "d1", "table_hint": "Cost of Sales"}}\n</toolcall>'
    )
    assert result is not None
    assert result[0]["function"]["name"] == "extract_table"


def test_parser_tolerates_stray_trailing_chars_after_json():
    """A trailing stray char (e.g. ')') after the closing brace of the JSON
    object must not silently drop a correct call."""
    result = _parse(
        '<tool_call>\n<toolcall>\n{"name": "calculate_metric", '
        '"arguments": {"expression": "1+1"}})\n</toolcall>'
    )
    assert result is not None
    assert result[0]["function"]["name"] == "calculate_metric"


def test_parser_rejects_unclosed_json():
    """Genuinely malformed output (outer object never closed - a stray ')'
    where the closing '}' should be) is NOT silently repaired: it must return
    None so the harness treats it as a stall/retry rather than execute garbage."""
    assert (
        _parse(
            '<tool_call>\n{"name": "extract_table", '
            '"arguments": {"doc_id": "d1", "table_hint": "x"})\n</tool_call>'
        )
        is None
    )


def test_parser_returns_none_for_plain_text():
    """Robustness must not over-accept: prose that happens to contain braces
    (or the literal string {name) must not be misparsed as a tool call."""
    assert _parse("I don't know the answer, let me think.") is None
    assert _parse("<answer>Paris</answer>") is None
    assert _parse("the value was {approximately 5} maybe") is None


def test_parser_does_not_crash_on_non_object_json():
    """DPO/GRPO completions often json.loads to a scalar or a list of ints.
    That is not a tool call; the parser must return None, not TypeError."""
    assert _parse("42") is None
    assert _parse("[1, 2, 3]") is None
    assert _parse('{"ok": true}') is None


def test_as_dataset_rows_unknown_fmt_raises():
    log = EventLog.from_trajectory(_sample_trajectory())
    with pytest.raises(ValueError, match="fmt"):
        log.as_dataset_rows("nonsense")


# ---------- Task 7: exports + cross-tier integration ----------


def test_exports_from_agentic_package():
    from agenttune.agentic import Event as E
    from agenttune.agentic import EventKind as K
    from agenttune.agentic import EventLog as L

    assert E is Event and K is EventKind and L is EventLog


def test_full_and_light_share_one_reward_api():
    full = EventLog.from_trajectory(_sample_trajectory())
    light = EventLog.from_pipeline_state(_sample_pipeline_state())
    assert full.rewards("episode") == [1.0]
    assert light.rewards("episode") == [0.8]
    assert full.masked_tokens() == [(0, 3)]
    with pytest.raises(ValueError):
        light.masked_tokens()
