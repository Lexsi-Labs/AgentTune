"""Answer rewards score one answer, taken from the final reply.

`reward_correct_answer` paid 1.5 when any number in the completion matched the gold, so
"41, 42 or 43" and "0;1;...;999" scored like "42". In agentic GRPO the completion also
holds the raw tool output, so a number the tool returned counted as the answer.
`answer_correctness_reward` paid 0.5 for an empty answer.
"""

import pytest

from agenttune.agentic.rewards.builtin_rewards import finqa, sql, use_case

SPRAY = ";".join(str(n) for n in range(1000))  # 1000 numbers in one whitespace "word"
TOOL_OUTPUT = "SKU A-7: on_hand=42 units, reorder_point=1000, last_audit=2023"


def _reply(text):
    return [{"role": "assistant", "content": text}]


def _decoded(final_reply, tool_output=TOOL_OUTPUT):
    """`completions` as TRL builds them for a rollout_func: completion_ids decoded into
    one assistant message, every turn and the raw tool output run together (shape taken
    from a Qwen2.5-0.5B-Instruct rollout)."""
    call = '<tool_call>\n{"name": "lookup", "arguments": {"sku": "A-7"}}\n</tool_call>'
    return _reply(f"{call}{tool_output}{final_reply}")


def _rollout(*conversations):
    """Reward kwargs as the agentic GRPO rollout sends them: `conversations` comes with
    `trajectories` (one per completion)."""
    return {"conversations": list(conversations), "trajectories": [object()] * len(conversations)}


def _conversation(final_reply, tool_output=TOOL_OUTPUT):
    """The rollout's `conversations` entry for the same trajectory."""
    call = {"function": {"name": "lookup", "arguments": {"sku": "A-7"}}}
    return [
        {"role": "user", "content": "How many units of SKU A-7 are on hand?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "content": tool_output},
        {"role": "assistant", "content": final_reply},
    ]


# ── reward_correct_answer ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("42", 1.5),
        ("The answer is 42.", 1.5),
        ("42.3", 0.5),
        ("41, 42 or 43", 0.0),
        ("43, 41 or 42", 1.5),  # a list scores like its last entry, never better
        (SPRAY, 0.0),
        ("<think>maybe 41</think>42", 1.5),
        ("<think>it is 42</think>41", 0.0),
        ("I could not work it out.", 0.0),
        ("40-44", 0.0),  # a range is not an answer
    ],
)
def test_reward_correct_answer_reads_one_number(text, expected):
    assert sql.reward_correct_answer([_reply(text)], answer=["42"]) == [expected]


@pytest.mark.parametrize("gold", ["42", "1000", "2023"])
def test_tool_output_is_not_the_answer(gold):
    reply = "I could not find that SKU."
    scores = sql.reward_correct_answer(
        [_decoded(reply)], answer=[gold], **_rollout(_conversation(reply))
    )
    assert scores == [0.0]


@pytest.mark.parametrize("gold,expected", [("42", 1.5), ("1000", 0.0), ("2023", 0.0)])
def test_final_reply_is_read_from_conversations(gold, expected):
    reply = "There are 42 units on hand."
    scores = sql.reward_correct_answer(
        [_decoded(reply)], answer=[gold], **_rollout(_conversation(reply))
    )
    assert scores == [expected]


@pytest.mark.parametrize(
    "text,gold",
    [
        ("3 units short on ORD-1003", "3"),
        ("7 units of SKU-C91", "7"),
        ("Gap is $96.00, due 2026-09-03", "96"),
        ("Found 42 rows in orders_2023", "42"),
        ("SKU A-7 has 42", "42"),
        ("12GB", "12"),
        ("Total: Rs.1,234", "1234"),
    ],
)
def test_identifiers_and_dates_after_the_answer_are_not_numbers(text, gold):
    assert sql.reward_correct_answer([_reply(text)], answer=[gold]) == [1.5]


def test_prompt_only_conversation_falls_back_to_the_completion():
    # Without tools the rollout records only the prompt as the conversation; a few-shot
    # assistant turn in it is not the model's reply.
    prompt = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "40+2?"},
    ]
    assert sql.reward_correct_answer([_reply("42")], answer=["42"], **_rollout(prompt)) == [1.5]


def test_empty_conversation_falls_back_to_the_completion():
    # A trajectory without recorded turns gets [] in `conversations`.
    assert sql.reward_correct_answer([_reply("42")], answer=["42"], **_rollout([])) == [1.5]
    assert sql.reward_correct_answer(["42"], answer=["42"], **_rollout([])) == [1.5]


def test_decoded_completion_without_conversations_reads_its_end():
    # Without `conversations` (agentic RLOO passes only completion_ids), the final reply
    # is still the end of the decoded text, so the last number is the model's answer.
    reply = "There are 42 units on hand."
    assert sql.reward_correct_answer([_decoded(reply)], answer=["42"]) == [1.5]
    assert sql.reward_correct_answer([_decoded(reply)], answer=["1000"]) == [0.0]


# ── other sql.py rewards that read the final reply ─────────────────────────────


def test_concise_reward_counts_the_final_reply_only():
    rows = " ".join(f"({n}, 'name{n}')" for n in range(100))  # a 200-word tool result
    reply = "There are 100 customers."
    scores = sql.reward_concise_answer(
        [_decoded(reply, rows)], **_rollout(_conversation(reply, rows))
    )
    assert scores == [0.3]


def test_concise_reward_needs_a_reply():
    # A rollout that ends on a tool call has no final reply.
    convo = _conversation("")[:-1] + [
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "lookup"}}]}
    ]
    assert sql.reward_concise_answer([_decoded("")], **_rollout(convo)) == [0.0]
    assert sql.reward_concise_answer([_reply("")]) == [0.0]


def test_format_reward_ignores_markdown_in_tool_output():
    report = "## Inventory\n- A-7: 42\n- B-2: 7"
    scores = sql.format_reward([_decoded("42", report)], **_rollout(_conversation("42", report)))
    assert scores == [0.0]


def test_yes_no_correctness_reads_the_final_reply():
    policy = "Refunds after 30 days: *no* exceptions."
    reply = "*yes*, it is refundable."
    scores = sql.correctness_reward(
        [_decoded(reply, policy)], answer=["yes"], **_rollout(_conversation(reply, policy))
    )
    assert scores == [0.6]


# ── answer_correctness_reward (use_case.py) ───────────────────────────────────


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("Paris", 1.0),
        ("paris", 1.0),
        ("Paris, France", 0.5),
        ("", 0.0),
        ("   ", 0.0),
        (".", 0.0),
        ("a", 0.0),
        ("Par", 0.0),
        ("Lyon", 0.0),
        ("Paris or Lyon or Marseille", 0.0),  # a list of candidates
    ],
)
def test_answer_correctness_needs_a_real_answer(pred, expected):
    scores = use_case.answer_correctness_reward(
        [None], [f"<answer>{pred}</answer>"], answer=["Paris"]
    )
    assert scores == [expected]


def test_answer_correctness_partial_credit_on_whole_words():
    score = use_case.answer_correctness_reward(
        [None], ["<answer>New York</answer>"], answer=["New York City"]
    )
    assert score == [0.5]


def test_answer_correctness_without_gold_scores_zero():
    # No gold used to give every answer 0.5 and an empty one 1.0.
    scores = use_case.answer_correctness_reward(
        [None, None], ["<answer>Paris</answer>", "<answer></answer>"]
    )
    assert scores == [0.0, 0.0]


# ── finqa answer_correctness_reward ───────────────────────────────────────────


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("42", 1.0),
        ("42, up from 30", 1.0),
        ("42 or 30", 1.0),
        ("30 or 42", 0.0),  # a list scores like its first entry, never better
        (SPRAY, 0.0),
        ("", 0.0),
    ],
)
def test_finqa_correctness_reads_one_number(pred, expected):
    scores = finqa.answer_correctness_reward([None], [f"<answer>{pred}</answer>"], answer=["42"])
    assert scores == [expected]


@pytest.mark.parametrize("pred,expected", [("", 0.0), ("a", 0.0), ("Paris, France", 0.5)])
def test_finqa_text_correctness_needs_a_real_answer(pred, expected):
    scores = finqa.answer_correctness_reward([None], [f"<answer>{pred}</answer>"], answer=["Paris"])
    assert scores == [expected]


def test_answer_correctness_fragment_of_a_long_gold_is_not_partial_credit():
    gold = ["The Walt Disney Company"]
    assert use_case.answer_correctness_reward([None], ["<answer>the</answer>"], answer=gold) == [
        0.0
    ]
    assert use_case.answer_correctness_reward(
        [None], ["<answer>Walt Disney Company</answer>"], answer=gold
    ) == [0.5]


def test_answer_correctness_per_row_none_gold_scores_zero():
    scores = use_case.answer_correctness_reward([None], ["<answer>None</answer>"], answer=[None])
    assert scores == [0.0]


def test_answer_rewards_do_not_read_tool_output():
    # No <answer> tag: the prediction is the final reply, not the decoded rollout.
    reply = "I could not compute it."
    hq = "HQ: Paris, France. Revenue grew 14.1%"
    kw = _rollout(_conversation(reply, hq))
    assert finqa.answer_correctness_reward(
        [None], [_decoded(reply, hq)], answer=["14.1"], **kw
    ) == [0.0]
    assert use_case.answer_correctness_reward(
        [None], [_decoded(reply, hq)], answer=["Paris"], **kw
    ) == [0.0]


def test_dataset_column_named_conversations_is_not_the_reply():
    # ShareGPT data standardized to role/content: the reference answer is an assistant
    # turn. Without `trajectories` it did not come from the rollout, so it is ignored.
    reference = [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "<answer>Paris</answer> 42"},
    ]
    kw = {"conversations": [reference]}
    assert use_case.answer_correctness_reward(
        [None], ["<answer>London</answer>"], answer=["Paris"], **kw
    ) == [0.0]
    assert sql.reward_correct_answer([_reply("7")], answer=["42"], **kw) == [0.0]


def test_a_thousands_group_is_not_read_alone():
    assert sql.reward_correct_answer([_reply("Total: Rs.1,234")], answer=["234"]) == [0.0]
