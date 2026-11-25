"""Tool turns in a multi-turn rollout's completion ids are the template's tokens.

The trainer scores ``prompt_ids + completion_ids``. For each generated token to be
scored in the context it was sampled from, the ids between two generations must be
what the chat template put there (the end of the model's turn, the tool turns with
their markers, the next generation prompt), not the bare tool result. CPU only:
vendored templates (Qwen3-0.6B's in fixtures/chat_templates/qwen3.jinja), byte-level
stand-in tokenizers and the scripted engine of test_cohere_tool_calls.
"""

import json
import warnings

import pytest
from test_cohere_tool_calls import (
    FALLBACK_CALL,
    R7B_CALL,
    TEMPLATES,
    _ScriptedEngine,
    _tokenizer,
    add,
)
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from agenttune.agentic.rollout_engines import rollout_factory
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

QWEN_TAGS = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"]  # fmt: skip


def _qwen3_tokenizer() -> PreTrainedTokenizerFast:
    # Like Qwen3's: <|im_start|>/<|im_end|> special, the tags added but not special.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
        chat_template=(TEMPLATES / "qwen3.jinja").read_text(),
    )
    tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in QWEN_TAGS])
    return tok


def _qwen3_call(name="add", a=2, b=3):
    args = json.dumps({"name": name, "arguments": {"a": a, "b": b}})
    return f"<tool_call>\n{args}\n</tool_call>"


def _rollout(tok, outputs):
    engine = _ScriptedEngine(tok, outputs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = create_rollout_fn(rollout_engine=engine, tools=[add], max_steps=3)(["What is 2+3?"])
    # The template's tool turn was rendered, not replaced by the bare-result fallback.
    assert not [w for w in caught if "Could not render the tool turn" in str(w.message)]
    return engine, out


def _between_generations(tok, out, first):
    """The decoded masked ids right after the first generation."""
    ids, mask = out["completion_ids"][0], out["env_mask"][0]
    n = len(tok(first, add_special_tokens=False).input_ids)
    assert mask[:n] == [1] * n  # the model's first turn is trained on
    env = []
    for i, m in zip(ids[n:], mask[n:], strict=True):
        if m:
            break
        env.append(i)
    assert env, "no tool-turn ids after the first generation"
    return tok.decode(env, skip_special_tokens=False)


def test_tiny_aya_completion_is_the_sampled_text():
    # Tiny Aya renders the model's turn verbatim, so prompt + completion must be
    # exactly the text the second turn was sampled from, plus that turn.
    tok = _tokenizer("tiny_aya_default.jinja")
    engine, out = _rollout(tok, [FALLBACK_CALL, "The answer is 5."])
    full = tok.decode(out["prompt_ids"][0] + out["completion_ids"][0], skip_special_tokens=False)
    assert full == engine.prompts[1] + "The answer is 5."


@pytest.mark.parametrize(
    "template,first,expected",
    [
        (
            "tiny_aya_default.jinja",
            FALLBACK_CALL,
            "<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            "Tool result for add: 5\nUse this result to answer the original question."
            "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>",
        ),
        (
            "aya_expanse.jinja",
            R7B_CALL,
            "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            "Tool result for add: 5\nUse this result to answer the original question."
            "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        ),
        (
            "command_r7b_tool_use.jinja",
            R7B_CALL,
            "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TOOL_RESULT|>",
        ),
    ],
)
def test_tool_turn_is_wrapped_like_the_next_prompt(template, first, expected):
    tok = _tokenizer(template)
    engine, out = _rollout(tok, [first, "The answer is 5."])
    text = _between_generations(tok, out, first)
    assert text.startswith(expected)
    assert engine.prompts[1].endswith(text)  # ends with the next generation prompt


def test_end_of_turn_the_model_generated_is_not_doubled():
    tok = _tokenizer("tiny_aya_default.jinja")
    for first in (
        FALLBACK_CALL + "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>",
        FALLBACK_CALL + "<|END_OF_TURN_TOKEN|>",  # EOS only
    ):
        _, out = _rollout(tok, [first, "The answer is 5."])
        text = _between_generations(tok, out, first)
        assert text.startswith("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Tool result for add: 5")


def test_qwen3_template_that_rerenders_the_last_turn():
    # Qwen3 renders the last assistant turn with an empty <think> block, so the
    # render before the tool turn is not a prefix of the next prompt.
    tok = _qwen3_tokenizer()
    first = _qwen3_call() + "<|im_end|>"
    engine, out = _rollout(tok, [first, "The answer is 5.<|im_end|>"])
    text = _between_generations(tok, out, first)
    assert text == (
        "\n<|im_start|>user\n<tool_response>\n5\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert engine.prompts[1].endswith(text)


def test_qwen3_two_calls_in_one_step():
    tok = _qwen3_tokenizer()
    first = _qwen3_call(a=2, b=3) + "\n" + _qwen3_call(a=1, b=1) + "<|im_end|>"
    engine, out = _rollout(tok, [first, "Done.<|im_end|>"])
    text = _between_generations(tok, out, first)
    assert text == (
        "\n<|im_start|>user\n<tool_response>\n5\n</tool_response>\n"
        "<tool_response>\n2\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert engine.prompts[1].endswith(text)


def test_call_the_template_rewrites():
    # Command R's legacy "Action:" call is re-rendered in R7B's action format, so
    # none of the generated text is found in the turn: the tool turn must still
    # start with the turn's markers, not part-way through them.
    tok = _tokenizer("command_r7b_tool_use.jinja")
    first = (
        'Action: ```json\n[\n    {\n        "tool_name": "add",\n'
        '        "parameters": {"a": 2, "b": 3}\n    }\n]\n```'
    )
    engine, out = _rollout(tok, [first, "The answer is 5."])
    text = _between_generations(tok, out, first)
    assert text.startswith(
        "<|END_ACTION|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
        "<|START_TOOL_RESULT|>"
    )
    assert engine.prompts[1].endswith(text)


def test_bare_results_if_the_tool_turn_cannot_be_rendered(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("template error")

    monkeypatch.setattr(rollout_factory, "_tool_turn_ids", broken)
    tok = _tokenizer("tiny_aya_default.jinja")
    engine = _ScriptedEngine(tok, [FALLBACK_CALL, "The answer is 5."])
    with pytest.warns(UserWarning, match="Could not render the tool turn"):
        out = create_rollout_fn(rollout_engine=engine, tools=[add], max_steps=3)(["2+3?"])
    assert _between_generations(tok, out, FALLBACK_CALL) == "5"
    assert len(out["completion_ids"][0]) == len(out["env_mask"][0]) == len(out["logprobs"][0])
