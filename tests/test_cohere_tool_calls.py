"""Cohere tool calling (Tiny Aya, Command R7B, Aya Expanse, Aya Vision, North).

CPU only, no downloads: the chat templates are vendored in
tests/fixtures/chat_templates (Tiny Aya's real ``default`` template, which
ignores ``tools`` and renders a ``tool`` turn empty; Command R7B's ``tool_use``
template, the Cohere format that does render tools; Aya Expanse's only
template), the tokenizer is a lossless byte-level stand-in with the Cohere
special tokens, and the model is a tiny random ``cohere2`` / ``cohere`` build.
"""

import json
import warnings
from pathlib import Path

import pytest
import torch
import transformers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from agenttune.agentic.rollout_engines import rollout_factory
from agenttune.agentic.rollout_engines.rollout_factory import (
    _extract_tool_calls,
    _render_chat_template,
    create_rollout_fn,
)

TEMPLATES = Path(__file__).parent / "fixtures" / "chat_templates"
COHERE_SPECIALS = [
    "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>",
    "<|SYSTEM_TOKEN|>", "<|START_THINKING|>", "<|END_THINKING|>", "<|START_ACTION|>",
    "<|END_ACTION|>", "<|START_RESPONSE|>", "<|END_RESPONSE|>", "<|START_TOOL_RESULT|>",
    "<|END_TOOL_RESULT|>",
]  # fmt: skip
R7B_CALL = (
    "<|START_THINKING|>I will add the numbers.<|END_THINKING|><|START_ACTION|>[\n"
    '    {"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}\n'
    "]<|END_ACTION|>"
)
# What the fallback prompt asks for, and what real Tiny Aya writes.
FALLBACK_CALL = '[{"tool_name": "add", "parameters": {"a": 2, "b": 3}}]'
ADD = {"type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}


def _tokenizer(template: str) -> PreTrainedTokenizerFast:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<BOS_TOKEN>",
        eos_token="<|END_OF_TURN_TOKEN|>",
        pad_token="<PAD>",
        additional_special_tokens=COHERE_SPECIALS,
        chat_template=(TEMPLATES / template).read_text(),
    )


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: first integer
        b: second integer
    """
    return a + b


# ── Parser ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        R7B_CALL,
        # What the trainer path sees: decoded with skip_special_tokens=True.
        'I will add the numbers.[\n    {"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}\n]',
        # Command-R / Aya Expanse style.
        'Action: ```json\n[\n    {\n        "tool_name": "add",\n        "parameters": {"a": 2, "b": 3}\n    }\n]\n```',
    ],
)
def test_cohere_tool_call_formats(text):
    assert _extract_tool_calls({"role": "assistant", "content": text}, text) == [ADD]


def test_cohere_multiple_calls_and_skip_special_tokens():
    tok = _tokenizer("command_r7b_tool_use.jinja")
    text = (
        "<|START_THINKING|>Two [sums].<|END_THINKING|><|START_ACTION|>["
        '{"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}, '
        '{"tool_call_id": "1", "tool_name": "add", "parameters": {"a": 1, "b": 1}}'
        "]<|END_ACTION|>"
    )
    stripped = tok.decode(tok.encode(text, add_special_tokens=False), skip_special_tokens=True)
    assert "START_ACTION" not in stripped
    calls = _extract_tool_calls({"role": "assistant", "content": stripped}, stripped)
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [
        {"a": 2, "b": 3},
        {"a": 1, "b": 1},
    ]


def test_cohere_plain_answer_is_not_a_tool_call():
    text = "<|START_RESPONSE|>The answer is [5].<|END_RESPONSE|>"
    assert _extract_tool_calls({"role": "assistant", "content": text}, text) is None


# ── Rollout loop with scripted outputs ───────────────────────────────────────


class _ScriptedEngine:
    """Returns scripted completions sequentially; records rendered prompts."""

    def __init__(self, tok, first, wrap=lambda text: text):
        if isinstance(first, list):
            self.outputs = list(first)
        else:
            self.outputs = [first, "The answer is 5."]
        self.tok = tok
        self.prompts = []
        self.wrap = wrap  # what the raw decode adds around the text

    def _get_tokenizer(self):
        return self.tok

    def generate(self, prompts, tools=None, gen_cfg=None):
        from agenttune.agentic.rollout_engines.rollout_factory import (
            _render_chat_template,
        )

        self.prompts.append(_render_chat_template(self.tok, prompts, tools, tokenize=False))
        idx = min(len(self.prompts) - 1, len(self.outputs) - 1)
        out = self.wrap(self.outputs[idx])
        ids = self.tok(out, add_special_tokens=False).input_ids
        return {"completions": [out], "logprobs": [None], "metadata": {"completion_ids": [ids]}}


def _run(template, first, **kw):
    tok = _tokenizer(template)
    engine = _ScriptedEngine(tok, first)
    max_steps = kw.pop("max_steps", 3)
    out = create_rollout_fn(rollout_engine=engine, tools=[add], max_steps=max_steps, **kw)(
        ["What is 2+3?"]
    )
    return engine, out["trajectories"][0]


def test_command_r7b_template_round_trip():
    engine, traj = _run("command_r7b_tool_use.jinja", R7B_CALL)
    assert traj.metadata["tool_call_count"] == 1
    turn2 = engine.prompts[1]
    assert '"tool_name": "add"' in turn2
    # Arguments are rendered as an object, not a double-encoded JSON string.
    assert '"parameters": {"a": 2, "b": 3}' in turn2
    assert "<|START_TOOL_RESULT|>" in turn2 and '"0": "5"' in turn2


def test_tiny_aya_template_uses_the_fallback():
    with pytest.warns(UserWarning, match="ignores `tools`"):
        engine, traj = _run("tiny_aya_default.jinja", FALLBACK_CALL)
    assert traj.metadata["tool_call_count"] == 1
    assert "Add two integers" in engine.prompts[0]  # schema injected into system prompt
    # The result reaches the model as a user turn instead of Tiny Aya's empty tool turn.
    assert "<|USER_TOKEN|>Tool result for add: 5" in engine.prompts[1]
    assert traj.final_response == "The answer is 5."


def test_aya_expanse_template_uses_documented_fallback():
    with pytest.warns(UserWarning, match="ignores `tools`"):
        engine, traj = _run("aya_expanse.jinja", R7B_CALL)
    assert traj.metadata["tool_call_count"] == 1
    assert "Add two integers" in engine.prompts[0]  # schema injected into system prompt
    assert "Tool result for add: 5" in engine.prompts[1]  # tool turn folded into a user turn


def test_tool_result_format_is_overridable():
    with pytest.warns(UserWarning, match="ignores `tools`"):
        engine, _ = _run("aya_expanse.jinja", R7B_CALL, tool_result_format="[{name}] -> {content}")
    assert "[add] -> 5" in engine.prompts[1]


def test_control_tokens_stay_out_of_the_answer_and_the_next_prompt():
    # TransformersRolloutEngine puts the generation prompt back in front of the
    # decode, and the raw decode keeps the end-of-turn token.
    def wrap(text):
        return f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{text}<|END_OF_TURN_TOKEN|>"

    tok = _tokenizer("aya_expanse.jinja")
    engine = _ScriptedEngine(tok, R7B_CALL, wrap)
    traj = create_rollout_fn(rollout_engine=engine, tools=[add], max_steps=3)(["2+3?"])[
        "trajectories"
    ][0]
    assert traj.final_response == "The answer is 5."
    # One CHATBOT_TOKEN for the model's call, one for the generation prompt.
    assert engine.prompts[1].count("<|CHATBOT_TOKEN|>") == 2


def test_aya_expanse_template_can_raise_instead():
    with pytest.raises(ValueError, match="ignores `tools`"):
        _run("aya_expanse.jinja", R7B_CALL, tools_fallback_prompt=None)


# ── Real GRPO trainer path on tiny random models ─────────────────────────────


@pytest.mark.parametrize(
    "config_cls,template",
    [
        (transformers.Cohere2Config, "command_r7b_tool_use.jinja"),  # Command R7B
        (transformers.Cohere2Config, "tiny_aya_default.jinja"),  # Tiny Aya (fallback)
        (transformers.CohereConfig, "aya_expanse.jinja"),  # Aya Expanse
    ],
)
def test_agentic_grpo_executes_cohere_tool_call(tmp_path, monkeypatch, config_cls, template):
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    tok = _tokenizer(template)
    cfg = config_cls(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )  # fmt: skip
    model_dir = tmp_path / "model"
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    # Random weights won't emit a call: force the first turn to R7B's native one
    # (also parsed on the fallback path).
    calls, state = [], {"n": 0}

    def fake_generate(self, input_ids=None, **_):
        state["n"] += 1
        text = R7B_CALL if state["n"] % 2 else "The answer is 5."
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([ids, torch.tensor([[tok.eos_token_id]])], 1)
        return torch.cat([input_ids, ids.repeat(input_ids.shape[0], 1)], 1)

    def add(a: int, b: int) -> int:  # noqa: F811 -- same name as the forced call
        """Add two integers.

        Args:
            a: first integer
            b: second integer
        """
        calls.append((a, b))
        return a + b

    monkeypatch.setattr(transformers.GenerationMixin, "generate", fake_generate)
    trainer = create_agentic_trainer(
        algorithm="grpo", backend="trl", model=str(model_dir), tools=[add],
        train_dataset=Dataset.from_list([{"prompt": [{"role": "user", "content": "2+3?"}]}] * 2),
        reward_funcs=[lambda completions, **kw: [float(len(str(c)) % 3) for c in completions]],
        output_dir=str(tmp_path / "out"), max_steps=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=1, num_generations=2, max_completion_length=16,
        max_steps_per_turn=2, use_vllm=False, report_to="none", save_steps=1000,
        warmup_steps=0, bf16=False, use_cpu=True, beta=0.0,
    )  # fmt: skip
    trainer.train()
    assert calls and all(c == (2, 3) for c in calls)


@pytest.mark.parametrize(
    "text", ["[2, 3]", "[]", '[{"name": "add", "arguments": {}}, 3]', "[[{}]]"]
)
def test_json_list_of_non_calls_is_not_a_tool_call(text):
    # A plain list answer used to raise TypeError and abort the GRPO step.
    assert _extract_tool_calls({"role": "assistant", "content": text}, text) is None


def test_single_tool_names_entry_is_a_call():
    # Tiny Aya's greedy reply to a weather prompt (real output, markers stripped).
    text = '\n{\n  "tool_names": ["get_weather"],\n  "parameters": {\n    "city": "Tokyo"\n  }\n}\n'
    assert _extract_tool_calls({"role": "assistant", "content": text}, text) == [
        {"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}
    ]
    two = '{"tool_names": ["a", "b"], "parameters": {}}'
    assert _extract_tool_calls({"role": "assistant", "content": two}, two) is None


def test_tool_turn_the_template_renders_empty_is_folded_not_dropped(monkeypatch):
    # Tiny Aya's real template renders a "tool" turn as an empty turn, no error.
    monkeypatch.setattr(rollout_factory, "_TOOL_TURN_DROPPED", {})
    tok = _tokenizer("tiny_aya_default.jinja")
    conv = [
        {"role": "user", "content": "What is 2+3?"},
        {"role": "assistant", "content": "", "tool_calls": [ADD]},
        {"role": "tool", "name": "add", "content": "5"},
    ]
    with pytest.warns(UserWarning, match="empty turn"):
        out = _render_chat_template(tok, conv, tokenize=False)
    assert "Tool result for add: 5" in out
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warns once per template
        assert _render_chat_template(tok, conv, tokenize=False) == out


def test_aya_expanse_fallback_max_steps_final_answer():
    """When max_steps is reached after a tool turn on fallback, final answer is generated."""
    with pytest.warns(UserWarning, match="ignores `tools`"):
        engine, traj = _run("aya_expanse.jinja", FALLBACK_CALL, max_steps=1)
    assert traj.metadata["tool_call_count"] == 1
    assert traj.final_response == "The answer is 5."
    assert len(traj.steps) == 2


def test_aya_expanse_fallback_sequential_multi_turn():
    """Multi-step tool calling works with fallback folding and valid role alternation."""
    tok = _tokenizer("aya_expanse.jinja")
    engine = _ScriptedEngine(tok, [FALLBACK_CALL, FALLBACK_CALL, "The answer is 5."])
    with pytest.warns(UserWarning, match="ignores `tools`"):
        out = create_rollout_fn(rollout_engine=engine, tools=[add], max_steps=3)(["What is 2+3?"])
    traj = out["trajectories"][0]
    assert traj.metadata["tool_call_count"] == 2
    assert traj.final_response == "The answer is 5."
    for prompt in engine.prompts:
        assert "<|START_OF_TURN_TOKEN|>" in prompt


def test_aya_expanse_fallback_force_final_answer():
    """A turn after the tool result that never answers triggers the force-final
    nudge, merged into a user turn under fold_tool_results."""
    tok = _tokenizer("aya_expanse.jinja")
    stall = "<|START_THINKING|>I am calculating...<|END_THINKING|>"
    engine = _ScriptedEngine(tok, [FALLBACK_CALL, stall, "<answer>5</answer>"])
    with pytest.warns(UserWarning, match="ignores `tools`"):
        out = create_rollout_fn(
            rollout_engine=engine, tools=[add], max_steps=1, force_final_answer=True
        )(["What is 2+3?"])
    traj = out["trajectories"][0]
    assert len(engine.prompts) == 3  # call, stalled final turn, nudged answer
    assert "Tool result for add: 5" in engine.prompts[1]  # the result reaches the next prompt
    assert "wrapped in <answer></answer> tags" in engine.prompts[2]
    assert traj.final_response == "<answer>5</answer>"


def test_aya_expanse_fallback_context_truncation():
    """_truncate_conversation_for_vllm drops the oldest (call, folded result) pair."""
    from agenttune.agentic.rollout_engines.rollout_factory import (
        _truncate_conversation_for_vllm,
    )

    tok = _tokenizer("aya_expanse.jinja")
    conv = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "call 1"},
        {"role": "user", "content": "Tool result for add: 1"},
        {"role": "assistant", "content": "call 2"},
        {"role": "user", "content": "Tool result for add: 2"},
    ]
    _truncate_conversation_for_vllm(conv, tok, schemas=None, max_prompt_len=50)
    rendered = _render_chat_template(tok, conv, tokenize=False)
    assert "Tool result for add: 2" in rendered
    assert "call 1" not in rendered
    assert [m["role"] for m in conv] == ["system", "user", "assistant", "user"]


def test_truncation_recognises_a_custom_tool_result_format():
    from agenttune.agentic.rollout_engines.rollout_factory import (
        _truncate_conversation_for_vllm,
    )

    tok = _tokenizer("aya_expanse.jinja")
    fmt = "[{name}] -> {content}"
    conv = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "call 1"},
        {"role": "user", "content": "[add] -> 1"},
        {"role": "assistant", "content": "call 2"},
        {"role": "user", "content": "[add] -> 2"},
    ]
    _truncate_conversation_for_vllm(
        conv, tok, schemas=None, max_prompt_len=50, tool_result_format=fmt
    )
    # The (call 1, result 1) pair goes together, so roles still alternate.
    assert [m["content"] for m in conv] == ["system", "q", "call 2", "[add] -> 2"]


def test_render_folds_with_the_callers_tool_result_format(monkeypatch):
    monkeypatch.setattr(rollout_factory, "_TOOL_TURN_DROPPED", {})
    tok = _tokenizer("tiny_aya_default.jinja")
    conv = [
        {"role": "user", "content": "What is 2+3?"},
        {"role": "assistant", "content": "", "tool_calls": [ADD]},
        {"role": "tool", "name": "add", "content": "5"},
    ]
    with pytest.warns(UserWarning, match="empty turn"):
        out = _render_chat_template(
            tok, conv, tokenize=False, tool_result_format="<{name}={content}>"
        )
    assert "<add=5>" in out


def test_template_that_raises_on_tools_does_not_render_them():
    from agenttune.agentic.rollout_engines.rollout_factory import _template_renders_tools

    tok = _tokenizer("aya_expanse.jinja")
    tok.chat_template = (
        "{% if tools %}{{ raise_exception('tools not supported') }}{% endif %}"
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    )
    schema = rollout_factory._build_tool_schema(add)
    assert _template_renders_tools(tok, [schema]) is False
    # A template that does render them is still detected.
    assert _template_renders_tools(_tokenizer("command_r7b_tool_use.jinja"), [schema]) is True


def test_engine_prefill_drops_the_assistant_turn_opener():
    """TransformersRolloutEngine puts back text the generation prompt adds beyond
    the assistant-turn opener (e.g. Functionary's ">>>"), not the opener itself
    ("<|im_start|>assistant\n"), which leaked into answers as "assistant\n..."."""
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    chatml = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{{ extra }}{% endif %}"
    )
    engine = TransformersRolloutEngine.__new__(TransformersRolloutEngine)
    engine.tokenizer = _tokenizer("aya_expanse.jinja")
    conv = [[{"role": "user", "content": "hi"}]]
    engine._chat_template = chatml
    assert engine._compute_assistant_prefill(conv, None, False) == [""]
    engine._chat_template = chatml.replace("{{ extra }}", ">>>")
    assert engine._compute_assistant_prefill(conv, None, False) == [">>>"]
