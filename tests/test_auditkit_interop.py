"""The run directory hands off to AuditKIT with no glue (CPU, no downloads).

A GRPO run writes ``trajectories.jsonl`` (tool calls as ``{"id", "name",
"arguments": dict}``, whatever the model's native call format), a standard PEFT
adapter folder, and ``lexsi_provenance.json``.
"""

import json

import torch
import transformers
from test_cohere_tool_calls import FALLBACK_CALL, _tokenizer

from agenttune.agentic.trajectory.dataset import Step, Trajectory, TrajectoryDataset
from agenttune.utils.provenance import PROVENANCE_FILE, read_provenance, write_provenance


def _traj(calls):
    return Trajectory(
        task="2+3?",
        steps=[
            Step(0, "s", {"tool_calls": calls}, "{'add': 5}", thought="native text"),
            Step(1, "t", {}, "5", thought="5"),
        ],
        final_response="5",
        logprobs=[(0.0, 1)],
        metadata={"prompt_ids": [1], "completion_ids": [2], "tool_call_count": len(calls)},
    )


def test_to_dict_normalises_calls_and_round_trips(tmp_path):
    calls = [
        {"id": "a1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2}'}},
        {"type": "function", "function": {"name": "add", "arguments": {"a": 1}}},
        {"type": "function", "function": {"name": "add", "arguments": "{not json"}},
        {"tool_name": "add", "parameters": {"a": 3}},
    ]
    d = _traj(calls).to_dict()
    assert d["steps"][0]["action"]["tool_calls"] == [
        {"id": "a1", "name": "add", "arguments": {"a": 2}},
        {"id": None, "name": "add", "arguments": {"a": 1}},
        {"id": None, "name": "add", "arguments": "{not json"},  # kept, not faked as {}
        {"id": None, "name": "add", "arguments": {"a": 3}},
    ]
    assert "logprobs" not in d and "prompt_ids" not in d["metadata"]
    assert calls[0]["function"]["arguments"] == '{"a": 2}'  # source not mutated

    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(d) + "\n")
    back = TrajectoryDataset.from_jsonl(str(path)).trajectories[0]
    assert back.steps[0].action["tool_calls"][0]["arguments"] == {"a": 2}


def test_provenance_embeds_dataset_lineage(tmp_path):
    data = tmp_path / "curated"
    data.mkdir()
    parent = {"schema": "lexsi.provenance/1", "library": "curatorkit"}
    (data / PROVENANCE_FILE).write_text(json.dumps(parent))
    rec = write_provenance(tmp_path / "out", "agentic.grpo", "base", str(data), "grpo")
    assert rec == read_provenance(tmp_path / "out")
    assert rec["schema"] == "lexsi.provenance/1" and rec["library"] == "agenttune"
    assert rec["inputs"] == [
        {"kind": "dataset", "ref": str(data), "config": "grpo", "provenance": parent}
    ]
    assert read_provenance(tmp_path / "missing") is None
    hub = write_provenance(tmp_path / "o2", "m", dataset="org/hub-id")
    assert hub["inputs"][0]["provenance"] is None


def test_grpo_run_dir_is_an_auditkit_handoff(tmp_path, monkeypatch):
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    tok = _tokenizer("tiny_aya_default.jinja")  # Tiny Aya's real template: fallback path
    cfg = transformers.Cohere2Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )  # fmt: skip
    base = tmp_path / "base"
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(base)
    tok.save_pretrained(base)

    state = {"n": 0}

    def fake_generate(self, input_ids=None, **_):  # force a fallback-format call on turn 1
        state["n"] += 1
        text = FALLBACK_CALL if state["n"] % 2 else "The answer is 5."
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([ids, torch.tensor([[tok.eos_token_id]])], 1)
        return torch.cat([input_ids, ids.repeat(input_ids.shape[0], 1)], 1)

    def add(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: first integer
            b: second integer
        """
        return a + b

    monkeypatch.setattr(transformers.GenerationMixin, "generate", fake_generate)
    out = tmp_path / "run"
    create_agentic_trainer(
        algorithm="grpo", backend="trl", model=str(base), tools=[add],
        train_dataset=Dataset.from_list([{"prompt": [{"role": "user", "content": "2+3?"}]}] * 2),
        reward_funcs=[lambda completions, **kw: [float(len(str(c)) % 3) for c in completions]],
        peft_config={"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"],
                     "task_type": "CAUSAL_LM"},
        output_dir=str(out), max_steps=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=1, num_generations=2, max_completion_length=16,
        max_steps_per_turn=2, use_vllm=False, report_to="none", save_steps=1000,
        warmup_steps=0, bf16=False, use_cpu=True, beta=0.0,
    ).train()  # fmt: skip

    records = [json.loads(line) for line in (out / "trajectories.jsonl").read_text().splitlines()]
    assert len(records) >= 2  # one per rollout
    for r in records:
        calls = r["steps"][0]["action"]["tool_calls"]
        assert [(c["name"], c["arguments"]) for c in calls] == [("add", {"a": 2, "b": 3})]
        assert r["reward"] is None and "completion_ids" not in r["metadata"]
        assert r["metadata"]["conversation"]  # per-call result pairing for AuditKIT

    adapter = json.loads((out / "adapter_config.json").read_text())
    assert adapter["base_model_name_or_path"] == str(base)
    assert (out / "tokenizer_config.json").exists()
    prov = read_provenance(out)
    assert prov["method"] == "agentic.grpo" and prov["base_model"] == str(base)
