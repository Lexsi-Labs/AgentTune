"""
Shared real-model / real-data helpers for the agentic_real test suite.

This module is the one place that talks to actual models, so every test file in
this folder swaps its fakes/mocks for these instead of hand-rolled canned
responses. Models are tiny on purpose — this box's GPU is limited — and are
loaded once per test session via module-level caching.

Generation:  Qwen/Qwen2.5-0.5B-Instruct        (~1GB on GPU, follows JSON-only
             instructions reliably — HuggingFaceTB/SmolLM2-135M-Instruct was
             tried first but was too small to follow structured-output
             instructions even with few-shot examples; see observations doc)
Embedding:   sentence-transformers/all-MiniLM-L6-v2 (~80MB, 384-dim)
"""

from __future__ import annotations

import json

import torch

GEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_gen_model = None
_gen_tokenizer = None
_embed_model = None
_rollout_engine = None


def _load_generator():
    global _gen_model, _gen_tokenizer
    if _gen_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
        _gen_model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_ID, dtype=torch.float16 if _DEVICE == "cuda" else torch.float32
        ).to(_DEVICE)
        _gen_model.eval()
    return _gen_model, _gen_tokenizer


def _load_embedder():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(EMBED_MODEL_ID, device=_DEVICE)
    return _embed_model


def real_generate(prompt: str, max_new_tokens: int = 64, system: str | None = None) -> str:
    """Run a real forward pass through SmolLM2-135M-Instruct and return the text."""
    model, tok = _load_generator()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # Newer transformers versions return a BatchEncoding here instead of a plain
    # tensor even without return_dict=True — unwrap it either way.
    if hasattr(inputs, "input_ids"):
        inputs = inputs.input_ids
    inputs = inputs.to(_DEVICE)
    attention_mask = torch.ones_like(inputs)
    with torch.no_grad():
        out = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.shape[1] :], skip_special_tokens=True).strip()


def real_rollout_engine():
    """Return a real production `TransformersRolloutEngine` (agenttune's own
    class, not a test fake) wrapping the small real generation model — a
    genuine drop-in replacement for the FakeEngine(RolloutEngine) stand-ins
    used across tests/agentic/test_*.py.
    """
    global _rollout_engine
    if _rollout_engine is None:
        from agenttune.agentic.rollout_engines.transformers_engine import (
            TransformersRolloutEngine,
        )

        model, tok = _load_generator()
        _rollout_engine = TransformersRolloutEngine(model, tok)
    return _rollout_engine


def real_embed(texts: list[str]):
    """Run real sentence-embedding inference; returns a list of float lists."""
    model = _load_embedder()
    vecs = model.encode(list(texts), normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def real_litellm_response(prompt: str, system: str | None = None, max_new_tokens: int = 64):
    """Build a litellm-shaped response object backed by a real generation call.

    Mirrors the MagicMock `resp.choices[0].message.content` shape the mocked
    tests patched `litellm.acompletion` with, so it's a drop-in replacement for
    those patch sites — but the content is genuinely produced by a real model
    forward pass instead of a canned string.
    """
    text = real_generate(prompt, max_new_tokens=max_new_tokens, system=system)

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    return _Resp(text)


def real_sft_train_step(train_dataset: list[dict]) -> dict:
    """Run ONE real gradient step of causal-LM SFT over `train_dataset` rows shaped
    like {"messages": [{"role": ..., "content": ...}, ...]}.

    This is a genuine forward+backward+optimizer.step() on a fresh copy of the
    small real generation model — not `trl`'s SFTTrainer (not installed in this
    env), but a
    real model actually updating real weights on real tokenized chat data,
    in place of the `_RecordingTrainer` stand-in from tests/agentic/test_train_wiring.py.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_ID, dtype=torch.float32).to(_DEVICE)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    texts = [tok.apply_chat_template(row["messages"], tokenize=False) for row in train_dataset]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(_DEVICE)
    out = model(**enc, labels=enc["input_ids"])
    out.loss.backward()
    optim.step()
    optim.zero_grad()

    return {"train_loss": float(out.loss.item()), "n_rows": len(train_dataset)}


_QA_SYSTEM_PROMPT = (
    "You are a JSON-only API. Given a Text, output exactly one JSON object "
    '{"question": "...", "answer": "..."} — a short factual question about the '
    "Text and its short answer. No other output.\n\n"
    "Example\nText: The Great Wall of China is in China.\n"
    'Output: {"question": "Where is the Great Wall of China?", "answer": "China"}'
)


def real_qa_generate(text: str) -> list[tuple[str, str]]:
    """Real drop-in for the `fake_generator`/`gen` callables in
    tests/rag/test_datagen.py: prompts the real small model for one grounded
    (question, answer) pair about `text` instead of returning a canned tuple.
    """
    out = real_generate(f"Text: {text}\nOutput:", system=_QA_SYSTEM_PROMPT, max_new_tokens=40)
    parsed = extract_json_object(out)
    if not parsed or "question" not in parsed or "answer" not in parsed:
        return []
    return [(parsed["question"], parsed["answer"])]


def extract_json_object(text: str) -> dict | None:
    """Extract the FIRST balanced {...} object from free-form model text.

    Small instruction-tuned models don't reliably emit *only* JSON the way a
    canned mock does, so real-model tests that need structured output should
    go through this instead of `json.loads(text)` directly. Bracket-counting
    (rather than a greedy `\\{.*\\}` regex) matters here: if the model rambles
    past its first JSON object and emits a second one, a greedy match spans
    from the first '{' to the LAST '}' and swallows both, failing to parse.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
