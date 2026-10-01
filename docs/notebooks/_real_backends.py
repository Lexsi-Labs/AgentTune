"""
Shared real-model / real-data helpers for testing every documented feature end
to end. Models are small on purpose (limited GPU):

Generation:  Qwen/Qwen2.5-0.5B-Instruct        (~1GB on GPU, follows JSON-only
             instructions reliably)
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
    model, tok = _load_generator()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    encoded = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(_DEVICE)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True).strip()


def real_embed(texts: list[str]):
    model = _load_embedder()
    vecs = model.encode(list(texts), normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def real_embed_one(text: str) -> list[float]:
    return real_embed([text])[0]


def real_rollout_engine():
    global _rollout_engine
    if _rollout_engine is None:
        from agenttune.agentic.rollout_engines.transformers_engine import (
            TransformersRolloutEngine,
        )

        model, tok = _load_generator()
        _rollout_engine = TransformersRolloutEngine(model, tok)
    return _rollout_engine


def real_litellm_response(prompt: str, system: str | None = None, max_new_tokens: int = 64):
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


def extract_json_object(text: str) -> dict | None:
    """Extract the FIRST balanced {...} object from free-form model text.

    A naive greedy regex (`\\{.*\\}`) spans from the first '{' to the LAST '}' —
    if a small model rambles past its first JSON object and emits a second one,
    the greedy match swallows both and fails to parse. Bracket-counting finds
    just the first balanced object instead.
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


_QA_SYSTEM_PROMPT = (
    "You are a JSON-only API. Given a Text, output exactly one JSON object "
    '{"question": "...", "answer": "..."} — a short factual question about the '
    "Text and its short answer. No other output.\n\n"
    "Example\nText: The Great Wall of China is in China.\n"
    'Output: {"question": "Where is the Great Wall of China?", "answer": "China"}'
)


def real_qa_generate(text: str) -> list[tuple[str, str]]:
    out = real_generate(f"Text: {text}\nOutput:", system=_QA_SYSTEM_PROMPT, max_new_tokens=40)
    parsed = extract_json_object(out)
    if not parsed or "question" not in parsed or "answer" not in parsed:
        return []
    return [(parsed["question"], parsed["answer"])]


_DECISION_SYSTEM_PROMPT = (
    "You are a JSON-only API for content moderation decisions. You ONLY ever output "
    'a single JSON object shaped exactly like {"decision": "approve"} or '
    '{"decision": "reject"}. Never output any other words, explanation, or punctuation.\n\n'
    'Example\nInput: "Hello, how is your day?"\nOutput: {"decision": "approve"}'
)


async def real_decision_acompletion(*args, **kwargs):
    """Drop-in for litellm.acompletion returning a real JSON decision."""
    messages = kwargs.get("messages") or (args[0] if args else [])
    prompt = messages[-1]["content"]
    text = real_generate(prompt, system=_DECISION_SYSTEM_PROMPT, max_new_tokens=20)
    return {"choices": [{"message": {"content": text}}]}


def real_sft_train_step(train_dataset: list[dict]) -> dict:
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
