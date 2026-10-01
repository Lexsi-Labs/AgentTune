"""
Embedding client — injectable, like the LLM client.

Real backend: Qwen3-Embedding-8B on the A100 via sentence-transformers (the
model has first-class ST support; vLLM/SGLang also serve it, but ST is the
simplest robust path and avoids a second server). 8B at fp16 ≈ 16GB, fits
comfortably on a 40GB A100 with headroom for the graph code.

Qwen3-Embedding requires an instruction prefix for queries (asymmetric):
  query:    "Instruct: {task}\nQuery: {text}"
  document: no prefix (passage side is unprefixed)
Per the official model card (QwenLM/Qwen3-Embedding). We use task =
"Given a web document, retrieve passages that answer the question" for
questions, matching the RAG retrieval task.
"""

from __future__ import annotations

from typing import Protocol

QUERY_INSTRUCTION = (
    "Instruct: Given a web document, retrieve passages that answer the question.\nQuery:"
)


class Embedder(Protocol):
    """Interface every embedder implements. Injectable for testing."""

    model_name: str

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class Qwen3Embedder:
    """Qwen3-Embedding-8B via sentence-transformers, on GPU.

    Loads once (cached); batch-embeds. dims default = the model's native 4096
    (configurable down to 1024 via `truncate_dim` if memory is tight — the
    model supports flexible dimensions).

    NOTE: 8B at fp16 ≈ 16GB params but ~30GB resident VRAM once loaded. On a
    40GB A100 this leaves little headroom and makes the pipeline fragile to
    repeated loads (each colab exec that touches the model holds a copy). For
    robustness prefer `BGEM3Embedder` (the default in `build_dataset`) which
    uses <3GB and leaves the GPU free for everything else.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._bs = batch_size
        self._model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": "auto"},
        )

    def embed_queries(self, texts):
        return self._model.encode(
            texts,
            prompt=QUERY_INSTRUCTION,
            batch_size=self._bs,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

    def embed_documents(self, texts):
        return self._model.encode(
            texts,
            batch_size=self._bs,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()


class BGEM3Embedder:
    """BAAI/bge-m3 via HF transformers directly (FlagEmbedding model).

    Chosen for robustness over Qwen3-Embedding-8B: 568M params, ~2.3GB VRAM at
    fp16, leaving ~37GB free on a 40GB A100 — eliminates the OOM fragility that
    Qwen3's 30GB footprint caused. MTEB retrieval ~63 (vs Qwen3's ~58 on the
    comparable retrieval split; both are strong). MIT license, 8K context,
    multilingual. Uses transformers `AutoModel` + mean-pooling + L2-normalize,
    the standard BGE serving recipe (no sentence-transformers dependency needed
    for this path, though it's installed anyway).

    BGE-M3 is symmetric (no query/document prefix needed), unlike Qwen3.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self._bs = batch_size
        self._maxlen = max_length
        self._device = device
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModel.from_pretrained(
                model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32
            )
            .to(device)
            .eval()
        )

    def _encode(self, texts):
        import torch

        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), self._bs):
                batch = texts[i : i + self._bs]
                enc = self._tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._maxlen,
                    return_tensors="pt",
                ).to(self._device)
                model_out = self._model(**enc)
                # mean pooling over token embeddings, masked by attention
                mask = enc["attention_mask"].unsqueeze(-1).float()
                tok_emb = model_out.last_hidden_state * mask
                summed = tok_emb.sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                emb = summed / counts
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outs.extend(emb.cpu().float().tolist())
        return outs

    def embed_queries(self, texts):
        return self._encode(texts)

    def embed_documents(self, texts):
        return self._encode(texts)


class Qwen3Embedding06B:
    """Qwen3-Embedding-0.6B via HF transformers directly (official recipe).

    Chosen for the legal track (NLLP_SynthData.md Part A): Qwen3-Embedding
    family is the current MTEB-retrieval leader among small embedders and
    0.6B keeps embedding fast on GPU (~0.7GB fp16) — speed and quality per
    the user's explicit requirement.

    Official recipe (verified against the Qwen3-Embedding-0.6B model card,
    2026-08-13):
      - ASYMMETRIC: queries MUST carry the prefix "Instruct: {task}\nQuery: ";
        documents get NO prefix ("No need to add instruction for retrieval
        documents").
      - Pooling = LAST TOKEN, attention-mask-aware (h[arange, mask.sum()-1]),
        then L2-normalize. (Not mean pooling — that would silently degrade
        quality; BGEM3's recipe differs.)
      - 32k context, 1024 dims, right-padding-safe (mask-aware indexing works
        for either padding side).

    `query_instruction` is the task description, configurable per corpus
    (the card: the instruction should describe the task). The CUAD CLI passes
    a legal-contract variant.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 32,
        max_length: int = 2048,
        query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self._bs = batch_size
        self._maxlen = max_length
        self._device = device
        self.query_instruction = query_instruction
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            )
            .to(device)
            .eval()
        )

    def _encode(self, texts, is_query: bool):
        import torch

        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), self._bs):
                batch = texts[i : i + self._bs]
                if is_query:
                    batch = [f"Instruct: {self.query_instruction}\nQuery: {t}" for t in batch]
                enc = self._tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._maxlen,
                    return_tensors="pt",
                ).to(self._device)
                h = self._model(**enc).last_hidden_state
                # last-token pooling: index the last NON-PADDING token per
                # sequence (attention-mask-aware — official recipe)
                seq_lens = enc["attention_mask"].sum(dim=1)
                pooled = h[torch.arange(h.size(0), device=h.device), seq_lens - 1]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outs.extend(pooled.cpu().float().tolist())
        return outs

    def embed_queries(self, texts):
        return self._encode(texts, is_query=True)

    def embed_documents(self, texts):
        return self._encode(texts, is_query=False)


class FakeEmbedder:
    """Deterministic hash-based fake embeddings for CPU testing.

    Not semantically meaningful, but stable & low-dim so graph/cosine code
    exercises real code paths without a GPU.
    """

    def __init__(self, dim: int = 64, model_name: str = "fake/hash-64"):
        import hashlib

        self.model_name = model_name
        self._dim = dim
        self._hash = hashlib.md5

    def _embed(self, text: str) -> list[float]:
        h = self._hash(text.encode()).digest()
        vec = [(b - 128) / 128.0 for b in (h * ((self._dim // 16) + 1))[: self._dim]]
        n = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / n for v in vec]

    def embed_queries(self, texts):
        return [self._embed(t) for t in texts]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]
