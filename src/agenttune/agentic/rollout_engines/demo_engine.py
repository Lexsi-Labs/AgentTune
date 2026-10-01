"""DemoRolloutEngine — a deterministic, GPU-free rollout engine.

Lets the full spine path (collect_rollout → train/distill/eval/heal) run without a model,
for demos, examples, and tests. Real training uses the vLLM/transformers/API engines.
"""

from __future__ import annotations

from .base import RolloutEngine


class _DemoTokenizer:
    def apply_chat_template(self, *a, **k):
        raise RuntimeError("demo tokenizer has no chat template")

    def encode(self, text, add_special_tokens=True):
        return []


class DemoRolloutEngine(RolloutEngine):
    """Emits a canned reasoning + answer with fixed logprobs, so a real ``Trajectory``
    (full tier, carrying logprobs) is produced through the real rollout machinery."""

    def _get_tokenizer(self):
        return _DemoTokenizer()

    def generate(self, prompts, tools, gen_cfg):
        return {
            "completions": ["Reasoning through the request, then the final answer."],
            "logprobs": [[-0.10, -0.20, -0.30]],
            "metadata": {"backend": "demo"},
        }
