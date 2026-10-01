"""
T1 — R1-Searcher solve-difficulty prober + contamination gate (Sprint 2 P3).

Per rag_plan_s2.md §2/§6: R1-Searcher's solve-difficulty notion = "how hard to
answer once found." Sample the base model k times on each question WITHOUT the
retrieval corpus (parametric-only); difficulty = failure rate (1 - pass_rate).
Drop already-solved questions (contamination gate: if the model answers correctly
in all k samples without retrieval, it doesn't need training — drop it).

This is the second difficulty axis alongside GRADE's retrieval_difficulty (in
difficulty.py). GRADE = how hard to FIND evidence (retrieval); this = how hard
to ANSWER once found (solve). Keep both per plan §6.

R1-Searcher's probing code does NOT exist in their released repo (the
`pred_anses` field sits unused in stage_1.jsonl). Implemented from their paper
description: sample k times, compute pass rate, filter.

Requires a model + tokenizer (transformers). Designed to run on GPU with
vLLM or HF generate. Pure function — no framework coupling.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def probe_solve_difficulty(
    questions: list[dict[str, str]],
    model_path: str,
    k: int = 4,
    max_new_tokens: int = 128,
    use_vllm: bool = True,
    extraction_fn: callable | None = None,
) -> list[dict[str, Any]]:
    """Probe the base model k times per question WITHOUT retrieval corpus.

    For each question, generates k completions (temperature>0 for variance),
    extracts the answer, checks EM against gold. Returns per-question stats:
      - pass_rate: fraction of k samples that got EM=1
      - solve_difficulty: 1 - pass_rate (0=easy, 1=impossible for base model)
      - already_solved: True if pass_rate == 1.0 (contamination gate — drop)
      - samples: the k extracted answers

    Args:
        questions: list of {"question": str, "answer": str}
        model_path: HF model id or path
        k: number of samples per question (R1-Searcher uses k=5; we default 4)
        max_new_tokens: generation length
        use_vllm: use vLLM for fast batched generation (recommended on GPU)
        extraction_fn: callable(str) -> str to extract answer from generation.
            Defaults to extracting <answer>...</answer> tags.

    Returns:
        list of dicts with question, gold, pass_rate, solve_difficulty,
        already_solved, samples.
    """
    from agenttune.rag.rewards.qa_metrics import exact_match_score, extract_answer_tag

    if extraction_fn is None:
        extraction_fn = extract_answer_tag

    # Build prompts: ask the model to answer directly (no tools, no retrieval).
    # This tests parametric knowledge — can the model answer from memory?
    prompts = []
    for q in questions:
        msg = [
            {
                "role": "system",
                "content": "Answer the question directly. If you don't know, say 'I don't know'. Keep your answer short.",
            },
            {"role": "user", "content": q["question"]},
        ]
        prompts.append(msg)

    # Generate k samples per prompt
    if use_vllm:
        all_completions = _generate_vllm(model_path, prompts, k, max_new_tokens, len(questions))
    else:
        all_completions = _generate_hf(model_path, prompts, k, max_new_tokens, len(questions))

    # Score each question
    results = []
    for q, completions in zip(questions, all_completions, strict=False):
        gold = q["answer"]
        samples = []
        correct = 0
        for comp in completions:
            pred = extraction_fn(comp) if comp else ""
            em = exact_match_score(pred, gold) if pred else 0
            samples.append({"answer": pred[:200], "em": em})
            correct += em
        pass_rate = correct / k if k > 0 else 0.0
        results.append(
            {
                "question": q["question"],
                "gold": gold,
                "pass_rate": pass_rate,
                "solve_difficulty": 1.0 - pass_rate,
                "already_solved": pass_rate >= 1.0,
                "samples": samples,
            }
        )
    return results


def filter_contaminated(
    probe_results: list[dict[str, Any]],
    questions: list[dict[str, str]],
    threshold: float = 1.0,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Contamination gate: drop questions the base model already solves.

    R1-Searcher drops questions the model answers correctly in all k samples
    (they don't need RL — already solved). Returns (kept_questions, kept_results).

    Args:
        probe_results: output from probe_solve_difficulty
        questions: original question list (parallel to probe_results)
        threshold: drop if pass_rate >= threshold (1.0 = all correct)
    """
    kept_q, kept_r = [], []
    for q, r in zip(questions, probe_results, strict=False):
        if r["pass_rate"] < threshold:
            kept_q.append(q)
            kept_r.append(r)
    dropped = len(questions) - len(kept_q)
    logger.info(
        f"[solve_difficulty] contamination gate: dropped {dropped}/{len(questions)} "
        f"already-solved questions (pass_rate >= {threshold})"
    )
    return kept_q, kept_r


def _generate_vllm(model_path, prompts, k, max_new_tokens, n_questions):
    """Generate k samples per prompt using vLLM (fast batched)."""
    from agenttune.utils.optional import require_vllm

    require_vllm()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # Build chat-formatted prompts
    formatted = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    # k samples per prompt
    sp = SamplingParams(
        n=k,
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(formatted, sp)

    # Collect k completions per question
    all_comps = []
    for out in outputs:
        comps = [o.text for o in out.outputs]  # k samples
        all_comps.append(comps)
    return all_comps


def _generate_hf(model_path, prompts, k, max_new_tokens, n_questions):
    """Generate k samples per prompt using HF generate (slower, no vLLM dep)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    all_comps = []
    for prompt_msgs in prompts:
        text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        # Generate k samples with temperature
        comps = []
        for _ in range(k):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            comps.append(gen)
        all_comps.append(comps)
    return all_comps
