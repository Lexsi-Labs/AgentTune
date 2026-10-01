from collections.abc import Callable


class RuleGuardCombinator:
    """
    Wraps an existing reward function (like an LLM judge) and applies deterministic
    rules to clamp or override the score. This prevents the RM from inheriting exploits.
    """

    def __init__(self, base_judge_fn: Callable):
        self.base_judge_fn = base_judge_fn

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        # 1. Get the base scores from the enthusiastic LLM judge
        base_scores = self.base_judge_fn(prompts, completions, **kwargs)

        # 2. Apply deterministic rule guards
        final_scores = []
        for _prompt, comp, score in zip(prompts, completions, base_scores, strict=False):
            # Check 1: Gold-Chunk Semantic Overlap
            # A simple heuristic: if the completion doesn't contain at least one significant
            # entity or keyword from the retrieved chunks (passed via kwargs), clamp it.
            # (In production, this relies on the `EventLog` passing 'retrieved_chunks' in kwargs)
            chunks = kwargs.get("retrieved_chunks", [])
            if chunks and not self._check_grounding_overlap(comp, chunks):
                # Clamp score to 0.2 maximum if ungrounded hallucination is detected
                score = min(score, 0.2)

            final_scores.append(score)

        return final_scores

    def _check_grounding_overlap(self, completion: str, chunks: list[str]) -> bool:
        """
        Simple deterministic check: Does the final answer share significant vocabulary
        with the retrieved chunks? If not, it's likely an ungrounded hallucination.
        """
        if not chunks:
            return True  # Pass if no chunks were provided

        # Very basic overlap logic: check if any chunk word > 5 chars is in the completion
        # In a real system, this would use a more sophisticated N-gram or semantic overlap.
        comp_lower = completion.lower()
        for chunk in chunks:
            words = [w for w in chunk.lower().split() if len(w) > 5]
            for word in words:
                if word in comp_lower:
                    return True
        return False
