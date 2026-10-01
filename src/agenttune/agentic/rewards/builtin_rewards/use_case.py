import os
import re

from agenttune.utils.score_logger import log_score


# ---------------------------------------------------------------------
# 1. Tool usage / grounding reward
# ---------------------------------------------------------------------
def search_grounding_reward(prompts, completions, tool_call_counts=None, **kwargs):
    """
    Did the agent actually use tools before answering?

      0 calls  → 0.0
      1 call   → 0.2
      2 calls  → 0.3
      3+ calls → 0.4
    """
    if tool_call_counts is None:
        log_score(
            "use_case.search_grounding_reward",
            0.0,
            reasons=["tool_call_counts is None -> 0.0 for every completion"],
            meta={"n": len(completions)},
        )
        return [0.0] * len(completions)

    tiers = {0: 0.0, 1: 0.2, 2: 0.3}
    rewards = []
    for i, c in enumerate(tool_call_counts):
        score = tiers.get(min(c, 2), 0.4)
        log_score(
            "use_case.search_grounding_reward",
            score,
            reasons=[f"tool_call_count={c} -> tier score {score}"],
            meta={"index": i},
        )
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------
# 2. Evidence / citation reward
# ---------------------------------------------------------------------
def message_id_citation_reward(prompts, completions, **kwargs):
    """
    Reward if the model cites evidence (email or message_id).

    +0.2 if any of:
      - email address
      - message_id mention
    """
    patterns = [
        r"<[^>]+@[^>]+>",  # <email@domain>
        r"\b[\w.+\-]+@[\w.\-]+\b",  # email
        r"\bmessage[_\-]?id\b",  # message_id keyword
    ]
    combined = "|".join(patterns)

    rewards = []
    for i, c in enumerate(completions):
        text = str(c)
        hit = re.search(combined, text, re.IGNORECASE)
        score = 0.2 if hit else 0.0
        reason = (
            f"citation pattern matched ({hit.group(0)!r}) -> 0.2"
            if hit
            else "no email/message_id citation found -> 0.0"
        )
        log_score("use_case.message_id_citation_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


def format_reward(prompts, completions, **kwargs):
    """
    +0.1 if the completion wraps the final answer in <answer> tags.
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else str(completion)
        has_tags = bool(re.search(r"<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL))
        score = 0.1 if has_tags else 0.0
        reason = (
            "<answer>...</answer> tags present -> 0.1"
            if has_tags
            else "no <answer> tags found -> 0.0"
        )
        log_score("use_case.format_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------
# 3. Answer format reward
# ---------------------------------------------------------------------
def answer_format_reward(prompts, completions, **kwargs):
    """
    Reward for proper formatting.

    +0.1 if:
      - contains <answer>...</answer>
      - content is non-trivial (>=10 chars)
    """
    rewards = []
    for i, c in enumerate(completions):
        text = str(c)
        match = re.search(r"<answer>(.{10,}?)</answer>", text, re.IGNORECASE | re.DOTALL)
        score = 0.1 if match else 0.0
        reason = (
            "<answer> tag with >=10 chars of content -> 0.1"
            if match
            else "no <answer> tag with >=10 chars of content -> 0.0"
        )
        log_score("use_case.answer_format_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------
# 4. (Optional but recommended) Answer correctness reward
# ---------------------------------------------------------------------
def answer_correctness_reward(prompts, completions, answer=None, **kwargs):
    """
    Compare predicted answer with gold answer.

    Scoring:
      +1.0 exact match
      +0.5 partial overlap
       0.0 no match
    """
    rewards = []

    for i, (completion, gold) in enumerate(
        zip(completions, answer or [""] * len(completions), strict=False)
    ):
        text = str(completion)
        gold = str(gold)

        # extract <answer> if present
        match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
        pred = match.group(1).strip().lower() if match else text.lower()

        gold_clean = gold.strip().lower()

        if pred == gold_clean:
            score = 1.0
            reason = f"prediction exactly matches gold ({gold_clean!r}) -> 1.0"
        elif gold_clean in pred or pred in gold_clean:
            score = 0.5
            reason = "prediction and gold partially overlap (substring match) -> 0.5"
        else:
            score = 0.0
            reason = f"prediction {pred!r} does not match gold {gold_clean!r} -> 0.0"

        log_score("use_case.answer_correctness_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)

    return rewards


def placeholder_coverage_reward(prompts, completions, answer=None, **kwargs):
    """
    Fraction of gold numerical values present in prediction.
    +1.0  >= 80% correct
    +0.5  >= 50% correct
    +0.2  >= 20% correct
     0.0  < 20% correct
    """

    def _count_filled(pred, gold):
        gold_numbers = re.findall(r"[\$]?[\-]?\d+[\.,]?\d*[BMK%]?", gold)
        if not gold_numbers:
            return 0.0
        hits = sum(1 for n in gold_numbers if n.lower().rstrip("%bm") in pred.lower())
        return hits / len(gold_numbers)

    rewards = []
    for i, (completion, gold) in enumerate(
        zip(completions, answer or [""] * len(completions), strict=False)
    ):
        text = str(completion)
        match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
        pred = match.group(1).strip() if match else text.strip()
        coverage = _count_filled(pred, str(gold))

        if coverage >= 0.8:
            score = 1.0
        elif coverage >= 0.5:
            score = 0.5
        elif coverage >= 0.2:
            score = 0.2
        else:
            score = 0.0

        log_score(
            "use_case.placeholder_coverage_reward",
            score,
            reasons=[f"{coverage:.0%} of gold numeric values found in prediction -> {score}"],
            components={"coverage": coverage},
            meta={"index": i},
        )
        rewards.append(score)
    return rewards


def template_structure_reward(prompts, completions, answer=None, **kwargs):
    """
    +0.3 if prediction preserves section headers from the template.
    """
    rewards = []
    for i, (completion, gold) in enumerate(
        zip(completions, answer or [""] * len(completions), strict=False)
    ):
        text = str(completion)
        match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
        pred = match.group(1).strip() if match else text.strip()
        headers = re.findall(r"\*\*([^*]{10,})\*\*", str(gold))[:5]

        if not headers:
            score = 0.1
            log_score(
                "use_case.template_structure_reward",
                score,
                reasons=["gold has no section headers to check -> default 0.1"],
                meta={"index": i},
            )
            rewards.append(score)
            continue

        hits = sum(1 for h in headers if h[:20].lower() in pred.lower())
        score = 0.3 * (hits / len(headers))
        log_score(
            "use_case.template_structure_reward",
            score,
            reasons=[
                f"{hits}/{len(headers)} gold section headers preserved -> 0.3 * {hits}/{len(headers)} = {score}"
            ],
            components={"headers_matched": hits, "headers_total": len(headers)},
            meta={"index": i},
        )
        rewards.append(score)
    return rewards


def computation_reward(prompts, completions, tool_call_counts=None, **kwargs):
    """
    +0.2 if run_python called once, +0.3 if called 2+ times.
    """
    if tool_call_counts is None:
        log_score(
            "use_case.computation_reward",
            0.0,
            reasons=["tool_call_counts is None -> 0.0 for every completion"],
            meta={"n": len(completions)},
        )
        return [0.0] * len(completions)
    rewards = []
    for i, c in enumerate(tool_call_counts):
        score = 0.3 if c >= 2 else (0.2 if c == 1 else 0.0)
        reason = (
            f"run_python called {c} times (>=2) -> 0.3"
            if c >= 2
            else ("run_python called once -> 0.2" if c == 1 else "run_python never called -> 0.0")
        )
        log_score("use_case.computation_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


def numerical_match_reward(prompts, completions, answer=None, **kwargs):
    """
    Exact or near-exact match on numerical values extracted from prediction.

    +1.0  exact numeric match
    +0.5  within 5% tolerance
     0.0  no match
    """
    import re

    rewards = []
    for i, (completion, gold) in enumerate(
        zip(completions, answer or [""] * len(completions), strict=False)
    ):
        text = str(completion)
        match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
        pred = match.group(1).strip() if match else text.strip()

        gold_nums = re.findall(r"-?\d+\.?\d*", str(gold))
        pred_nums = re.findall(r"-?\d+\.?\d*", pred)

        if not gold_nums:
            score = 0.0
            reason = "gold contains no numeric value to compare against -> 0.0"
        else:
            gold_val = float(gold_nums[0])
            pred_val = float(pred_nums[0]) if pred_nums else None

            if pred_val is None:
                score = 0.0
                reason = "prediction contains no numeric value -> 0.0"
            elif pred_val == gold_val:
                score = 1.0
                reason = f"predicted value {pred_val} exactly equals gold {gold_val} -> 1.0"
            elif gold_val != 0 and abs(pred_val - gold_val) / abs(gold_val) <= 0.05:
                score = 0.5
                reason = f"predicted value {pred_val} within 5% of gold {gold_val} -> 0.5"
            else:
                score = 0.0
                reason = (
                    f"predicted value {pred_val} not within tolerance of gold {gold_val} -> 0.0"
                )

        log_score("use_case.numerical_match_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)

    return rewards


def exploration_reward(prompts, completions, tool_call_counts=None, **kwargs):
    """
    Rewards breadth of tool usage — encourages the agent to explore
    multiple tools rather than relying on just one.

    0 calls  → 0.0
    1 call   → 0.1
    2 calls  → 0.2
    3+ calls → 0.3
    """
    if tool_call_counts is None:
        log_score(
            "use_case.exploration_reward",
            0.0,
            reasons=["tool_call_counts is None -> 0.0 for every completion"],
            meta={"n": len(completions)},
        )
        return [0.0] * len(completions)

    tiers = {0: 0.0, 1: 0.1, 2: 0.2}
    rewards = []
    for i, c in enumerate(tool_call_counts):
        score = tiers.get(min(c, 2), 0.3)
        log_score(
            "use_case.exploration_reward",
            score,
            reasons=[f"tool_call_count={c} -> tier score {score}"],
            meta={"index": i},
        )
        rewards.append(score)
    return rewards


def tool_chain_reward(prompts, completions, **kwargs):
    """
    +0.1 for each tool in the expected file ingestion chain
    that appears in the completion:
        list_dir, read_file, run_python, write_file → max 0.4
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else str(completion)
        expected = ["list_dir", "read_file", "run_python", "write_file"]
        hits = [t for t in expected if t in text]
        score = 0.1 * len(hits)
        log_score(
            "use_case.tool_chain_reward",
            score,
            reasons=[
                f"found {hits or 'none'} of expected chain {expected} -> 0.1 * {len(hits)} = {score}"
            ],
            meta={"index": i},
        )
        rewards.append(score)
    return rewards


def summary_written_reward(prompts, completions, sample_dir=None, **kwargs):
    """
    +0.2 if the agent wrote a new file to the sample directory.
    Checks for any file not in the original set of 4.
    """
    rewards = []
    dirs = sample_dir if sample_dir is not None else [None] * len(completions)
    for i, (_completion, sdir) in enumerate(zip(completions, dirs, strict=False)):
        if sdir is None or not os.path.isdir(sdir):
            log_score(
                "use_case.summary_written_reward",
                0.0,
                reasons=[f"sample_dir {sdir!r} missing or not a directory -> 0.0"],
                meta={"index": i},
            )
            rewards.append(0.0)
            continue
        original = {"revenue.csv", "expenses.csv", "notes.txt", "server.log"}
        new_files = set(os.listdir(sdir)) - original
        score = 0.2 if new_files else 0.0
        reason = (
            f"new file(s) written: {sorted(new_files)} -> 0.2"
            if new_files
            else "no new files beyond the original 4 -> 0.0"
        )
        log_score("use_case.summary_written_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards
