"""The model's final reply, for rewards that score an answer."""


def final_reply(completion, i, kwargs) -> str:
    """The content of the last assistant turn of example `i`.

    In agentic GRPO, TRL builds `completions` by decoding the rollout's `completion_ids`,
    which hold every turn run together, raw tool output included. They decode to one
    assistant message such as `<tool_call>...</tool_call>on_hand=42,
    reorder_point=1000There are 42 units.` The rollout's `conversations` keep the final
    reply on its own, so read it from there when present, else from the completion.

    `conversations` counts only when it came from the rollout, which also sends
    `trajectories`. A dataset column of that name (e.g. ShareGPT data standardized to
    role/content) holds the reference turns, and its assistant turn is not the model's
    reply. A `conversations` entry that ends on a user or system turn is the prompt alone
    (the rollout without tools records no reply there), so it is skipped too.
    """
    sources = []
    convos = kwargs.get("conversations") if kwargs.get("trajectories") is not None else None
    if convos is not None and i < len(convos) and isinstance(convos[i], list) and convos[i]:
        last = convos[i][-1]
        if not (isinstance(last, dict) and last.get("role") in ("user", "system")):
            sources.append(convos[i])
    if isinstance(completion, list):
        sources.append(completion)
    for turns in sources:
        for msg in reversed(turns):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content") or "")
    return "" if isinstance(completion, list) else str(completion)
