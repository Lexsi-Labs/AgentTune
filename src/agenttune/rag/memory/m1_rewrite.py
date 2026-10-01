"""
M1 — MEM1-style rewritten running state (Sprint 2 flagship).

Mechanism (from MEM1/Mem1/inference/data_pipelines.py:109-181, see
rag_plan_s2.md §2): after each turn, replace the agent's accumulated
conversation history with a compact rewritten state, so context size stays
roughly constant no matter how many searches it runs. The model itself writes
the running state (a think block); RL trains it to write one that preserves
answer-relevant info. Reward = outcome EM/F1 at the last token.

This module is RAG-scoped and additive. It provides:
  - extract_internal_state(response)  — regex-pull the think block (MEM1's
    extract_internal_state, ~20 lines, lifted).
  - compress_tool_output(tool_text)    — summarize long retrieved docs into
    evidence notes with source IDs preserved (so groundedness checks still
    work). Cheap heuristic compressor (no extra model call) — keeps chunk_ids
    + first sentence of each chunk, capped.
  - build_rewritten_state(...)         — assemble the compact state message.
  - mem1_post_step_hook                — the hook wired into create_rollout_fn.
    Receives (step, conversation, ...) and returns a (possibly rewritten)
    conversation. Requires the minimal core edit to rollout_factory.py that
    passes `conversation` into the hook and honors its return value.

The prompt adaptation ({running_summary, open_questions, evidence_notes}) lives
in data/hotpotqa.py's M1 system prompt (added there).

NO core edits in this file — it's pure RAG-package code. The one core edit it
needs (threading conversation through post_step_hook) is in rollout_factory.py
and logged in the core-changes register (Sprint2_readme §4).
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# State extraction (MEM1 extract_internal_state, lifted)
# ─────────────────────────────────────────────────────────────────────────────
# MEM1 uses tag="think". Qwen3/3.5 thinking mode emits <think>...</think>.
# We support both the raw <think> tag and a structured <state>...</state> tag
# (the M1 prompt asks the model to emit a structured state; if it only emits
# a free-form think block, we use that as the running summary).

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_STATE_RE = re.compile(r"<state>(.*?)</state>", re.DOTALL | re.IGNORECASE)


def extract_internal_state(response: str, tag: str = "think") -> str | None:
    """Pull the model's internal-state block out of its response.

    Mirrors MEM1's extract_internal_state (data_pipelines.py:57). Returns the
    inner text stripped, or None if no block found. Prefers a structured
    <state> block (the M1 prompt's requested format) and falls back to
    <think> (Qwen's native thinking tag).
    """
    if not isinstance(response, str):
        response = str(response)
    if tag == "state":
        m = _STATE_RE.search(response)
    else:
        # Prefer <state> if present (structured M1 format), else <think>.
        m = _STATE_RE.search(response) or _THINK_RE.search(response)
    if m is None:
        return None
    return m.group(1).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool-output compressor
# ─────────────────────────────────────────────────────────────────────────────
# MEM1 truncates retrieved docs to 1000 tokens. We do better for RAG: keep the
# chunk_id + doc_id + score metadata (so groundedness/citation checks still
# work) and the first sentence of each chunk's content, capped at N chunks.
# This preserves source IDs (the plan §2 requirement) while shrinking context.

_CHUNK_HEADER_RE = re.compile(r"\[chunk_id=(\S+)\s+doc_id=(\S+)\s+score=([\d.]+)\]")


def compress_tool_output(
    tool_text: str, max_chunks: int = 5, max_chars_per_chunk: int = 300
) -> str:
    """Compress a search_corpus result into evidence notes with source IDs.

    Input format (from SearchCorpusTool.execute):
        [chunk_id=... doc_id=... score=...]
        <chunk text>

    Output: one line per chunk, "evidence[chunk_id=.. doc_id=..]: <first sentence>".
    Keeps source IDs (for groundedness) + the first sentence (the fact). Caps
    at max_chunks to bound context. read_document output (no chunk headers)
    is passed through with a length cap.
    """
    if not isinstance(tool_text, str):
        tool_text = str(tool_text)

    # Split on the chunk-header marker — each chunk is header + body.
    parts = _CHUNK_HEADER_RE.split(tool_text)
    # parts[0] is text before the first header (usually "" or "No results found.")
    preamble = parts[0].strip() if parts else ""
    if "No results found" in preamble:
        return "No results found."

    chunks = []
    # parts after preamble come in groups of 3: (chunk_id, doc_id, score, body...)
    # because re.split with 3 capture groups yields [preamble, id1, doc1, score1, body1, id2, ...]
    i = 1
    while i < len(parts) - 2 and len(chunks) < max_chunks:
        chunk_id, doc_id, score = parts[i], parts[i + 1], parts[i + 2]
        body = parts[i + 3] if i + 3 < len(parts) else ""
        i += 4
        body = body.strip()
        # First sentence (up to first period or newline), capped.
        first_sent = re.split(r"(?<=[.!?])\s|\n", body, maxsplit=1)[0].strip()
        if len(first_sent) > max_chars_per_chunk:
            first_sent = first_sent[:max_chars_per_chunk].rsplit(" ", 1)[0] + "…"
        chunks.append(f"evidence[chunk_id={chunk_id} doc_id={doc_id} score={score}]: {first_sent}")

    if not chunks:
        # No chunk headers — probably a read_document result or an error string.
        # Pass through with a length cap so we don't blow context on a full doc.
        capped = tool_text.strip()
        if len(capped) > max_chars_per_chunk * max_chunks:
            capped = capped[: max_chars_per_chunk * max_chunks].rsplit(" ", 1)[0] + "…"
        return capped

    return "\n".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Rewritten-state assembly
# ─────────────────────────────────────────────────────────────────────────────


def build_rewritten_state(
    running_state: str,
    latest_tool_results: list[str],
    question: str,
) -> list[dict]:
    """Assemble the compact conversation the model sees next turn.

    Returns a fresh conversation list:
      [ {system}, {user: question}, {assistant: <state>running_state</state>},
        {tool: compressed latest results} ]

    This is MEM1's cur_obs="" wipe: everything before this turn is gone; only
    the running state (the model's own think/state block) + this turn's
    compressed tool output survive. Context stays constant per turn.
    """
    # The running state is injected as an assistant message carrying a <state>
    # block — the model wrote it last turn, so it's "its own" memory. The
    # compressed tool results go in as a tool-role message so the chat template
    # renders them as observations (and they get env_mask=0, excluded from loss).
    (
        "\n\n".join(compress_tool_output(r) for r in latest_tool_results)
        if latest_tool_results
        else ""
    )
    msgs: list[dict] = []
    # NOTE: system prompt is prepended by _build_conversation at rollout time;
    # we only return user + state + tool here. But to keep this hook
    # self-contained when it rewrites an existing conversation (which already
    # has system+user at the front), we return the state + tool messages and
    # let the caller splice them after the original system+user.
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# The post_step_hook (wired into create_rollout_fn)
# ─────────────────────────────────────────────────────────────────────────────
# Signature after the minimal core edit:
#   post_step_hook(step, conversation=None, raw_outputs=None) -> Optional[list]
# If it returns a list, _execute_trajectory replaces `conversation` with it.
# If it returns None, conversation is left untouched (backward-compatible with
# the old single-arg hooks — TraceLogger etc. still work).


def mem1_post_step_hook(step, conversation=None, raw_outputs=None, **kwargs):
    """MEM1 rewrite: after a tool-result step, wipe history and keep only the
    model's latest internal-state block + compressed tool output.

    Called after each tool-call step (rollout_factory.py:1167). `step` carries
    the tool results in step.metadata['tool_results'] and the model's raw
    generation in step.thought. `conversation` is the full chat history so far.

    Returns a rewritten conversation: [system, user, {assistant: <state>...},
    {tool: compressed results}] — or None to leave conversation untouched (e.g.
    on the terminal step, or if no state block was emitted yet).
    """
    if conversation is None:
        return None

    # Only rewrite after a tool-result step (not the terminal answer step).
    meta = getattr(step, "metadata", None) or {}
    is_terminal = meta.get("is_terminal", False)
    tool_results = meta.get("tool_results", [])
    if is_terminal or not tool_results:
        return None

    # The model's latest generation (carries the think/state block).
    thought = getattr(step, "thought", None) or ""
    state = extract_internal_state(thought, tag="state")
    if state is None:
        # No state block emitted yet (early turns, or model not yet trained to
        # emit one). Don't rewrite — let the full history flow until the model
        # starts emitting state. This is the safe zero-shot fallback.
        return None

    # Compress this turn's tool outputs into evidence notes.
    compressed_results = []
    for _name, result in tool_results:
        compressed_results.append(compress_tool_output(str(result)))
    compressed = "\n\n".join(compressed_results)

    # Find the original system + user messages (front of conversation).
    system_msg = None
    user_msg = None
    for m in conversation:
        if m.get("role") == "system" and system_msg is None:
            system_msg = m
        elif m.get("role") == "user" and user_msg is None:
            user_msg = m
        if system_msg and user_msg:
            break

    if user_msg is None:
        return None  # can't rebuild without the question

    # The rewritten conversation: system + user + state(as assistant) + tool.
    new_conv: list[dict] = []
    if system_msg is not None:
        new_conv.append(dict(system_msg))
    new_conv.append(dict(user_msg))
    new_conv.append(
        {
            "role": "assistant",
            "content": f"<state>\n{state}\n</state>",
        }
    )
    new_conv.append(
        {
            "role": "tool",
            "name": tool_results[0][0] if tool_results else "search_corpus",
            "content": compressed,
            "tool_call_id": "mem1_rewrite",
        }
    )
    return new_conv


# ─────────────────────────────────────────────────────────────────────────────
# Recent-k truncation baseline (for the zero-shot comparison)
# ─────────────────────────────────────────────────────────────────────────────
# The plan §10 / m1_channel_update.md calls for a zero-shot comparison across:
#   (a) plain full-history agent, (b) recent-k truncation, (c) M1 rewrite.
# This hook implements (b): keep only the last k tool-result turns. Same
# signature as mem1_post_step_hook so it's a drop-in for the same eval harness.


def recent_k_post_step_hook(step, conversation=None, raw_outputs=None, k: int = 2, **kwargs):
    """Truncate conversation to the last k tool-result turns (plus system+user).

    A simple, non-learned baseline for the M1 comparison: keeps recent context,
    drops old. Same hook signature as mem1_post_step_hook. ``k`` defaults to 2
    (typical HotpotQA hop count); override by passing ``k=`` to the hook, or
    wrap in a lambda/functools.partial for the eval harness.
    """
    if conversation is None:
        return None
    meta = getattr(step, "metadata", None) or {}
    if meta.get("is_terminal", False) or not meta.get("tool_results"):
        return None

    system_msg = user_msg = None
    for m in conversation:
        if m.get("role") == "system" and system_msg is None:
            system_msg = m
        elif m.get("role") == "user" and user_msg is None:
            user_msg = m
        if system_msg and user_msg:
            break
    if user_msg is None:
        return None

    # Collect the last k (assistant, tool) pairs from the tail of the
    # conversation. We walk backward; each tool message is paired with the
    # nearest preceding assistant message.
    collected: list[dict] = []
    pairs = 0
    i = len(conversation) - 1
    while i >= 0 and pairs < k:
        m = conversation[i]
        if m.get("role") == "tool":
            collected.append(m)
            # find the nearest preceding assistant message
            j = i - 1
            while j >= 0 and conversation[j].get("role") != "assistant":
                j -= 1
            if j >= 0:
                collected.append(conversation[j])
            pairs += 1
        i -= 1
    collected.reverse()

    new_conv: list[dict] = []
    if system_msg is not None:
        new_conv.append(dict(system_msg))
    new_conv.append(dict(user_msg))
    new_conv.extend(dict(m) for m in collected)
    return new_conv
