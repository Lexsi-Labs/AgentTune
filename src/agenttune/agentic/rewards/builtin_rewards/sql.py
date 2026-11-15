"""
Built-in reward functions for agentic GRPO training.
Add new functions here — they are auto-discovered via REWARD_REGISTRY below.
"""

import ast
import itertools
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from agenttune.agentic.rollout_engines.tool_call_parse import _THINK_BLOCK_RE
from agenttune.utils.score_logger import log_score

logger = logging.getLogger(__name__)

# A reasoning block cut off by the token limit has no closing tag; drop it to the end.
_OPEN_THINK_RE = re.compile(r"(?:<think>|<\|START_THINKING\|>).*\Z", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove reasoning blocks (Qwen ``<think>``, Cohere ``<|START_THINKING|>``,
    Harmony analysis) so only the answer itself is scored. Text without them is
    returned unchanged."""
    return _OPEN_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", text))


_SQL_STRING_RE = re.compile(r"'(?:[^']|'')*'")
_SQL_WORD_RE = re.compile(r"[a-z_][a-z0-9_]*")
# Where the WHERE clause ends; a subquery's own clauses count as part of it.
_WHERE_END_RE = re.compile(r"\b(?:group\s+by|order\s+by|having|limit|union|intersect|except)\b|;")
_WHERE_KEYWORDS = frozenset(
    "and or not null is in like ilike glob between true false exists select distinct "
    "case when then else end escape collate".split()
)


def _has_real_where(sql: str) -> bool:
    """True when the WHERE clause names a column once string literals, numbers and SQL
    keywords are removed. `where 1 = 1` and `where 'a' = 'a'` filter nothing."""
    m = re.search(r"\bwhere\b(.*)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return False
    clause = _SQL_STRING_RE.sub(" ", m.group(1).lower())
    end = _WHERE_END_RE.search(clause)
    if end:
        clause = clause[: end.start()]
    # ponytail: a function call (`where abs(1) = 1`) still counts as a column name.
    return any(w not in _WHERE_KEYWORDS for w in _SQL_WORD_RE.findall(clause))


def _tool_output(content):
    """A tool message's content as a Python value. Rollouts store `str(result)`, so
    `"[(7, 'Ana')]"` is parsed back to rows and `"{'error': ...}"` to a dict.
    Returns None for text that is not a literal list, tuple or dict."""
    if isinstance(content, str):
        try:
            content = ast.literal_eval(content.strip())
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None
    if isinstance(content, tuple):
        return list(content)
    return content if isinstance(content, list | dict) else None


def _turns(completion, i, kwargs):
    """The turns of example `i`. Agentic rollouts pass the assistant text as `completions`
    and the full turn list (tool calls and tool results) as `conversations`; use that
    when present, else the completion itself."""
    convos = kwargs.get("conversations")
    if convos is not None and i < len(convos) and isinstance(convos[i], list):
        return convos[i]
    return completion if isinstance(completion, list) else []


def _sql_of(call) -> str:
    """The `sql_command` of a tool call; rollouts store `arguments` as a JSON string."""
    args = (call.get("function") or {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return ""
    return str(args.get("sql_command", "")) if isinstance(args, dict) else ""


# ── Individual reward functions ───────────────────────────────────────────────


def correctness_reward(completions, answer=None, **kwargs):
    """Reward *yes*/*no* correctness enclosed in stars."""
    rewards = []
    for i, (completion, ans) in enumerate(zip(completions, answer or [], strict=False)):
        raw = (
            completion[-1]["content"].lower()
            if isinstance(completion, list)
            else str(completion).lower()
        )
        match = re.search(r"\*(yes|no)\*", raw)
        guess = match.group(1) if match else None
        if guess is None:
            score = -0.5
            reason = "no *yes*/*no* guess found in completion -> -0.5"
        elif guess == ans.lower():
            score = 0.6
            reason = f"guess {guess!r} matches gold {ans!r} -> 0.6"
        else:
            score = -1.0
            reason = f"guess {guess!r} does not match gold {ans!r} -> -1.0"
        log_score("sql.correctness_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


def structure_reward(completions, **kwargs):
    """Reward proper tool-call → response → content structure."""
    rewards = []
    for i, completion in enumerate(completions):
        has_call = has_response = has_other = False
        for turn in _turns(completion, i, kwargs):
            role = turn.get("role")
            if role == "assistant" and turn.get("tool_calls"):
                has_call = True
            elif role == "tool":
                has_response = True
            elif role == "assistant" and str(turn.get("content") or "").strip() not in [
                "",
                "<think>",
            ]:
                has_other = True
        if has_call and has_response:
            score = 0.1 if has_other else 0.05
            reason = (
                "tool call + tool response + other content -> 0.1"
                if has_other
                else "tool call + tool response, no other content -> 0.05"
            )
        elif has_call:
            score = -0.15
            reason = "tool call made but no tool response -> -0.15"
        else:
            score = 0.0
            reason = "no tool call made -> 0.0"
        log_score("sql.structure_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


def query_reward(completions, answer=None, **kwargs):
    """Reward effective SQL query strategy."""
    rewards = []
    for i, (completion, ans) in enumerate(zip(completions, answer or [], strict=False)):
        reward = 0.0
        reasons = []
        sql_queries, tool_results = [], []
        for turn in _turns(completion, i, kwargs):
            if turn.get("tool_calls"):
                for call in turn["tool_calls"]:
                    sql_queries.append(_sql_of(call).lower())
            if turn.get("role") == "tool" and turn.get("content"):
                tool_results.append(turn["content"])
        if len(sql_queries) > 3:
            reward -= 1.5
            reasons.append(f"{len(sql_queries)} SQL queries (>3) -> -1.5")
        where_count = 0
        for q in sql_queries:
            if "limit 1" in q:
                reward -= 1.0
                reasons.append(f"query {q!r} uses 'limit 1' -> -1.0")
            if not _has_real_where(q):
                reward -= 0.5
                reasons.append(f"query {q!r} has no WHERE clause naming a column -> -0.5")
            else:
                where_count += 1
        if where_count:
            bonus = min(where_count, 3) * 0.4
            reward += bonus
            reasons.append(f"{where_count} queries with WHERE (capped at 3) -> +{bonus}")
        combined_results, error_detected = [], False
        for res in map(_tool_output, tool_results):
            if isinstance(res, dict) and "error" in res:
                error_detected = True
            elif isinstance(res, list):
                combined_results.extend(res)
        if error_detected:
            reward -= 2.0
            reasons.append("tool result contained an error -> -2.0")
        elif not sql_queries:
            reward -= 1.5
            reasons.append("no SQL queries issued at all -> -1.5")
        else:
            has_hits = len(combined_results) > 0
            correct_answer = (ans or "").lower()
            if (has_hits and correct_answer == "yes") or (not has_hits and correct_answer == "no"):
                reward += 2.0
                reasons.append(
                    f"has_hits={has_hits} agrees with gold answer {correct_answer!r} -> +2.0"
                )
            else:
                reward -= 1.5
                reasons.append(
                    f"has_hits={has_hits} disagrees with gold answer {correct_answer!r} -> -1.5"
                )
        log_score("sql.query_reward", reward, reasons=reasons, meta={"index": i})
        rewards.append(reward)
    return rewards


def _canon(value):
    """A hashable form of one cell: lists become tuples, a dict its sorted items."""
    if isinstance(value, dict):
        items = ((str(k), _canon(v)) for k, v in value.items())
        return ("<dict>", *sorted(items, key=repr))
    if isinstance(value, list | tuple):
        return tuple(map(_canon, value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _row_key(row):
    """A row as a tuple of hashable cells. A dict row (a tool that returns records)
    is its values in column order; gold rows carry no column names."""
    if isinstance(row, dict):
        row = list(row.values())
    return tuple(map(_canon, row)) if isinstance(row, list | tuple) else (_canon(row),)


def _close(a, b):
    """Equal, with float tolerance (abs 1e-6, rel 1e-9) when a float is involved;
    two ints compare exactly. Tuples compare item by item."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(map(_close, a, b))
    numbers = isinstance(a, int | float) and isinstance(b, int | float)
    if numbers and (isinstance(a, float) or isinstance(b, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
    return a == b


def _sort_key(value):
    """A total order over mixed cells (None, numbers, text, bytes, tuples)."""
    if isinstance(value, tuple):
        return (4, tuple(map(_sort_key, value)))
    if value is None:
        return (0, 0)
    if isinstance(value, int | float):
        return (1, value)
    if isinstance(value, str):
        return (2, value)
    if isinstance(value, bytes):
        return (3, value)
    return (5, repr(value))


def _same_rows(pred, gold, ordered):
    """Rows equal in order (`ordered`) or as a multiset, floats within tolerance."""
    if len(pred) != len(gold):
        return False
    if not ordered:
        if Counter(pred) == Counter(gold):
            return True
        # ponytail: pairs rows by sorting; two rows that differ only by float noise in
        # opposite directions on an earlier column can sort apart (a false reject).
        pred, gold = sorted(pred, key=_sort_key), sorted(gold, key=_sort_key)
    return all(map(_close, pred, gold))


_MAX_PERMUTED_COLUMNS = 6


def _rows_match(pred, gold, ordered=False):
    """Rows equal (as a multiset unless `ordered`), allowing one column permutation
    applied to every row (`select age, name` answers `select name, age`). Permutations
    are tried only for 2 to _MAX_PERMUTED_COLUMNS columns of equal width."""
    pred_rows, gold_rows = list(map(_row_key, pred)), list(map(_row_key, gold))
    if _same_rows(pred_rows, gold_rows, ordered):
        return True
    widths = {len(r) for r in pred_rows} | {len(r) for r in gold_rows}
    if len(widths) != 1 or not 1 < next(iter(widths)) <= _MAX_PERMUTED_COLUMNS:
        return False
    n = next(iter(widths))
    # Pair a predicted column with a gold one only when they hold the same values;
    # ponytail: columns with identical values still try every pairing (720 at most).
    cols_p = [[(r[j],) for r in pred_rows] for j in range(n)]
    cols_g = [[(r[k],) for r in gold_rows] for k in range(n)]
    same = [[_same_rows(cols_p[j], cols_g[k], ordered) for j in range(n)] for k in range(n)]
    return any(
        _same_rows([tuple(r[j] for j in perm) for r in pred_rows], gold_rows, ordered)
        for perm in itertools.permutations(range(n))
        if all(same[k][perm[k]] for k in range(n))
    )


def _results_with_sql(turns):
    """(sql, content) per "tool" turn, in order. A result is paired with the first
    unused call of the same tool name in the assistant turn before it (any call when
    the turn has no name); sql is None when there is no such call."""
    out, calls = [], []
    for turn in turns:
        if turn.get("role") == "assistant":
            calls = list(turn.get("tool_calls") or [])
        elif turn.get("role") == "tool":
            name = turn.get("name")
            call = next(
                (c for c in calls if not name or (c.get("function") or {}).get("name") == name),
                None,
            )
            if call is not None:
                calls.remove(call)
            out.append((_sql_of(call) or None if call else None, turn.get("content")))
    return out


def _is_error(content) -> bool:
    """A failed call: an `{'error': ...}` result or text starting with "Error"."""
    out = _tool_output(content)
    if isinstance(out, dict):
        return "error" in out
    return isinstance(content, str) and content.lstrip().lower().startswith("error")


def _reads_no_table(sql: str):
    """True when `sql` compiles on an empty SQLite database, so it reads none of the
    user's tables: literals, VALUES, CTEs of literals, sqlite_master, pragma functions.
    False when it names a table an empty database lacks. None when SQLite cannot tell
    (another SQL dialect, a syntax error, several statements)."""
    con = sqlite3.connect(":memory:")
    try:
        con.execute("EXPLAIN " + sql)  # compiles, never runs
        return True
    except sqlite3.Error as e:
        return False if "no such table" in str(e) else None
    except Exception:
        return None
    finally:
        con.close()


# Strings, quoted identifiers and comments, earliest first, so none hides inside another.
_SQL_NOISE_RE = re.compile(
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?(?:\*/|\Z)", re.DOTALL
)


def _has_top_level_order_by(sql: str) -> bool:
    """True when the query itself sorts its rows: ORDER BY outside any parentheses
    (not in a subquery or `over (...)`), strings or comments."""
    depth, top = 0, []
    for ch in _SQL_NOISE_RE.sub(" ", sql):
        depth += (ch == "(") - (ch == ")")
        top.append(ch if depth == 0 and ch != ")" else " ")
    return re.search(r"\border\s+by\b", "".join(top), re.IGNORECASE) is not None


_QUERY_SECONDS = 5.0
_MAX_VALUE_BYTES = 1_000_000  # one string or blob
_MAX_GOLD_ROWS = 100_000
_READ_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class _TooManyRows(Exception):
    pass


def _run_read_only(path, sql: str, max_rows: int) -> list:
    """Rows of `sql` on the SQLite file at `path`. The query is model-written: the file
    is opened read-only, only reads are allowed, no value may exceed _MAX_VALUE_BYTES,
    the query stops after _QUERY_SECONDS, and more than `max_rows` rows raise
    _TooManyRows without fetching the rest."""
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_VALUE_BYTES)
        con.set_authorizer(
            lambda action, *_: sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY
        )
        deadline = time.monotonic() + _QUERY_SECONDS
        con.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
        rows = con.execute(sql).fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise _TooManyRows
        return rows
    finally:
        con.close()


def _match(turns, gold, gold_sql=None, db_paths=()):
    """(score, reason) for one example's turns against its gold rows."""
    if db_paths and not gold_sql:
        raise ValueError("result_match_reward: db_paths needs gold_sql for the same example")
    results = _results_with_sql(turns)
    if not results:
        return -1.0, "no tool result -> -1.0"
    ran = [(sql, content) for sql, content in results if not _is_error(content)]
    if not ran:
        return -1.0, "every tool call failed -> -1.0"
    sql, content = ran[-1]
    ordered = bool(gold_sql) and _has_top_level_order_by(gold_sql)
    rows = _tool_output(content)
    if not isinstance(rows, list) or not _rows_match(rows, gold, ordered):
        return -1.0, "last result that ran differs from the gold rows -> -1.0"
    if sql and _reads_no_table(sql):
        return -1.0, "the query reads no table; its rows are written in the query -> -1.0"
    for path in db_paths:
        if not sql:
            return -1.0, "no query to re-run on the comparison databases -> -1.0"
        try:
            want = _run_read_only(path, gold_sql, _MAX_GOLD_ROWS)
        except (sqlite3.Error, _TooManyRows) as e:
            raise ValueError(f"result_match_reward: gold_sql failed on {path}: {e!r}") from e
        try:
            # One row more than gold is enough to tell a mismatch.
            got = _run_read_only(path, sql, len(want))
        except _TooManyRows:
            return -1.0, f"the query returns more rows than gold_sql on {path} -> -1.0"
        except sqlite3.Error as e:
            return -1.0, f"the query fails on comparison database {path} ({e}) -> -1.0"
        if not _rows_match(got, want, ordered):
            return -1.0, f"the query differs from gold_sql on comparison database {path} -> -1.0"
    return 2.0, "last result that ran equals the gold rows -> +2.0"


def result_match_reward(completions, gold_rows=None, gold_sql=None, db_paths=None, **kwargs):
    """Opt-in: +2.0 when the rows of the last query that ran equal the gold query's
    rows (row order ignored, duplicates counted, column order ignored, floats within
    abs 1e-6 / rel 1e-9; a dict row compares by its values), else -1.0.

    "Last query that ran" is the last tool result that is not an error: a failed
    look-around query after the answer does not undo it, a later query that ran does.
    A query that reads no table (literals, VALUES, sqlite_master) never matches; this
    check uses SQLite's parser, and a query it cannot parse is judged by its rows.

    `gold_rows` is one list of rows per example, e.g. a dataset column TRL passes in.
    A text check of the query cannot tell `where customer_id > 0` from the real
    filter; comparing the rows can. Examples without gold rows score 0.0.

    Optional, per example (dataset columns too):
      `gold_sql`: the gold query. When it ends in a top-level ORDER BY, row order
        counts; otherwise it does not. Without it, row order never counts.
      `db_paths`: SQLite files to re-run both queries on (needs `gold_sql`). The
        rows must match on each one as well. One database cannot tell
        `age >= 30` from `age > 30` when nobody is 30, nor a query that reads a
        table but writes its rows in literals; a database with other data can.
        Files are opened read-only, only reads are allowed, each query stops after
        5 seconds, a value over 1 MB is an error, and the rollout's query is read
        only up to one row more than gold_sql returns.
    """
    rewards = []
    for i, completion in enumerate(completions):
        gold = gold_rows[i] if gold_rows is not None and i < len(gold_rows) else None
        if gold is None:
            score, reason = 0.0, "no gold_rows for this example -> 0.0"
        else:
            paths = db_paths[i] if db_paths is not None and i < len(db_paths) else None
            score, reason = _match(
                _turns(completion, i, kwargs),
                gold,
                gold_sql[i] if gold_sql is not None and i < len(gold_sql) else None,
                [paths] if isinstance(paths, str) else paths or (),
            )
        log_score("sql.result_match_reward", score, reasons=[reason], meta={"index": i})
        rewards.append(score)
    return rewards


def reward_correct_answer(completions, answer=None, **kwargs):
    """Exact/near-miss numeric answer reward."""
    scores = []
    for i, (comp, gt) in enumerate(zip(completions, answer or [], strict=False)):
        final_text = ""
        if isinstance(comp, list):
            for msg in reversed(comp):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content", ""))
                    break
        else:
            final_text = str(comp)
        # A number the model only mentions while reasoning is not its answer.
        final_text = _strip_thinking(final_text)
        numbers = re.findall(r"-?\d+(?:,\d{3})*\.?\d*", final_text)
        try:
            gt_val = float(gt)
        except (ValueError, TypeError):
            want = str(gt).strip()
            score = 1.5 if want and final_text.strip() == want else 0.0
            reason = (
                f"non-numeric gold {gt!r}: exact text match -> 1.5"
                if score
                else f"non-numeric gold {gt!r}: text does not match -> 0.0"
            )
            log_score("sql.reward_correct_answer", score, reasons=[reason], meta={"index": i})
            scores.append(score)
            continue
        score = 0.0
        reason = f"no number within tolerance of gold {gt_val} found in completion -> 0.0"
        for n in numbers:
            try:
                pred = float(n.replace(",", ""))
                if abs(pred - gt_val) < 1e-3:
                    score = 1.5
                    reason = f"predicted {pred} matches gold {gt_val} exactly -> 1.5"
                    break
                if gt_val != 0 and abs(pred - gt_val) / abs(gt_val) < 0.01:
                    if score < 0.5:
                        score = 0.5
                        reason = f"predicted {pred} within 1% of gold {gt_val} -> 0.5"
            except ValueError:
                continue
        log_score("sql.reward_correct_answer", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


def reward_tool_used(completions, **kwargs):
    """Reward if model called at least one tool."""
    counts = kwargs.get("tool_call_counts")
    convos = kwargs.get("conversations")
    scores = []
    for i, comp in enumerate(completions):
        if counts is not None and i < len(counts) and int(counts[i] or 0) > 0:
            log_score(
                "sql.reward_tool_used",
                1.0,
                reasons=[f"tool_call_counts[{i}]={counts[i]} > 0 -> 1.0"],
                meta={"index": i},
            )
            scores.append(1.0)
            continue
        msgs = comp if isinstance(comp, list) else []
        if not msgs and convos is not None and i < len(convos):
            msgs = convos[i] or []
        used = any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in msgs)
        score = 1.0 if used else 0.0
        reason = (
            "found a message with role='tool' -> 1.0"
            if used
            else "no tool-role message found -> 0.0"
        )
        log_score("sql.reward_tool_used", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


def reward_concise_answer(completions, **kwargs):
    """Small bonus for short final replies (<= 80 words)."""
    scores = []
    for i, comp in enumerate(completions):
        final_text = ""
        if isinstance(comp, list):
            for msg in reversed(comp):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content", ""))
                    break
        else:
            final_text = str(comp)
        n_words = len(final_text.split())
        score = 0.3 if n_words <= 80 else 0.0
        reason = (
            f"final reply has {n_words} words (<=80) -> 0.3"
            if score
            else f"final reply has {n_words} words (>80) -> 0.0"
        )
        log_score("sql.reward_concise_answer", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


def format_reward(completions, **kwargs):
    """Reward well-formed markdown or structured output."""
    scores = []
    for i, comp in enumerate(completions):
        final_text = ""
        if isinstance(comp, list):
            for msg in reversed(comp):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content", ""))
                    break
        else:
            final_text = str(comp)
        has_structure = bool(re.search(r"(\*\*|#{1,3} |\n-\s|\n\d+\.\s|```)", final_text))
        score = 0.2 if has_structure else 0.0
        reason = (
            "markdown/structured formatting detected -> 0.2"
            if has_structure
            else "no markdown/structured formatting -> 0.0"
        )
        log_score("sql.format_reward", score, reasons=[reason], meta={"index": i})
        scores.append(score)
    return scores


# ── Registry — add new functions here, nothing else needs to change ───────────
