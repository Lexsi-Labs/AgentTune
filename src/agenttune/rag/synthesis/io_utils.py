"""
Tabular IO + JSON-structured LLM output helpers.

`dump_table` writes a list of records to CSV and Parquet (Parquet if
pyarrow is installed, else CSV only). Every stage's output is dumped this
way so the pipeline is inspectable at every step.

`extract_json` robustly parses JSON out of an LLM response (handles
```json fences, trailing prose, and partial JSON by locating the outermost
{...} or [...] block).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def dump_table(rows: list[dict[str, Any]], path: str) -> str:
    """Write `rows` to CSV (always) and Parquet (if pyarrow available).

    Returns the CSV path (Parquet path is `<path>.parquet` when written).
    List/dict values should already be JSON-stringified by the caller.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return str(p)
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(p, index=False)
        try:
            import pyarrow  # noqa

            parquet_path = str(p).rsplit(".", 1)[0] + ".parquet"
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass  # pyarrow not installed — CSV is enough
    except Exception:
        # No pandas — minimal CSV writer
        cols = list(rows[0].keys())
        with open(p, "w", encoding="utf-8") as f:
            import csv

            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})
    return str(p)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating fences/prose.

    Tries direct parse, then fenced block, then the outermost {...}/[...].
    Raises ValueError if nothing parses.
    """
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # outermost object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    # outermost array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    raise ValueError(f"no JSON found in response: {text[:200]}")
