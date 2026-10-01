"""``lexsi_provenance.json``: the lineage record every Lexsi library writes into its output dirs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROVENANCE_FILE = "lexsi_provenance.json"


def read_provenance(directory: Any) -> dict | None:
    """The provenance object in ``directory``, or None (missing file, not a folder, Hub id)."""
    try:
        return json.loads((Path(directory) / PROVENANCE_FILE).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None


def write_provenance(
    out_dir: str | Path,
    method: str,
    base_model: Any = None,
    dataset: Any = None,
    dataset_config: str | None = None,
    params: dict | None = None,
) -> dict:
    """Write ``lexsi.provenance/1`` to ``out_dir`` and return it.

    ``base_model`` may be a path/Hub id or a loaded model (its ``name_or_path`` is used).
    A string ``dataset`` becomes an input; when it is a folder carrying its own
    ``lexsi_provenance.json`` (e.g. a CuratorKIT export) that object is embedded.
    """
    from agenttune import __version__

    if base_model is not None and not isinstance(base_model, str):
        base_model = getattr(base_model, "name_or_path", None) or None
    inputs = []
    if isinstance(dataset, str):
        inputs.append(
            {
                "kind": "dataset",
                "ref": dataset,
                "config": dataset_config,
                "provenance": read_provenance(dataset),
            }
        )
    record = {
        "schema": "lexsi.provenance/1",
        "library": "agenttune",
        "version": __version__,
        "git_sha": None,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": base_model,
        "method": method,
        "inputs": inputs,
        "params": params or {},
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / PROVENANCE_FILE).write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    return record
