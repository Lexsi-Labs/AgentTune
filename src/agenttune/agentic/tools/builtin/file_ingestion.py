"""
File Ingestion Tool
Generates synthetic directories of SMALL files for agent training.

Files are intentionally compact so that a full agentic rollout
(list_dir → read_file × 2 → run_python → write_file) stays well
under 1024 tokens, preventing the completion_mask / tool_mask
shape-mismatch RuntimeError in TRL's GRPO agentic trainer.

Approximate token budget per rollout:
    system prompt      ~40  tokens
    list_dir result    ~30  tokens
    read revenue.csv   ~80  tokens
    read expenses.csv  ~60  tokens
    run_python result  ~50  tokens
    write_file confirm ~20  tokens
    model reasoning    ~300 tokens
    ─────────────────────────────
    total              ~580 tokens  ← safely under max_completion_length=1024
"""

import csv
import os
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Stand-alone base classes — work both as a plain module and
# inside agenttune pipelines that expect BaseTool instances.
# ─────────────────────────────────────────────────────────────

try:
    from agenttune.agentic.tools.base import BaseTool, ToolResult  # type: ignore
except ImportError:

    class ToolResult:  # type: ignore
        def __init__(self, success: bool, data=None, error: str = ""):
            self.success = success
            self.data = data
            self.error = error

    class BaseTool:  # type: ignore
        name: str = ""
        description: str = ""


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

REGIONS = ["North", "South", "East", "West", "Central"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Operations", "Finance"]
PRODUCTS = ["ProductA", "ProductB", "ProductC", "ProductD"]
LOG_ERRORS = [
    "NullPointerException",
    "TimeoutError",
    "ConnectionRefused",
    "OutOfMemoryError",
    "404 Not Found",
    "500 Internal Server Error",
]


def _rand(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


# ─────────────────────────────────────────────────────────────
# COMPACT FILE GENERATORS
# ─────────────────────────────────────────────────────────────


def _write_revenue_csv(path: str, quarter: str, year: int) -> dict[str, float]:
    """
    Compact: one row per region, revenue total only.
    5 data rows → ~80 tokens when read back.
    Returns {region: revenue} for ground-truth computation.
    """
    rows = [["region", "revenue"]]
    region_rev = {}
    for region in REGIONS:
        revenue = round(sum(_rand(500_000, 3_000_000) for _ in PRODUCTS), 2)
        region_rev[region] = revenue
        rows.append([region, revenue])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return region_rev


def _write_expenses_csv(path: str, quarter: str, year: int) -> dict[str, float]:
    """
    Compact: one row per department, total only.
    5 data rows → ~60 tokens when read back.
    Returns {dept: expenses} for ground-truth computation.
    """
    rows = [["department", "expenses"]]
    dept_exp = {}
    for dept in DEPARTMENTS:
        total = round(
            sum(
                _rand(100_000, 800_000)
                for _ in ["Salaries", "Travel", "Software", "Infrastructure"]
            ),
            2,
        )
        dept_exp[dept] = total
        rows.append([dept, total])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return dept_exp


def _write_notes_txt(path: str, quarter: str, year: int):
    """3-line summary. ~120 tokens."""
    highlight = random.choice(
        [
            f"North region outperformed targets by {_rand(5, 20):.1f}%.",
            f"ProductB saw a {_rand(10, 30):.1f}% increase in units sold.",
            f"Marketing spend reduced by {_rand(5, 15):.1f}% this quarter.",
            f"New enterprise contracts added ${_rand(1, 5):.1f}M in pipeline.",
            f"Operations costs exceeded budget by {_rand(2, 8):.1f}%.",
        ]
    )
    Path(path).write_text(
        f"{quarter} {year} Report\n"
        f"Highlight: {highlight}\n"
        f"Outlook: Management expects continued growth next quarter.\n"
    )


def _write_log_file(path: str, date_str: str, n_lines: int = 20):
    """Short server log — 20 lines. ~200 tokens."""
    base_dt = datetime.strptime(date_str, "%Y-%m-%d")
    lines = []
    for i in range(n_lines):
        dt = base_dt + timedelta(seconds=i * 30)
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        level = random.choices(["INFO", "WARNING", "ERROR"], weights=[80, 15, 5])[0]
        if level == "ERROR":
            msg = f"[{ts}] {level} error={random.choice(LOG_ERRORS)}"
        elif level == "WARNING":
            msg = f"[{ts}] {level} latency_ms={random.randint(800, 2000)}"
        else:
            msg = f"[{ts}] {level} status=200 latency_ms={random.randint(10, 200)}"
        lines.append(msg)
    Path(path).write_text("\n".join(lines))


def _compute_ground_truth(
    region_rev: dict[str, float],
    dept_exp: dict[str, float],
    quarter: str,
    year: int,
) -> list[dict]:
    """Build Q&A pairs directly from dicts — no CSV re-read needed."""
    total_rev = round(sum(region_rev.values()), 2)
    total_exp = round(sum(dept_exp.values()), 2)
    net_profit = round(total_rev - total_exp, 2)
    margin = round(net_profit / total_rev * 100, 2)
    top_region = max(region_rev, key=region_rev.get)
    top_dept = max(dept_exp, key=dept_exp.get)

    return [
        {
            "question": f"What was the total revenue in {quarter} {year}?",
            "answer": f"${total_rev:,.2f}",
        },
        {
            "question": f"What was the net profit margin in {quarter} {year}?",
            "answer": f"{margin:.2f}%",
        },
        {
            "question": f"Which region generated the most revenue in {quarter} {year}?",
            "answer": top_region,
        },
        {
            "question": f"Which department had the highest expenses in {quarter} {year}?",
            "answer": top_dept,
        },
    ]


# ─────────────────────────────────────────────────────────────
# MAIN DATASET GENERATOR  (module-level helper)
# ─────────────────────────────────────────────────────────────


def generate_file_ingestion_dataset(
    n_samples: int = 200,
    seed: int = 42,
    base_dir: str | None = None,
    system_prompt: str | None = None,
):
    """
    Generate a synthetic file ingestion dataset as a HuggingFace Dataset.

    Each sample directory contains 4 compact files:
        revenue.csv   — 5 rows  (~80  tokens when read)
        expenses.csv  — 5 rows  (~60  tokens when read)
        notes.txt     — 3 lines (~120 tokens when read)
        server.log    — 20 lines (~200 tokens when read)

    Args:
        n_samples:     Number of sample directories to create.
        seed:          Random seed for reproducibility.
        base_dir:      Root directory for files (tmp dir if None).
        system_prompt: Override the default agent system prompt.

    Returns:
        (base_dir, HuggingFace Dataset) with columns:
            prompt, answer, sample_dir, quarter, year
    """
    from datasets import Dataset  # type: ignore

    random.seed(seed)
    base_dir = base_dir or tempfile.mkdtemp(prefix="agenttune_files_")
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    years = [2022, 2023, 2024]

    default_prompt = (
        "You are a data analyst. Use tools to answer the question.\n"
        "Steps: list_dir → read_file → run_python → write_file\n"
        "Wrap your final answer in <answer> tags."
    )
    prompt = system_prompt or default_prompt

    rows = []
    for i in range(n_samples):
        quarter = random.choice(quarters)
        year = random.choice(years)
        sample_dir = os.path.join(base_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        rev_path = os.path.join(sample_dir, "revenue.csv")
        exp_path = os.path.join(sample_dir, "expenses.csv")
        txt_path = os.path.join(sample_dir, "notes.txt")
        log_path = os.path.join(sample_dir, "server.log")

        region_rev = _write_revenue_csv(rev_path, quarter, year)
        dept_exp = _write_expenses_csv(exp_path, quarter, year)
        _write_notes_txt(txt_path, quarter, year)
        date_str = f"{year}-{random.randint(1, 12):02d}-01"
        _write_log_file(log_path, date_str, n_lines=20)

        qa_pairs = _compute_ground_truth(region_rev, dept_exp, quarter, year)
        qa = random.choice(qa_pairs)

        rows.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": prompt + f"\n\nDirectory: {sample_dir}",
                    },
                    {
                        "role": "user",
                        "content": qa["question"],
                    },
                ],
                "answer": qa["answer"],
                "sample_dir": sample_dir,
                "quarter": quarter,
                "year": year,
            }
        )

    return base_dir, Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────
# BaseTool WRAPPER
# ─────────────────────────────────────────────────────────────


class FileIngestionTool(BaseTool):
    """
    Agent-callable tool that generates compact synthetic file directories
    for file ingestion training. Each directory contains:
        revenue.csv   — 5 rows  (~80  tokens)
        expenses.csv  — 5 rows  (~60  tokens)
        notes.txt     — 3 lines (~120 tokens)
        server.log    — 20 lines (~200 tokens)

    Three agent-callable actions:
        generate   — create n_samples directories and return the HF dataset
        list_dir   — list files in a sample directory
        read_file  — read the contents of a file in a sample directory
    """

    name = "file_ingestion"
    description = (
        "Generate compact synthetic directories of mixed files "
        "(CSV, TXT, LOG) for file ingestion agent training, and "
        "expose list_dir / read_file tools for agents to use during rollouts."
    )

    def __init__(
        self,
        base_dir: str | None = None,
        seed: int = 42,
        system_prompt: str | None = None,
    ):
        """
        Args:
            base_dir:      Root directory for generated sample dirs (tmp if None).
            seed:          Random seed for reproducibility.
            system_prompt: Override the default agent system prompt.
        """
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="agenttune_files_")
        self.seed = seed
        self.system_prompt = system_prompt
        self._dataset = None

    # ── Parameters schema ─────────────────────────────────────────────────────

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["generate", "list_dir", "read_file"],
                    "description": (
                        "- generate  : Create n_samples synthetic directories and dataset.\n"
                        "- list_dir  : List all files in a sample directory.\n"
                        "- read_file : Read the contents of a specific file."
                    ),
                },
                "n_samples": {
                    "type": "integer",
                    "default": 200,
                    "description": "Number of sample directories to generate (generate action).",
                },
                "sample_dir": {
                    "type": "string",
                    "description": "Absolute path to the sample directory (list_dir / read_file).",
                },
                "filename": {
                    "type": "string",
                    "description": "File name within sample_dir to read (read_file). E.g. 'revenue.csv'.",
                },
            },
            "required": ["action"],
        }

    # ── Core execute ──────────────────────────────────────────────────────────

    def execute(
        self,
        action: str,
        n_samples: int = 200,
        sample_dir: str | None = None,
        filename: str | None = None,
    ) -> ToolResult:
        """
        Execute any supported action.

        Args:
            action:     One of 'generate', 'list_dir', 'read_file'.
            n_samples:  Directories to create (generate only).
            sample_dir: Path to an existing sample directory (list_dir / read_file).
            filename:   File name within sample_dir to read (read_file only).

        Returns:
            ToolResult with .success and .output / .error populated.
        """
        try:
            if action == "generate":
                return self._generate(n_samples=n_samples)
            elif action == "list_dir":
                return self._list_dir(sample_dir=sample_dir)
            elif action == "read_file":
                return self._read_file(sample_dir=sample_dir, filename=filename)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown action '{action}'. Choose: generate, list_dir, read_file.",
                )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    # ── Action implementations ────────────────────────────────────────────────

    def _generate(self, n_samples: int = 200) -> ToolResult:
        """Generate n_samples synthetic directories and build the HF dataset."""
        base_dir, dataset = generate_file_ingestion_dataset(
            n_samples=n_samples,
            seed=self.seed,
            base_dir=self.base_dir,
            system_prompt=self.system_prompt,
        )
        self._dataset = dataset
        return ToolResult(
            success=True,
            output=(
                f"Generated {n_samples} sample directories in {base_dir}. "
                f"Dataset has {len(dataset)} rows with columns: "
                f"{dataset.column_names}."
            ),
        )

    def _list_dir(self, sample_dir: str | None) -> ToolResult:
        """
        List all files in a sample directory. Safe — read-only.

        Args:
            sample_dir: Absolute path to the directory.

        Returns:
            Newline-separated list of file names.
        """
        if not sample_dir:
            return ToolResult(
                success=False, output=None, error="sample_dir is required for list_dir."
            )
        if not os.path.isdir(sample_dir):
            return ToolResult(
                success=False, output=None, error=f"Directory not found: {sample_dir}"
            )
        files = sorted(os.listdir(sample_dir))
        return ToolResult(
            success=True,
            output="\n".join(files) if files else "(empty directory)",
        )

    def _read_file(
        self,
        sample_dir: str | None,
        filename: str | None,
    ) -> ToolResult:
        """
        Read the contents of a file inside a sample directory.
        Restricted to the sample directory — no path traversal.

        Args:
            sample_dir: Absolute path to the directory.
            filename:   File name to read (e.g. 'revenue.csv').

        Returns:
            Full file contents as a string.
        """
        if not sample_dir:
            return ToolResult(
                success=False, output=None, error="sample_dir is required for read_file."
            )
        if not filename:
            return ToolResult(
                success=False, output=None, error="filename is required for read_file."
            )

        safe_path = Path(sample_dir).resolve() / Path(filename).name
        if not safe_path.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"File not found: {filename} in {sample_dir}",
            )
        if not safe_path.is_file():
            return ToolResult(success=False, output=None, error=f"Path is not a file: {safe_path}")

        return ToolResult(success=True, output=safe_path.read_text())

    # ── Convenience methods ───────────────────────────────────────────────────

    def get_dataset(self):
        """
        Return the cached HuggingFace Dataset.
        Calls generate() with defaults if not yet built.

        Returns:
            HuggingFace Dataset with columns: prompt, answer, sample_dir, quarter, year
        """
        if self._dataset is None:
            self._generate()
        return self._dataset
