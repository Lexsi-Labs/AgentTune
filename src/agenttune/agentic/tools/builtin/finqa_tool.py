"""
finqa_tool.py
-------------
Wraps your SQLDatabaseTool as four plain callables ready for
create_agentic_trainer(tools=[...]).

Replaces the rllm tool stack with your existing SQLDatabaseTool:

    rllm tool              →  SQLDatabaseTool method
    ─────────────────────────────────────────────────
    GetTableNames          →  list_tables()        (unwrapped ToolResult)
    GetTableInfo           →  schema(table)        (unwrapped ToolResult)
    SQLQuery               →  query(sql)           (JSON-stringified)
    Calculator             →  standalone function  (unchanged)

Usage in train.py
-----------------
    from projects.finqa.finqa_tool import build_finqa_tools, calculator

    db_tool = SQLDatabaseTool("sqlite:///path/to/finqa.db")
    list_tables, get_table_schema, query_finqa_tables = build_finqa_tools(db_tool)

    trainer = create_agentic_trainer(
        tools=[list_tables, get_table_schema, query_finqa_tables, calculator],
        ...
    )

Design notes
------------
* NO get_explanation exposed — it leaks gold reasoning during RL training.
* Calculator is kept as a standalone import; it has no SQL equivalent.
* All three SQL wrappers return plain strings so the model can read them
  directly in its context window.
* sql_grounding_reward in finqa_rewards.py greps for the function name
  "query_finqa_tables" — do not rename it without updating that reward.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ── FIXED: correct repo and data directory ──────────────────────────────────
HF_REPO_ID = "rLLM/rLLM-FinQA-Dataset"
DATA_DIR = Path("./data/data")

if TYPE_CHECKING:
    from your_package.tools.sql_database import SQLDatabaseTool


TABLES_ROOT = Path("data/data/company_tables")
DB_PATH = "finqa.db"


def build_finqa_db(db_path: str = DB_PATH, tables_root: Path = TABLES_ROOT):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")

    companies = [d for d in tables_root.iterdir() if d.is_dir()]
    loaded, skipped = 0, 0

    for company_dir in companies:
        company = company_dir.name
        metadata_file = company_dir / "tables_cleaned_all_companies.json"

        if not metadata_file.exists():
            continue

        with open(metadata_file) as f:
            all_tables = json.load(f)

        for table_name, info in all_tables.items():
            try:
                table_json = info.get("table", "{}")
                import io

                import pandas as pd

                df = pd.read_json(io.StringIO(table_json), convert_dates=False)

                if df.empty:
                    skipped += 1
                    continue

                # Sanitize columns
                df.columns = [
                    str(c) if str(c).strip() else f"col_{i}" for i, c in enumerate(df.columns)
                ]

                # SQL table name: company_tablename
                base = os.path.splitext(os.path.basename(table_name))[0]
                sql_name = f"{company}_{base}".replace("-", "_").replace(" ", "_")

                df.to_sql(sql_name, conn, if_exists="replace", index=False)
                loaded += 1
            except Exception:
                skipped += 1

    conn.commit()
    conn.close()
    logger.info(f"Done. Loaded: {loaded}, Skipped: {skipped}")


# ---------------------------------------------------------------------------
# Calculator — standalone, no SQL dependency
# ---------------------------------------------------------------------------


def calculator(expression: str, variables: dict | None = None) -> float | str:
    """
    Evaluate a mathematical expression and return the numeric result.
    Use this AFTER retrieving values from query_finqa_tables to perform
    arithmetic. Do NOT pass SQL queries here — only arithmetic expressions.

    Supported:
      - Standard operators:  +  -  *  /  **  (  )
      - ^ treated as power:  2^3  →  8  (not XOR)
      - Currency stripped:   $  €  £
      - Percentage expanded: 15%  →  (15/100)
      - Thousand commas:     1,234,567  →  1234567
      - Unicode minus / en-dash / em-dash all treated as minus.
      - Named variables:     {"Total Cash 2023": 85200, "2024_val": 900}
      - Digit-leading names: "2024_val" →  auto-prefixed to "var_2024_val"
      - Spaced names:        "Total Cash 2023" → auto-sanitized

    Args:
        expression: A mathematical expression string,
                    e.g. "(85200 - 78129) / 78129 * 100"
                    or   "(Total Cash 2023 - Total Cash 2024) / Total Cash 2023 * 100"
        variables:  Optional dict mapping named placeholders to numeric values,
                    e.g. {"Total Cash 2023": 85200, "Total Cash 2024": 78129}

    Returns:
        float result on success, or an error string starting with "Error:"
        on failure.

    Examples:
        calculator("(85200 - 78129) / 78129 * 100")                          ->  9.053...
        calculator("3500 + 2000 + 1500")                                     ->  7000.0
        calculator("15%")                                                     ->  0.15
        calculator("(A - B) / B * 100", {"A": 85200, "B": 78129})            ->  9.053...
        calculator("(Total Cash 2023 - Total Cash 2024) / Total Cash 2023 * 100",
                   {"Total Cash 2023": 85200, "Total Cash 2024": 78129})      ->  9.053...
    """
    from asteval import Interpreter

    if variables is None:
        variables = {}
    if not isinstance(expression, str):
        return "Error: expression must be a string."

    expr = expression.strip()

    # ── Normalise whitespace / newlines ───────────────────────────────────────
    expr = expr.replace("\n", " ").replace("\r", " ")

    # ── Substitute named variables BEFORE any other transforms ───────────────
    if variables:

        def sanitize(name: str) -> str:
            s = name.strip().replace(" ", "_").replace("-", "_")
            s = re.sub(r"[^\w]", "_", s)
            if re.match(r"^\d", s):
                s = "var_" + s
            return s

        sanitized_vars = {sanitize(k): v for k, v in variables.items()}

        # Replace longest keys first to avoid partial matches
        for original in sorted(variables.keys(), key=len, reverse=True):
            expr = expr.replace(original, sanitize(original))
    # if variables:
    #     def sanitize(name: str) -> str:
    #         s = name.strip().replace(" ", "_").replace("-", "_")
    #         s = re.sub(r"[^\w]", "_", s)
    #         if re.match(r"^\d", s):
    #             s = "var_" + s
    #         return s

    #     sanitized_vars = {}
    #     for k, v in variables.items():
    #         key = sanitize(k)
    #         # Force to float — model often passes strings like "85,200" or "85200"
    #         if isinstance(v, str):
    #             try:
    #                 v = float(v.replace(",", "").replace("$", "").strip())
    #             except ValueError:
    #                 pass  # leave as-is, will fail gracefully in asteval
    #         sanitized_vars[key] = v
    else:
        sanitized_vars = {}

    # ── Symbol / encoding normalisations ─────────────────────────────────────
    expr = expr.replace("^", "**")
    expr = expr.replace("$", "").replace("€", "").replace("£", "")
    expr = expr.replace("\u00a0", " ")
    expr = expr.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")

    for old, new in {"\uff08": "(", "\uff09": ")", "\xd7": "*", "\xf7": "/"}.items():
        expr = expr.replace(old, new)

    for i, fw in enumerate("\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19"):
        expr = expr.replace(fw, str(i))

    # ── Percentage and thousand-comma expansion ───────────────────────────────
    expr = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expr)
    expr = re.sub(r"\d{1,3}(?:,\d{3})+", lambda m: m.group(0).replace(",", ""), expr)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    try:
        aeval = Interpreter()
        for name, val in sanitized_vars.items():
            aeval.symtable[name] = val
        result = aeval.eval(expr)
        if aeval.error:
            errs = "; ".join(str(e.get_error()) for e in aeval.error)
            return f"Error evaluating expression: '{expression}'. Details: {errs}"
        return float(result)
    except Exception as e:
        return f"Error evaluating expression: '{expression}'. Details: {e}"


# ---------------------------------------------------------------------------
# Factory — builds the three SQL-backed tools from a SQLDatabaseTool instance
# ---------------------------------------------------------------------------


def build_finqa_tools(db_tool: SQLDatabaseTool):
    """
    Return (list_tables, get_table_schema, query_finqa_tables) bound to db_tool.

    Args:
        db_tool: An initialised SQLDatabaseTool instance pointing at the
                 FinQA SQLite database, e.g.:
                     SQLDatabaseTool("sqlite:///finqa.db")

    Returns:
        Tuple of three callables ready for create_agentic_trainer(tools=[...]).
    """

    # def list_tables() -> str:
    #     """
    #     Return the names of all tables available in the FinQA database.

    #     Call this first to discover which financial tables exist before
    #     inspecting schemas or running queries.

    #     Returns:
    #         Comma-separated string of table names, e.g.
    #         "apple_revenue, apple_expenses, 3m_income_statement, ..."
    #         Returns an error string if the database cannot be reached.

    #     Example:
    #         list_tables()
    #         -> "apple_revenue, apple_operating_segments, 3m_long_term_debt"
    #     """
    #     result = db_tool.list_tables()
    #     if result.success:
    #         output = result.output
    #         return str(output) if output is not None else "No tables found."
    #     return f"Error listing tables: {result.error}"
    def list_tables(company_prefix: str = "") -> str:
        """
        Return table names in the FinQA database, optionally filtered by company.

        Args:
            company_prefix: Filter tables to only those starting with this prefix.
                            E.g. "apple" returns "apple_revenue, apple_expenses, ..."
                            Leave empty to list ALL tables (avoid — can be very long).

        Returns:
            Comma-separated string of matching table names, capped at 50.
            Returns an error string if the database cannot be reached.

        Example:
            list_tables("apple")
            -> "apple_revenue, apple_operating_segments, apple_long_term_debt"
        """
        result = db_tool.list_tables()
        if not result.success:
            return f"Error listing tables: {result.error}"

        all_tables = str(result.output) if result.output is not None else ""
        if not all_tables:
            return "No tables found."

        # Parse the comma-separated string back into a list
        table_list = [t.strip() for t in all_tables.split(",") if t.strip()]

        # Filter by prefix if provided
        if company_prefix:
            prefix = company_prefix.lower().replace(" ", "_").replace("-", "_")
            table_list = [t for t in table_list if t.lower().startswith(prefix)]

        if not table_list:
            return f"No tables found matching prefix '{company_prefix}'."

        # Hard cap to prevent context flooding
        MAX_TABLES = 50
        truncated = table_list[:MAX_TABLES]
        suffix = (
            f" ... ({len(table_list) - MAX_TABLES} more, narrow your prefix)"
            if len(table_list) > MAX_TABLES
            else ""
        )
        return ", ".join(truncated) + suffix

    def get_table_schema(table_name: str) -> str:
        """
        Return the column names, data types, and sample rows for a table.

        Call this after list_tables() and before query_finqa_tables() to
        learn the exact column names to use in your SQL query.

        Args:
            table_name: A single table name, or a comma-separated list of
                        table names (e.g. "apple_revenue" or
                        "apple_revenue, apple_expenses").
                        Use the exact names returned by list_tables().

        Returns:
            CREATE TABLE statement(s) with all columns and types, plus
            sample rows showing real values.
            Returns an error string if the table is not found.

        Tips:
          - Numeric values are often stored as strings with commas,
            e.g. "1,234". Cast them in your query:
                CAST(REPLACE(amount, ',', '') AS REAL)
          - The first column is usually a row-label index (e.g. "Net income",
            "Less: imputed interest"). Filter with a WHERE clause on it.

        Example:
            get_table_schema("apple_revenue")
            -> "CREATE TABLE apple_revenue (row_label TEXT, \\"2023\\" TEXT, ...)"
        """
        result = db_tool.schema(table_name)
        if result.success:
            output = result.output
            return str(output) if output is not None else "No schema found."
        return f"Error fetching schema: {result.error}"

    def query_finqa_tables(sql_command: str) -> str:
        """
        Execute a SQLite SELECT query over the FinQA financial tables and
        return the results as a JSON string.

        Always call list_tables() then get_table_schema() first so you know
        the exact table name and column names to use.

        Args:
            sql_command: A valid SQLite SELECT statement.

        Returns:
            JSON array of row objects, e.g.:
            '[{"row_label": "Net income", "2023": "96,995", "2022": "99,803"}]'
            Returns an error string (starting with "Error") on failure.

        Rules / Tips:
          - Do NOT use SELECT * — always list columns explicitly.
          - Filter to the rows you need with a WHERE clause:
                WHERE row_label = 'Net income'
          - Numeric columns are stored as strings — cast before arithmetic:
                CAST(REPLACE(\\"2023\\", ',', '') AS REAL)
          - Use LIMIT to avoid returning huge result sets.
          - Use the exact table name from list_tables() in the FROM clause.

        Example:
            query_finqa_tables(
                "SELECT row_label, \\"2023\\", \\"2022\\" "
                "FROM apple_revenue "
                "WHERE row_label = 'Net income'"
            )
            -> '[{"row_label": "Net income", "2023": "96,995", "2022": "99,803"}]'
        """
        output = db_tool.query(sql_command)

        if isinstance(output, dict) and "error" in output:
            return f"Error: {output['error']}"

        try:
            return json.dumps(output)
        except (TypeError, ValueError):
            return str(output)

    return list_tables, get_table_schema, query_finqa_tables


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except json.JSONDecodeError:
            return [stripped]
    return []


def _preprocess(df) -> list[dict]:
    rows = []
    for _, ex in df.iterrows():
        rows.append(
            {
                "question": ex["user_query"],
                "ground_truth": str(ex["answer"]),
                "company": ex["company"],
                "question_id": str(ex["id"]),
                "question_type": ex["question_type"],
            }
        )
    return rows


# def _build_rows(rows: list[dict], prompt_template: str | None = None) -> list[dict]:
#     out = []
#     for example in rows:
#         prompt_text = prompt_template or (
#             f"Company: {example['company']}\n"
#             f"Question: {example['question']}\n\n"
#             "Use the available tools to look up the relevant financial data "
#             "and compute the answer.\n"
#             "Suggested steps:\n"
#             "  1. Call list_tables() to see what tables exist for this company.\n"
#             "  2. Call get_table_schema(table_name) on the relevant table.\n"
#             "  3. Call query_finqa_tables(sql) to retrieve the specific values.\n"
#             "  4. Call calculator(expression) to compute the final number.\n"
#             "Wrap your final answer in <answer>...</answer> tags."
#         )
#         out.append({
#             "prompt":        [{"role": "user", "content": prompt_text}],
#             "answer":        example["ground_truth"],
#             "company":       example["company"],
#             "question_id":   example["question_id"],
#             "question_type": example["question_type"],
#         })
#     return out
def _build_rows(rows: list[dict], prompt_template: str | None = None) -> list[dict]:
    out = []
    for example in rows:
        company_slug = example["company"].lower().replace(" ", "_").replace("-", "_")
        prompt_text = prompt_template or (
            f"Company: {example['company']}\n"
            f"Question: {example['question']}\n\n"
            "Use the available tools to look up the relevant financial data "
            "and compute the answer.\n"
            "Suggested steps:\n"
            f'  1. Call list_tables("{company_slug}") to see tables for this company.\n'
            "  2. Call get_table_schema(table_name) on the relevant table.\n"
            "  3. Call query_finqa_tables(sql) to retrieve the specific values.\n"
            "  4. Call calculator(expression) to compute the final number.\n"
            "Wrap your final answer in <answer>...</answer> tags."
        )
        out.append(
            {
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": example["ground_truth"],
                "company": example["company"],
                "question_id": example["question_id"],
                "question_type": example["question_type"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_finqa_datasets(
    data_dir: Path | str = DATA_DIR,
    force_download: bool = False,
) -> tuple[Dataset, Dataset]:
    """
    Download (if needed) and return (train_dataset, val_dataset) as
    HuggingFace Dataset objects ready for create_agentic_trainer().

    Args:
        data_dir:        Where to find the CSVs. Defaults to ./data/multi_table_data
        force_download:  Re-download even if data already exists.

    Returns:
        (train_dataset, val_dataset)

    Usage:
        from projects.finqa.finqa_tool import load_finqa_datasets
        train_dataset, val_dataset = load_finqa_datasets()
    """
    data_dir = Path(data_dir)

    # ── Download & extract ────────────────────────────────────────────────
    train_csv = data_dir / "train_finqa.csv"
    val_csv = data_dir / "val_finqa.csv"

    if force_download or not train_csv.exists() or not val_csv.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading data.tar.gz from {HF_REPO_ID}...")
        tar_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="data.tar.gz",
            repo_type="dataset",
        )
        # Extract to parent of data_dir so that multi_table_data/ and
        # company_tables/ land at ./data/multi_table_data and
        # ./data/company_tables respectively.
        with tarfile.open(tar_path, "r:gz") as tar:
            # filter="data" (py3.12+) rejects path-traversal / absolute members before write.
            tar.extractall(path=Path("./data"), filter="data")
        logger.info("Extracted files:")
        for f in sorted(data_dir.rglob("*.csv")):
            logger.info(f"  {f}")
    else:
        logger.info(f"Data already exists at {data_dir}, skipping download.")

    # ── Load CSVs ─────────────────────────────────────────────────────────
    # FIXED: paths now point to data/multi_table_data/train_finqa.csv etc.
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # ── Build datasets ────────────────────────────────────────────────────
    train_dataset = Dataset.from_list(_build_rows(_preprocess(train_df)))
    val_dataset = Dataset.from_list(_build_rows(_preprocess(val_df)))

    logger.info(f"Train: {len(train_dataset)} rows | Val: {len(val_dataset)} rows")
    return train_dataset, val_dataset
