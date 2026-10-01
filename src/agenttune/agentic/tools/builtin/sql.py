import ast
import logging
import sqlite3
from pathlib import Path

import pandas as pd
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDatabaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase

from ..base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class SQLDatabaseTool(BaseTool):
    name = "sql_database"
    description = (
        "Query and inspect a SQL database. Supports listing tables, viewing schemas, "
        "executing queries, creating a database from a HuggingFace dataset, and "
        "email-specific actions: build_enron_db, search_inbox, read_email, list_senders."
    )

    def __init__(self, db_uri: str | None = None):
        """
        Args:
            db_uri: SQLAlchemy URI set once at construction.
                    e.g. "sqlite:///enron.db" or "postgresql://user:pass@host/db"
                    Never needs to be passed again after this.
        """
        self.db_uri = db_uri
        self._db = None  # lazy-loaded LangChain SQLDatabase instance

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_uri(self, db_uri: str | None = None) -> str:
        """Instance uri wins unless an explicit override is passed."""
        uri = db_uri or self.db_uri
        if not uri:
            raise ValueError(
                "No db_uri set. Pass it at construction: SQLDatabaseTool('sqlite:///mydb.db')"
            )
        return uri

    def _get_db(self, db_uri: str | None = None) -> SQLDatabase:
        """
        Return a cached LangChain SQLDatabase.
        Re-creates only if an explicit override uri differs from the cached one.
        """
        uri = self._resolve_uri(db_uri)
        if self._db is None or uri != self.db_uri:
            self._db = SQLDatabase.from_uri(uri)
        return self._db

    def _parse_output(self, raw):
        """Parse QuerySQLDatabaseTool string output back to a list if possible."""
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list | tuple):
                    return list(parsed)
            except (ValueError, SyntaxError):
                pass
        return raw

    def _sqlite_path(self, db_uri: str | None = None) -> str:
        """Extract the file path from a sqlite:/// URI."""
        uri = self._resolve_uri(db_uri)
        if not uri.startswith("sqlite:///"):
            raise ValueError(f"This action requires a sqlite:/// URI, got: {uri}")
        return uri.replace("sqlite:///", "")

    def _conn(self) -> sqlite3.Connection:
        """Open a raw sqlite3 connection to the instance db."""
        return sqlite3.connect(self._sqlite_path(), check_same_thread=False)

    # ── Parameters schema ─────────────────────────────────────────────────────

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        # Generic SQL actions
                        "sql_db_list_tables",
                        "sql_db_schema",
                        "sql_db_query",
                        "create_from_dataset",
                        # Email / Enron actions
                        "build_finqa_db",
                        "build_enron_db",
                        "search_inbox",
                        "read_email",
                        "list_senders",
                    ],
                },
                # Generic params
                "input": {
                    "type": "string",
                    "description": (
                        "- sql_db_list_tables : leave empty.\n"
                        "- sql_db_schema      : comma-separated table names.\n"
                        "- sql_db_query       : SQL query string.\n"
                        "- create_from_dataset: HuggingFace dataset name."
                    ),
                    "default": "",
                },
                "table_name": {
                    "type": "string",
                    "description": "Table name when creating from dataset. Defaults to 'data'.",
                    "default": "data",
                },
                "split": {
                    "type": "string",
                    "description": "Dataset split. Defaults to 'train'.",
                    "default": "train",
                },
                # build_enron_db params
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows to load when building the Enron DB (0 = all). Defaults to 50000.",
                    "default": 50_000,
                },
                # search_inbox params
                "inbox": {
                    "type": "string",
                    "description": "Inbox email address to search within (required for search_inbox).",
                    "default": "",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords to match via FTS5 AND logic (required for search_inbox).",
                },
                "from_addr": {
                    "type": "string",
                    "description": "Filter by sender address (search_inbox).",
                    "default": "",
                },
                "to_addr": {
                    "type": "string",
                    "description": "Filter to emails sent to this address (search_inbox).",
                    "default": "",
                },
                "sent_after": {
                    "type": "string",
                    "description": "ISO date 'YYYY-MM-DD' — return emails after this date (search_inbox).",
                    "default": "",
                },
                "sent_before": {
                    "type": "string",
                    "description": "ISO date 'YYYY-MM-DD' — return emails before this date (search_inbox).",
                    "default": "",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return from search_inbox (1–10). Defaults to 10.",
                    "default": 10,
                },
                # read_email params
                "message_id": {
                    "type": "string",
                    "description": "The message_id from search_inbox results (required for read_email).",
                },
                # list_senders params
                "min_emails": {
                    "type": "integer",
                    "description": "Only show senders with at least this many emails (list_senders). Defaults to 10.",
                    "default": 10,
                },
            },
            "required": ["action"],
        }

    # ── Core execute ──────────────────────────────────────────────────────────
    def _build_finqa_db(
        self,
        db_path: str,
        tables_root: Path,
    ) -> "ToolResult":
        """
        Build the FinQA SQLite database from local company_tables JSON files.

        Each company folder must contain tables_cleaned_all_companies.json.
        SQL table names follow: {company}_{table_base_name}

        Args:
            db_path:     Path to the SQLite file to create/overwrite.
            tables_root: Path to the directory containing per-company folders.

        Returns:
            ToolResult with success/failure message.
        """
        import io
        import json
        import os
        from pathlib import Path

        tables_root = Path(tables_root)

        if not tables_root.exists():
            return ToolResult(
                success=False,
                output=None,
                error=f"tables_root not found: {tables_root}",
            )

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")

            companies = [d for d in tables_root.iterdir() if d.is_dir()]
            loaded, skipped = 0, 0

            for company_dir in sorted(companies):
                company = company_dir.name
                metadata_file = company_dir / "tables_cleaned_all_companies.json"

                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file) as f:
                        all_tables = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"  [WARN] Skipping {company}: bad JSON — {e}")
                    continue

                for table_name, info in all_tables.items():
                    try:
                        table_json = info.get("table", "{}")
                        df = pd.read_json(io.StringIO(table_json), convert_dates=False)

                        if df.empty or len(df.columns) == 0:
                            skipped += 1
                            continue

                        # Sanitize column names
                        df.columns = [
                            str(c).strip() if str(c).strip() else f"col_{i}"
                            for i, c in enumerate(df.columns)
                        ]

                        # Build SQL table name: company_tablebasename
                        base = os.path.splitext(os.path.basename(table_name))[0]
                        sql_name = f"{company}_{base}".replace("-", "_").replace(" ", "_")

                        df.to_sql(sql_name, conn, if_exists="replace", index=False)
                        loaded += 1

                    except Exception:
                        skipped += 1

            conn.commit()
            conn.close()

            self._db = None  # invalidate LangChain cache

            logger.info(f"✅ FinQA DB built at {db_path}")
            logger.info(f"   Loaded : {loaded:,}")
            logger.info(f"   Skipped: {skipped:,}")

            return ToolResult(
                success=True,
                output=f"FinQA DB built at {db_path} — {loaded:,} tables loaded, {skipped:,} skipped",
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def build_finqa_db(
        self,
        tables_root: str = "data/data/company_tables",
    ) -> "ToolResult":
        """
        Build the FinQA database from local company_tables JSON files.
        Uses the instance db_uri — no uri needed.

            tool = SQLDatabaseTool("sqlite:///finqa.db")
            tool.build_finqa_db(tables_root="data/data/company_tables")
        """
        return self.execute(action="build_finqa_db", tables_root=tables_root)

    def execute(
        self,
        action: str,
        input: str = "",
        table_name: str = "data",
        split: str = "train",
        db_uri: str | None = None,
        tables_root: str = "data/data/company_tables",
        # Email / Enron params
        max_rows: int = 50_000,
        inbox: str = "",
        keywords: list[str] | None = None,
        from_addr: str = "",
        to_addr: str = "",
        sent_after: str = "",
        sent_before: str = "",
        max_results: int = 10,
        message_id: str = "",
        min_emails: int = 10,
    ) -> ToolResult:
        """
        Execute any supported action against the database.

        db_uri is optional — uses the instance uri set at construction.
        """
        try:
            uri = self._resolve_uri(db_uri)

            # ── Email: build Enron database ───────────────────────────────
            if action == "build_enron_db":
                return self._build_enron_db(
                    db_path=self._sqlite_path(uri),
                    max_rows=max_rows,
                )

            # ── Email: FTS5-backed search/read actions ────────────────────
            if action == "search_inbox":
                return ToolResult(
                    success=True,
                    output=self.search_inbox(
                        inbox=inbox,
                        keywords=keywords or [],
                        from_addr=from_addr,
                        to_addr=to_addr,
                        sent_after=sent_after,
                        sent_before=sent_before,
                        max_results=max_results,
                    ),
                )

            if action == "read_email":
                if not message_id:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="read_email requires a message_id.",
                    )
                return ToolResult(success=True, output=self.read_email(message_id))

            if action == "list_senders":
                return ToolResult(
                    success=True,
                    output=self.list_senders(min_emails=min_emails),
                )

            # ── Generic: create from HuggingFace dataset ──────────────────
            if action == "create_from_dataset":
                return self._create_from_dataset(
                    dataset_name=input,
                    db_uri=uri,
                    table_name=table_name,
                    split=split,
                )

            if action == "build_finqa_db":
                # tables_root = kwargs.get("tables_root", "data/data/company_tables")
                return self._build_finqa_db(
                    db_path=self._sqlite_path(uri),
                    tables_root=Path(tables_root),
                )

            # ── Generic: LangChain SQL tools ──────────────────────────────
            db = self._get_db(uri)
            tools = {
                "sql_db_list_tables": ListSQLDatabaseTool(db=db),
                "sql_db_schema": InfoSQLDatabaseTool(db=db),
                "sql_db_query": QuerySQLDatabaseTool(db=db),
            }

            if action not in tools:
                return ToolResult(
                    success=False,
                    output=None,
                    error=(
                        f"Unknown action '{action}'. "
                        f"Available: {list(tools.keys()) + ['build_enron_db', 'search_inbox', 'read_email', 'list_senders']}"
                    ),
                )

            raw = tools[action].run(input)
            return ToolResult(success=True, output=self._parse_output(raw))

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    # ── Email search methods (general — works with any inbox/email SQLite DB) ─

    def search_inbox(
        self,
        inbox: str,
        keywords: list[str],
        from_addr: str = "",
        to_addr: str = "",
        sent_after: str = "",
        sent_before: str = "",
        max_results: int = 10,
    ) -> str:
        """
        Search an inbox for emails matching ALL provided keywords via FTS5.

        Scoped to a specific inbox address — returns emails where that address
        appears as sender or recipient. Use read_email() for the full body.

        Args:
            inbox:       The inbox email address to search within (required).
            keywords:    List of keywords to match (AND logic). Required.
            from_addr:   Optionally filter by sender address.
            to_addr:     Optionally filter to emails sent to this address.
            sent_after:  ISO date 'YYYY-MM-DD' — return emails after this date.
            sent_before: ISO date 'YYYY-MM-DD' — return emails before this date.
            max_results: Maximum results to return (1–10, default 10).

        Returns:
            Formatted list of matching emails with message_id and snippet.
        """
        if not keywords:
            return "Error: keywords list must not be empty."
        if max_results > 10:
            max_results = 10

        try:
            conn = self._conn()

            fts_query = " ".join(f'"{k.replace(chr(34), chr(34)*2)}"' for k in keywords)

            where = [
                "emails_fts MATCH ?",
                "(e.from_address = ? OR EXISTS ("
                "  SELECT 1 FROM recipients r"
                "  WHERE r.recipient_address = ? AND r.email_id = e.message_id"
                "))",
            ]
            params = [fts_query, inbox, inbox]

            if from_addr:
                where.append("e.from_address = ?")
                params.append(from_addr)
            if to_addr:
                where.append(
                    "EXISTS (SELECT 1 FROM recipients r2 "
                    "WHERE r2.recipient_address = ? AND r2.email_id = e.message_id)"
                )
                params.append(to_addr)
            if sent_after:
                where.append("e.date >= ?")
                params.append(f"{sent_after} 00:00:00")
            if sent_before:
                where.append("e.date < ?")
                params.append(f"{sent_before} 00:00:00")

            sql = f"""
                SELECT
                    e.message_id,
                    e.date,
                    e.from_address,
                    e.subject,
                    snippet(emails_fts, -1, '<<', '>>', ' … ', 15) AS snippet
                FROM emails e
                JOIN emails_fts ON e.id = emails_fts.rowid
                WHERE {" AND ".join(where)}
                ORDER BY e.date DESC
                LIMIT ?
            """
            params.append(max_results)

            rows = conn.execute(sql, params).fetchall()
            conn.close()

            if not rows:
                return "No emails found matching your search."

            out = [f"Found {len(rows)} email(s):\n"]
            for msg_id, date, sender, subject, snippet in rows:
                out.append(
                    f"message_id : {msg_id}\n"
                    f"Date       : {date or '(unknown)'}\n"
                    f"From       : {sender or '(unknown)'}\n"
                    f"Subject    : {subject or '(no subject)'}\n"
                    f"Snippet    : {snippet}\n"
                    f"{'─' * 40}"
                )
            return "\n".join(out)

        except Exception as e:
            return f"Search error: {e}"

    def read_email(self, message_id: str) -> str:
        """
        Retrieve the full content of an email by its message_id.
        Use the message_id returned by search_inbox().

        Args:
            message_id: The message_id string from search_inbox results.

        Returns:
            Full email with headers (From, To, CC, Date, Subject) and body.
        """
        try:
            conn = self._conn()

            row = conn.execute(
                "SELECT message_id, date, subject, from_address, body "
                "FROM emails WHERE message_id = ?",
                (message_id,),
            ).fetchone()

            if not row:
                return f"No email found with message_id '{message_id}'."

            msg_id, date, subject, from_addr, body = row

            recips = conn.execute(
                "SELECT recipient_address, recipient_type " "FROM recipients WHERE email_id = ?",
                (message_id,),
            ).fetchall()
            conn.close()

            to_list = [a for a, t in recips if t == "to"]
            cc_list = [a for a, t in recips if t == "cc"]
            bcc_list = [a for a, t in recips if t == "bcc"]

            lines = [
                f"message_id : {msg_id}",
                f"Date       : {date or '(unknown)'}",
                f"From       : {from_addr or '(unknown)'}",
                f"To         : {', '.join(to_list) or '(none)'}",
            ]
            if cc_list:
                lines.append(f"CC         : {', '.join(cc_list)}")
            if bcc_list:
                lines.append(f"BCC        : {', '.join(bcc_list)}")
            lines += [
                f"Subject    : {subject or '(no subject)'}",
                "─" * 40,
                body or "(no body)",
            ]
            return "\n".join(lines)

        except Exception as e:
            return f"Error reading email: {e}"

    def list_senders(self, min_emails: int = 10) -> str:
        """
        List the most active senders in the database.
        Use this to discover valid from_addr values for search_inbox().

        Args:
            min_emails: Only show senders with at least this many emails.

        Returns:
            Formatted list of top senders and email counts.
        """
        try:
            conn = self._conn()
            rows = conn.execute(
                """
                SELECT from_address, COUNT(*) AS cnt
                FROM emails
                WHERE from_address != ''
                GROUP BY from_address
                HAVING cnt >= ?
                ORDER BY cnt DESC
                LIMIT 20
            """,
                (min_emails,),
            ).fetchall()
            conn.close()

            if not rows:
                return "No senders found with that minimum count."
            lines = ["Top senders:"]
            for sender, cnt in rows:
                lines.append(f"  {sender}: {cnt} emails")
            return "\n".join(lines)

        except Exception as e:
            return f"Error listing senders: {e}"

    # ── Enron DB builder ──────────────────────────────────────────────────────

    def _build_enron_db(self, db_path: str, max_rows: int = 50_000) -> ToolResult:
        """
        Build the Enron email SQLite database from corbt/enron-emails on HuggingFace.
        After building, the LangChain db cache is invalidated so the next
        generic query picks up the new tables.

        Schema:
            emails      — message_id, subject, from_address, date, body, file_name
            recipients  — email_id, recipient_address, recipient_type (to/cc/bcc)
            emails_fts  — FTS5 virtual table over subject + body

        Args:
            db_path:  Path to the SQLite file to create.
            max_rows: Rows to load before stopping (0 = all ~517k rows).

        Returns:
            ToolResult with success/failure message.
        """
        try:

            from datasets import Features, Sequence, Value, load_dataset

            logger.info(f"Building Enron DB → {db_path}")
            logger.info(
                f"Loading up to {max_rows:,} rows..." if max_rows else "Loading full dataset..."
            )

            expected_features = Features(
                {
                    "message_id": Value("string"),
                    "subject": Value("string"),
                    "from": Value("string"),
                    "to": Sequence(Value("string")),
                    "cc": Sequence(Value("string")),
                    "bcc": Sequence(Value("string")),
                    "date": Value("timestamp[us]"),
                    "body": Value("string"),
                    "file_name": Value("string"),
                }
            )

            dataset = load_dataset(
                "corbt/enron-emails",
                features=expected_features,
                split="train",
            )
            logger.info(f"Dataset has {len(dataset):,} total emails")

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.executescript(
                """
                DROP TABLE IF EXISTS recipients;
                DROP TABLE IF EXISTS emails_fts;
                DROP TABLE IF EXISTS emails;

                CREATE TABLE emails (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id    TEXT UNIQUE,
                    subject       TEXT,
                    from_address  TEXT,
                    date          TEXT,
                    body          TEXT,
                    file_name     TEXT
                );

                CREATE TABLE recipients (
                    email_id          TEXT,
                    recipient_address TEXT,
                    recipient_type    TEXT
                );
            """
            )
            conn.commit()

            conn.execute("PRAGMA synchronous = OFF;")
            conn.execute("PRAGMA journal_mode = MEMORY;")
            conn.execute("BEGIN TRANSACTION;")

            inserted = 0
            skipped = 0
            duplicates = 0

            for i, row in enumerate(dataset):
                if max_rows and i >= max_rows:
                    break

                body = row["body"] or ""
                from_address = row["from"] or ""
                subject = row["subject"] or ""
                to_list = [a for a in (row["to"] or []) if a]
                cc_list = [a for a in (row["cc"] or []) if a]
                bcc_list = [a for a in (row["bcc"] or []) if a]
                len(to_list) + len(cc_list) + len(bcc_list)

                # if len(body) > 5_000 or total_recip > 30:
                #     skipped += 1
                #     continue

                # key = (subject, body, from_address)
                # if key in seen:
                #     duplicates += 1
                #     continue
                # seen.add(key)

                date_obj = row["date"]
                date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S") if date_obj else ""

                cur.execute(
                    "INSERT INTO emails (message_id, subject, from_address, date, body, file_name) "
                    "VALUES (?,?,?,?,?,?)",
                    (row["message_id"], subject, from_address, date_str, body, row["file_name"]),
                )

                recip_rows = (
                    [(row["message_id"], a, "to") for a in to_list]
                    + [(row["message_id"], a, "cc") for a in cc_list]
                    + [(row["message_id"], a, "bcc") for a in bcc_list]
                )
                if recip_rows:
                    cur.executemany(
                        "INSERT INTO recipients (email_id, recipient_address, recipient_type) "
                        "VALUES (?,?,?)",
                        recip_rows,
                    )

                inserted += 1
                if inserted % 10_000 == 0:
                    logger.info(f"  {inserted:,} inserted…")

            conn.commit()

            logger.info("Creating indexes and FTS5 table…")
            cur.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_emails_from       ON emails(from_address);
                CREATE INDEX IF NOT EXISTS idx_emails_date       ON emails(date);
                CREATE INDEX IF NOT EXISTS idx_emails_message_id ON emails(message_id);

                CREATE INDEX IF NOT EXISTS idx_recip_address  ON recipients(recipient_address);
                CREATE INDEX IF NOT EXISTS idx_recip_type     ON recipients(recipient_type);
                CREATE INDEX IF NOT EXISTS idx_recip_email_id ON recipients(email_id);
                CREATE INDEX IF NOT EXISTS idx_recip_addr_eid ON recipients(recipient_address, email_id);

                CREATE VIRTUAL TABLE emails_fts USING fts5(
                    subject,
                    body,
                    content='emails',
                    content_rowid='id'
                );

                CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
                    INSERT INTO emails_fts(rowid, subject, body)
                    VALUES (new.id, new.subject, new.body);
                END;

                CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
                    DELETE FROM emails_fts WHERE rowid = old.id;
                END;

                CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
                    UPDATE emails_fts SET subject = new.subject, body = new.body
                    WHERE rowid = old.id;
                END;
            """
            )

            cur.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
            conn.commit()
            conn.close()

            self._db = None  # invalidate LangChain cache

            logger.info(f"✅ Done — {db_path}")
            logger.info(f"   Inserted  : {inserted:,}")
            logger.info(f"   Skipped   : {skipped:,}  (body too long / too many recipients)")
            logger.info(f"   Duplicates: {duplicates:,}")

            return ToolResult(
                success=True, output=f"Enron DB built at {db_path} ({inserted:,} emails)"
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    # ── Public convenience methods ────────────────────────────────────────────

    def build_enron_db(self, max_rows: int = 50_000) -> ToolResult:
        """
        Build the Enron email database from corbt/enron-emails on HuggingFace.
        Uses the instance db_uri — no uri needed.

            tool = SQLDatabaseTool("sqlite:///enron.db")
            tool.build_enron_db(max_rows=50_000)
        """
        return self.execute(action="build_enron_db", max_rows=max_rows)

    def create_from_dataset(
        self,
        dataset_name: str,
        table_name: str = "data",
        split: str = "train",
    ) -> ToolResult:
        """
        Create the database from a generic HuggingFace dataset.
        Uses the instance db_uri — no uri needed.

            tool = SQLDatabaseTool("sqlite:///biogrid.db")
            tool.create_from_dataset("qgallouedec/biogrid", table_name="interactions")
        """
        return self.execute(
            action="create_from_dataset",
            input=dataset_name,
            table_name=table_name,
            split=split,
        )

    def list_tables(self) -> ToolResult:
        """List all tables in the database."""
        return self.execute(action="sql_db_list_tables")

    def schema(self, tables: str = "") -> ToolResult:
        """Show schema for given tables (comma-separated) or all tables."""
        return self.execute(action="sql_db_schema", input=tables)

    # ── Universal query method ────────────────────────────────────────────────

    def query(self, sql_command: str) -> list:
        """
        Execute a read-only SQL query via LangChain. Universal — works with
        any db set at init. Note: use search_inbox / read_email for FTS5 queries
        against email DBs — this method does not support FTS5 MATCH syntax.

        Can be passed directly as a tool to any agentic trainer:

            tool    = SQLDatabaseTool("sqlite:///any.db")
            trainer = create_agentic_trainer(tools=[tool.query], ...)

        Args:
            sql_command: A read-only SQL query to execute.

        Returns:
            A list of tuples with query results, or an error dict on failure.
        """
        result = self.execute(action="sql_db_query", input=sql_command)
        if result.success:
            return result.output
        return {"error": result.error}

    def make_query_tool(self, name: str, description: str) -> callable:
        """
        Create a named query callable with a custom docstring.
        Use when you want a tool with a specific name visible in the model's tool schema.

            tool = SQLDatabaseTool("sqlite:///biogrid.db")

            query_biogrid = tool.make_query_tool(
                name        = "query_biogrid",
                description = "Query the BioGRID protein interaction database.",
            )
            trainer = create_agentic_trainer(tools=[query_biogrid], ...)

        Args:
            name:        Tool name the model sees in its schema.
            description: Description used in the tool schema.

        Returns:
            A named callable bound to this instance.
        """

        def _tool(sql_command: str) -> list:
            return self.query(sql_command)

        _tool.__name__ = name
        _tool.__qualname__ = name
        _tool.__doc__ = (
            f"{description}\n\n"
            "    Args:\n"
            "        sql_command: A read-only SQL query to execute.\n\n"
            "    Returns:\n"
            "        A list of tuples containing the query results.\n"
            "        Returns an error dict if the query fails.\n"
        )
        return _tool

    # ── Internal dataset loader ───────────────────────────────────────────────

    def _create_from_dataset(
        self,
        dataset_name: str,
        db_uri: str,
        table_name: str,
        split: str,
    ) -> ToolResult:
        try:
            from datasets import load_dataset

            if not db_uri.startswith("sqlite:///"):
                return ToolResult(
                    success=False,
                    output=None,
                    error="create_from_dataset only supports sqlite:/// URIs for now.",
                )

            db_path = db_uri.replace("sqlite:///", "")
            dataset = load_dataset(dataset_name, split=split)
            df = dataset.to_pandas()
            df.columns = [c.replace(" ", "_") for c in df.columns]

            conn = sqlite3.connect(db_path)
            try:
                df.to_sql(table_name, conn, if_exists="replace", index=False)
            finally:
                conn.close()

            self._db = None  # invalidate LangChain cache

            return ToolResult(
                success=True,
                output=(
                    f"Created '{table_name}' table in {db_path} "
                    f"with {len(df)} rows and columns: {list(df.columns)}"
                ),
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
