"""
Trajectory database — persistent storage for all RAG agent trajectories.

Stores the full trace of every rollout: question, gold answer, tool calls
(queries + retrieved text), final answer, reward, reward components, step-level
data (thoughts, observations, tool results), and metadata (model, condition,
training step, timestamp).

Uses SQLite (stdlib, zero new dependency) — same pattern as the FTS5 retrieval
backend. The store is pickle-safe (stores only the file path, opens connections
lazily) so it can be passed through the rollout pipeline like the other backends.

Two usage modes:
  1. As a post_step_hook / on_trajectory_end callback (like TraceLogger) —
     captures every trajectory during training or eval.
  2. As a standalone query API — retrieve trajectories by question, condition,
     reward range, etc. for analysis or dataset construction.

Schema:
  - trajectories: one row per rollout (question, gold, answer, reward, etc.)
  - steps: one row per step within a trajectory (thought, action, observation)
  - tool_calls: one row per tool call (query, result, tool name)
  - runs: metadata about the training/eval run (model, config, timestamp)
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryRecord:
    """One complete trajectory for storage."""

    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    question: str = ""
    gold_answer: str = ""
    final_answer: str = ""
    reward: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)
    n_tool_calls: int = 0
    has_answer_tag: bool = False
    condition: str = ""  # plain / m1 / recent_k
    model: str = ""
    training_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    steps: list[dict[str, Any]] = field(default_factory=list)


class TrajectoryStore:
    """SQLite-backed persistent store for RAG agent trajectories.

    Pickle-safe: stores only the db_path, opens connections lazily (same pattern
    as SQLiteFTSBackend). Thread-safe via check_same_thread=False.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                run_type TEXT,          -- 'training' or 'eval'
                model TEXT,
                condition TEXT,
                reward_stack TEXT,
                config_json TEXT,
                created_at REAL,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                run_id TEXT,
                question TEXT,
                gold_answer TEXT,
                final_answer TEXT,
                reward REAL,
                reward_components_json TEXT,
                n_tool_calls INTEGER,
                has_answer_tag INTEGER,
                condition TEXT,
                model TEXT,
                training_step INTEGER,
                metadata_json TEXT,
                timestamp REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trajectory_id TEXT,
                step_number INTEGER,
                thought TEXT,
                action_name TEXT,
                action_args_json TEXT,
                observation TEXT,
                reward REAL,
                is_tool_step INTEGER,
                is_terminal INTEGER,
                FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
            );

            CREATE TABLE IF NOT EXISTS tool_calls (
                tool_call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trajectory_id TEXT,
                step_number INTEGER,
                tool_name TEXT,
                query TEXT,
                result TEXT,
                FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
            );

            CREATE INDEX IF NOT EXISTS idx_traj_run ON trajectories(run_id);
            CREATE INDEX IF NOT EXISTS idx_traj_question ON trajectories(question);
            CREATE INDEX IF NOT EXISTS idx_traj_reward ON trajectories(reward);
            CREATE INDEX IF NOT EXISTS idx_traj_condition ON trajectories(condition);
            CREATE INDEX IF NOT EXISTS idx_steps_traj ON steps(trajectory_id);
            CREATE INDEX IF NOT EXISTS idx_tc_traj ON tool_calls(trajectory_id);
        """
        )
        conn.commit()
        conn.close()

    def register_run(
        self,
        run_id: str,
        run_type: str,
        model: str,
        condition: str = "",
        reward_stack: str = "",
        config: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Register a training or eval run. Returns the run_id."""
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, run_type, model, condition, reward_stack, config_json, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                run_type,
                model,
                condition,
                reward_stack,
                json.dumps(config or {}),
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        conn.close()
        return run_id

    def store(self, record: TrajectoryRecord) -> str:
        """Store a complete trajectory with all its steps and tool calls."""
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO trajectories "
                "(trajectory_id, run_id, question, gold_answer, final_answer, reward, "
                "reward_components_json, n_tool_calls, has_answer_tag, condition, model, "
                "training_step, metadata_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.trajectory_id,
                    record.run_id,
                    record.question,
                    record.gold_answer,
                    record.final_answer,
                    record.reward,
                    json.dumps(record.reward_components),
                    record.n_tool_calls,
                    int(record.has_answer_tag),
                    record.condition,
                    record.model,
                    record.training_step,
                    json.dumps(record.metadata),
                    record.timestamp,
                ),
            )
            for step in record.steps:
                conn.execute(
                    "INSERT INTO steps "
                    "(trajectory_id, step_number, thought, action_name, action_args_json, "
                    "observation, reward, is_tool_step, is_terminal) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.trajectory_id,
                        step.get("step_number", 0),
                        step.get("thought", ""),
                        step.get("action_name", ""),
                        json.dumps(step.get("action_args", {})),
                        step.get("observation", ""),
                        step.get("reward"),
                        int(step.get("is_tool_step", False)),
                        int(step.get("is_terminal", False)),
                    ),
                )
                if step.get("tool_calls"):
                    for tc in step["tool_calls"]:
                        conn.execute(
                            "INSERT INTO tool_calls "
                            "(trajectory_id, step_number, tool_name, query, result) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (
                                record.trajectory_id,
                                step.get("step_number", 0),
                                tc.get("tool_name", ""),
                                tc.get("query", ""),
                                tc.get("result", ""),
                            ),
                        )
            conn.commit()
        finally:
            conn.close()
        return record.trajectory_id

    # ── Query API ─────────────────────────────────────────────────────────

    def get_by_question(self, question: str, limit: int = 100) -> list[dict]:
        """Retrieve all trajectories for a given question."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM trajectories WHERE question = ? ORDER BY timestamp DESC LIMIT ?",
            (question, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_by_run(self, run_id: str, limit: int = 1000) -> list[dict]:
        """Retrieve all trajectories for a given run."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM trajectories WHERE run_id = ? ORDER BY timestamp LIMIT ?",
            (run_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_by_reward_range(
        self, min_reward: float, max_reward: float, condition: str = None, limit: int = 100
    ) -> list[dict]:
        """Retrieve trajectories within a reward range (for curriculum/dataset building)."""
        conn = self._conn()
        if condition:
            rows = conn.execute(
                "SELECT * FROM trajectories WHERE reward >= ? AND reward <= ? AND condition = ? "
                "ORDER BY reward DESC LIMIT ?",
                (min_reward, max_reward, condition, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trajectories WHERE reward >= ? AND reward <= ? "
                "ORDER BY reward DESC LIMIT ?",
                (min_reward, max_reward, limit),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_steps(self, trajectory_id: str) -> list[dict]:
        """Retrieve all steps for a trajectory."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM steps WHERE trajectory_id = ? ORDER BY step_number", (trajectory_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_tool_calls(self, trajectory_id: str) -> list[dict]:
        """Retrieve all tool calls for a trajectory."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM tool_calls WHERE trajectory_id = ? ORDER BY tool_call_id",
            (trajectory_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_full_trajectory(self, trajectory_id: str) -> dict | None:
        """Retrieve a complete trajectory with steps and tool calls."""
        conn = self._conn()
        traj = conn.execute(
            "SELECT * FROM trajectories WHERE trajectory_id = ?", (trajectory_id,)
        ).fetchone()
        if traj is None:
            conn.close()
            return None
        result = dict(traj)
        result["reward_components"] = json.loads(result.get("reward_components_json", "{}"))
        result["metadata"] = json.loads(result.get("metadata_json", "{}"))
        result.pop("reward_components_json", None)
        result.pop("metadata_json", None)
        result["steps"] = self.get_steps(trajectory_id)
        result["tool_calls"] = self.get_tool_calls(trajectory_id)
        conn.close()
        return result

    def stats(self) -> dict[str, Any]:
        """Summary statistics for the store."""
        conn = self._conn()
        n_traj = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        n_steps = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        n_tc = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        avg_reward = conn.execute("SELECT AVG(reward) FROM trajectories").fetchone()[0] or 0.0
        avg_calls = conn.execute("SELECT AVG(n_tool_calls) FROM trajectories").fetchone()[0] or 0.0
        conn.close()
        return {
            "total_trajectories": n_traj,
            "total_steps": n_steps,
            "total_tool_calls": n_tc,
            "total_runs": n_runs,
            "avg_reward": round(avg_reward, 4),
            "avg_tool_calls": round(avg_calls, 2),
        }

    def export_jsonl(self, output_path: str, run_id: str = None, condition: str = None):
        """Export trajectories as JSONL (one line per trajectory with full data)."""
        conn = self._conn()
        if run_id:
            rows = conn.execute(
                "SELECT trajectory_id FROM trajectories WHERE run_id = ? ORDER BY timestamp",
                (run_id,),
            ).fetchall()
        elif condition:
            rows = conn.execute(
                "SELECT trajectory_id FROM trajectories WHERE condition = ? ORDER BY timestamp",
                (condition,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT trajectory_id FROM trajectories ORDER BY timestamp"
            ).fetchall()
        conn.close()
        with open(output_path, "w") as f:
            for row in rows:
                full = self.get_full_trajectory(row["trajectory_id"])
                if full:
                    f.write(json.dumps(full, default=str) + "\n")
        logger.info(f"[trajectory_store] exported {len(rows)} trajectories to {output_path}")


def make_trajectory_callback(
    store: TrajectoryStore,
    run_id: str,
    model: str,
    condition: str = "",
    reward_fn=None,
    gold_by_question: dict[str, str] | None = None,
):
    """Create a callback that stores trajectories during training/eval.

    Usage: pass as on_trajectory_end to create_rollout_fn, or call directly
    with a Trajectory object. Mirrors TraceLogger's interface so it's a drop-in
    alongside the existing trace.jsonl logging.
    """
    from agenttune.rag.trajectory_utils import extract_question_text, extract_tool_calls

    def _callback(trajectory) -> None:
        question = extract_question_text(trajectory.task)
        gold = ""
        if gold_by_question:
            gold = gold_by_question.get(question, "")
        tool_call_count = (
            trajectory.metadata.get("tool_call_count", 0) if trajectory.metadata else 0
        )

        # Build step records from the trajectory
        steps_data = []
        for s in trajectory.steps:
            step_dict = {
                "step_number": s.step_number,
                "thought": getattr(s, "thought", "") or "",
                "action_name": s.action.get("name", "") if s.action else "",
                "action_args": s.action.get("arguments", {}) if s.action else {},
                "observation": s.observation or "",
                "reward": s.reward,
                "is_tool_step": bool(s.action and s.action.get("name")),
                "is_terminal": (trajectory.metadata or {}).get("is_terminal", False),
                "tool_calls": [],
            }
            # Extract tool calls from this step
            for call in extract_tool_calls(s):
                step_dict["tool_calls"].append(
                    {
                        "tool_name": call.get("name", ""),
                        "query": str(call.get("arguments", "")),
                        "result": s.observation or "",
                    }
                )
            steps_data.append(step_dict)

        # Compute reward if reward_fn provided
        reward = trajectory.reward or 0.0
        reward_components = {}
        if reward_fn:
            try:
                reward = reward_fn(
                    completions=[trajectory.final_response],
                    prompts=[question],
                    gold_answer=[gold],
                    tool_call_counts=[tool_call_count],
                )[0]
                from agenttune.rag.rewards.phase1_rewards import get_last_component_scores

                comp = get_last_component_scores()
                reward_components = (
                    {k: (v[0] if v else 0.0) for k, v in comp.items()} if comp else {}
                )
            except Exception:
                pass

        record = TrajectoryRecord(
            run_id=run_id,
            question=question,
            gold_answer=gold,
            final_answer=trajectory.final_response or "",
            reward=reward,
            reward_components=reward_components,
            n_tool_calls=tool_call_count,
            has_answer_tag="<answer>" in (trajectory.final_response or "").lower(),
            condition=condition,
            model=model,
            steps=steps_data,
        )
        store.store(record)

    return _callback
