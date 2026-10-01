"""
agenttune.rag.storage — persistent trajectory storage for RAG agents.

Stores all trajectories (questions, tool calls, retrieved text, final answers,
rewards, step-level data) in a SQLite database for later analysis, dataset
construction, and curriculum sampling.
"""

from .trajectory_store import (
    TrajectoryRecord,
    TrajectoryStore,
    make_trajectory_callback,
)

__all__ = [
    "TrajectoryStore",
    "TrajectoryRecord",
    "make_trajectory_callback",
]
