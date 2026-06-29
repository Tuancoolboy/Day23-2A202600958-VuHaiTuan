"""Checkpointer adapter."""

from __future__ import annotations

from typing import Any


class _MemoryCheckpointerFallback:
    """No-op checkpointer used when langgraph is unavailable locally."""

    storage: dict[str, Any]

    def __init__(self) -> None:
        self.storage = {}


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    TODO(student): implement SQLite support for the persistence extension track.
    The starter provides MemorySaver only — SQLite/Postgres are extension tasks.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError:
            return _MemoryCheckpointerFallback()

        return MemorySaver()
    if kind == "sqlite":
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("Install sqlite support: pip install '.[sqlite]'") from exc
        if not database_url:
            database_url = "checkpoints.sqlite"
        conn = sqlite3.connect(database_url, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn)
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
