import sqlite3

from langchain_core.tools import tool
from src.config import RUNTIME


DB_PATH = RUNTIME / "agent_memory.sqlite3"


def _connect():
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories ("
        "key TEXT PRIMARY KEY, "
        "value TEXT NOT NULL)"
    )

    conn.commit()
    return conn


@tool
def save_memory(key: str, value: str) -> str:
    """Store one explicitly supplied personal fact by key and value, e.g. key='name', value='Alex'. Requires approval."""
    key = key.strip().lower()
    value = value.strip()

    if (
        not key
        or not value
        or len(key) > 80
        or len(value) > 500
    ):
        return "Invalid memory: provide a short nonempty key and value."

    with _connect() as conn:
        conn.execute(
            "INSERT INTO memories(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value)
        )

    return f"Saved memory for key: {key}"


@tool
def search_memory(query: str) -> str:
    """Look up previously saved user information in SQLite, e.g. search for 'name'."""
    term = query.strip().lower()

    if not term or len(term) > 80:
        return "Please provide a short search term."

    with _connect() as conn:
        rows = conn.execute(
            "SELECT key, value FROM memories "
            "WHERE key LIKE ? OR value LIKE ? "
            "LIMIT 10",
            (f"%{term}%", f"%{term}%")
        ).fetchall()

    if not rows:
        return "No matching memory found."

    return "\n".join(
        f"{key}: {value}"
        for key, value in rows
    )