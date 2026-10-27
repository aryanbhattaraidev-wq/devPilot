"""SQLite persistence for project tasks and notes.

Plain sqlite3 with a short-lived connection per call: the data is tiny, calls come
from a single server process, and it avoids sharing connections across threads.
Validation lives here (not in the MCP layer) so any caller gets the same rules.
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = ("todo", "in_progress", "blocked", "done")
PRIORITIES = ("low", "medium", "high", "critical")  # ordered lowest -> highest

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'todo'   CHECK (status IN ('todo','in_progress','blocked','done')),
    priority     TEXT NOT NULL DEFAULT 'medium' CHECK (priority IN ('low','medium','high','critical')),
    github_issue INTEGER,
    due_date     TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# Sort key: critical first, then most recently touched.
_PRIORITY_ORDER = (
    "CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END"
)


class NotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check(value: str | None, allowed: tuple[str, ...], field: str) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"{field} must be one of {', '.join(allowed)} (got {value!r})")


class TaskDB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with closing(self._conn()) as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def _write(self, sql: str, params: tuple = ()) -> int:
        with closing(self._conn()) as conn, conn:  # inner `with conn` = commit/rollback
            return conn.execute(sql, params).lastrowid or 0

    # ---- tasks -------------------------------------------------------------------------

    def create_task(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        github_issue: int | None = None,
        due_date: str | None = None,
    ) -> dict:
        if not title.strip():
            raise ValueError("title must not be empty")
        _check(priority, PRIORITIES, "priority")
        now = _now()
        task_id = self._write(
            "INSERT INTO tasks (title, description, priority, github_issue, due_date, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title.strip(), description, priority, github_issue, due_date, now, now),
        )
        return self.get_task(task_id)

    def get_task(self, task_id: int) -> dict:
        rows = self._query("SELECT * FROM tasks WHERE id = ?", (task_id,))
        if not rows:
            raise NotFound(f"task {task_id} does not exist")
        return {**rows[0], "notes": self.list_notes(task_id)}

    def list_tasks(
        self,
        status: str | None = None,
        priority: str | None = None,
        github_issue: int | None = None,
        include_done: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        _check(status, STATUSES, "status")
        _check(priority, PRIORITIES, "priority")
        where, params = [], []
        if status:
            where.append("status = ?")
            params.append(status)
        elif not include_done:
            where.append("status != 'done'")
        if priority:
            where.append("priority = ?")
            params.append(priority)
        if github_issue is not None:
            where.append("github_issue = ?")
            params.append(github_issue)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        return self._query(
            f"SELECT * FROM tasks {clause} ORDER BY {_PRIORITY_ORDER}, updated_at DESC LIMIT ?",
            (*params, max(1, min(limit, 200))),
        )

    def update_task(self, task_id: int, **changes: Any) -> dict:
        allowed = {"title", "description", "status", "priority", "github_issue", "due_date"}
        fields = {k: v for k, v in changes.items() if k in allowed and v is not None}
        _check(fields.get("status"), STATUSES, "status")
        _check(fields.get("priority"), PRIORITIES, "priority")
        self.get_task(task_id)  # raises NotFound
        if fields:
            sets = ", ".join(f"{k} = ?" for k in fields)
            self._write(
                f"UPDATE tasks SET {sets}, updated_at = ? WHERE id = ?",
                (*fields.values(), _now(), task_id),
            )
        return self.get_task(task_id)

    def delete_task(self, task_id: int) -> dict:
        task = self.get_task(task_id)
        self._write("DELETE FROM tasks WHERE id = ?", (task_id,))
        return task

    # ---- notes -------------------------------------------------------------------------

    def add_note(self, content: str, task_id: int | None = None) -> dict:
        if not content.strip():
            raise ValueError("content must not be empty")
        if task_id is not None:
            self.get_task(task_id)
        note_id = self._write(
            "INSERT INTO notes (task_id, content, created_at) VALUES (?, ?, ?)",
            (task_id, content.strip(), _now()),
        )
        return self._query("SELECT * FROM notes WHERE id = ?", (note_id,))[0]

    def list_notes(self, task_id: int | None = None, limit: int = 50) -> list[dict]:
        if task_id is None:
            return self._query("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,))
        return self._query("SELECT * FROM notes WHERE task_id = ? ORDER BY id", (task_id,))

    # ---- overview ----------------------------------------------------------------------

    def summary(self) -> dict:
        by_status = {r["status"]: r["n"] for r in self._query("SELECT status, COUNT(*) n FROM tasks GROUP BY status")}
        open_by_priority = {
            r["priority"]: r["n"]
            for r in self._query("SELECT priority, COUNT(*) n FROM tasks WHERE status != 'done' GROUP BY priority")
        }
        return {
            "by_status": {s: by_status.get(s, 0) for s in STATUSES},
            "open_by_priority": {p: open_by_priority.get(p, 0) for p in reversed(PRIORITIES)},
            "notes": self._query("SELECT COUNT(*) n FROM notes")[0]["n"],
        }
