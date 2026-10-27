"""Project MCP server (stdio): tasks, notes and priorities backed by SQLite.

Shows all three MCP server primitives side by side:

* **Tools**     -- actions the *model* decides to take (create/update tasks, add notes).
* **Resources** -- read-only data the *application* can load into context
                   (``project://summary``, ``project://tasks/{task_id}``).
* **Prompts**   -- reusable, user-invoked templates (``plan_next_work``).
"""

import json

from mcp.server import MCPServer
from mcp_types import ToolAnnotations
from mcp.server.mcpserver.exceptions import ToolError

from devpilot.config import Settings
from devpilot.db import NotFound, TaskDB

db = TaskDB(Settings().devpilot_db)

mcp = MCPServer(
    "project",
    instructions=(
        "Track the developer's own work: tasks (status todo/in_progress/blocked/done, priority "
        "low/medium/high/critical, optionally linked to a GitHub issue number) and free-form notes."
    ),
)


def _guard(fn, *args, **kwargs):
    """Expected failures become ToolError so the model can read the reason and adapt."""
    try:
        return fn(*args, **kwargs)
    except (ValueError, NotFound) as exc:
        raise ToolError(str(exc)) from exc


# ---- tools -------------------------------------------------------------------------------


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
def create_task(
    title: str,
    description: str = "",
    priority: str = "medium",
    github_issue: int | None = None,
    due_date: str | None = None,
) -> dict:
    """Create a task. priority: low|medium|high|critical. github_issue: linked issue number. due_date: YYYY-MM-DD."""
    return _guard(db.create_task, title, description, priority, github_issue, due_date)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def list_tasks(
    status: str | None = None,
    priority: str | None = None,
    github_issue: int | None = None,
    include_done: bool = False,
    limit: int = 50,
) -> list[dict]:
    """List tasks, highest priority first. Hides done tasks unless include_done or status='done'."""
    return _guard(db.list_tasks, status, priority, github_issue, include_done, limit)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def get_task(task_id: int) -> dict:
    """Get one task with all of its notes."""
    return _guard(db.get_task, task_id)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
def update_task(
    task_id: int,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    github_issue: int | None = None,
    due_date: str | None = None,
) -> dict:
    """Change any of a task's fields. Only the fields you pass are modified."""
    return _guard(
        db.update_task, task_id, title=title, description=description, status=status,
        priority=priority, github_issue=github_issue, due_date=due_date,
    )


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False))
def delete_task(task_id: int) -> dict:
    """Permanently delete a task and its notes. This cannot be undone."""
    return _guard(db.delete_task, task_id)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
def add_note(content: str, task_id: int | None = None) -> dict:
    """Add a free-form note, optionally attached to a task."""
    return _guard(db.add_note, content, task_id)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def project_summary() -> dict:
    """Counts of tasks by status and of open tasks by priority."""
    return db.summary()


# ---- resources ---------------------------------------------------------------------------


@mcp.resource("project://summary", mime_type="application/json")
def summary_resource() -> str:
    """Overview of the project's task counts."""
    return json.dumps(db.summary(), indent=2)


@mcp.resource("project://tasks/{task_id}", mime_type="application/json")
def task_resource(task_id: int) -> str:
    """A single task, including its notes."""
    return json.dumps(_guard(db.get_task, task_id), indent=2)


# ---- prompts -----------------------------------------------------------------------------


@mcp.prompt()
def plan_next_work(focus: str = "") -> str:
    """Ask the assistant to recommend what to work on next, using both tasks and GitHub."""
    extra = f" Focus on: {focus}." if focus else ""
    return (
        "Review my open GitHub issues and pull requests together with my own task list "
        "(including blocked and in-progress items). Tell me the three things I should work "
        "on next, in order, and why. Point out any GitHub issue that has no task yet and "
        f"any task whose priority looks wrong.{extra}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
