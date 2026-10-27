import json

import pytest

from devpilot.client.manager import MCPClientManager
from devpilot.config import AppConfig
from devpilot.db import NotFound, TaskDB


@pytest.fixture
def db(tmp_path):
    return TaskDB(tmp_path / "t.db")


def test_tasks_sorted_by_priority_and_done_hidden(db):
    db.create_task("low one", priority="low")
    crit = db.create_task("crit", priority="critical")
    db.create_task("high", priority="high")
    done = db.create_task("finished", priority="critical")
    db.update_task(done["id"], status="done")

    titles = [t["title"] for t in db.list_tasks()]
    assert titles == ["crit", "high", "low one"]
    assert crit["status"] == "todo"
    assert [t["title"] for t in db.list_tasks(status="done")] == ["finished"]
    assert len(db.list_tasks(include_done=True)) == 4


def test_update_changes_only_given_fields_and_bumps_timestamp(db):
    t = db.create_task("a", description="keep", priority="low")
    u = db.update_task(t["id"], status="in_progress", description=None)
    assert (u["status"], u["description"], u["priority"]) == ("in_progress", "keep", "low")
    assert u["updated_at"] >= t["updated_at"]


def test_validation_and_missing_rows(db):
    with pytest.raises(ValueError, match="priority"):
        db.create_task("x", priority="urgent")
    with pytest.raises(ValueError, match="title"):
        db.create_task("  ")
    with pytest.raises(ValueError, match="status"):
        db.list_tasks(status="nope")
    with pytest.raises(NotFound):
        db.get_task(999)
    with pytest.raises(NotFound):
        db.add_note("orphan", task_id=999)


def test_notes_attach_to_tasks_and_cascade_on_delete(db):
    t = db.create_task("a", github_issue=7)
    db.add_note("remember this", t["id"])
    assert db.get_task(t["id"])["notes"][0]["content"] == "remember this"
    assert [x["id"] for x in db.list_tasks(github_issue=7)] == [t["id"]]
    db.delete_task(t["id"])
    assert db.list_notes() == []


def test_summary_counts(db):
    db.create_task("a", priority="high")
    b = db.create_task("b")
    db.update_task(b["id"], status="blocked")
    s = db.summary()
    assert s["by_status"] == {"todo": 1, "in_progress": 0, "blocked": 1, "done": 0}
    assert s["open_by_priority"]["high"] == 1


# ---- the same thing through the real MCP server ---------------------------------------------


def project_manager(tmp_path) -> MCPClientManager:
    server = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "devpilot.servers.project_server"],
        "env": {"DEVPILOT_DB": str(tmp_path / "mcp.db")},
    }
    return MCPClientManager(AppConfig.model_validate({"servers": {"project": server}}))


async def test_project_server_exposes_tools_resources_and_prompts(tmp_path):
    async with project_manager(tmp_path) as m:
        state = m.servers()[0]
        assert {"create_task", "list_tasks", "update_task", "add_note"} <= {t.name for t in state.tools}
        assert [str(r.uri) for r in state.resources] == ["project://summary"]
        assert [t.uri_template for t in state.templates] == ["project://tasks/{task_id}"]
        assert [p.name for p in state.prompts] == ["plan_next_work"]

        created = await m.call_tool("project__create_task", {"title": "Fix login", "priority": "critical"})
        assert not created.is_error

        summary = json.loads(await m.read_resource("project", "project://summary"))
        assert summary["open_by_priority"]["critical"] == 1
        task = json.loads(await m.read_resource("project", "project://tasks/1"))
        assert task["title"] == "Fix login"

        prompt = await m.get_prompt("project", "plan_next_work", {"focus": "security"})
        assert "security" in prompt


async def test_server_errors_reach_the_model_as_readable_tool_errors(tmp_path):
    async with project_manager(tmp_path) as m:
        result = await m.call_tool("project__update_task", {"task_id": 42, "status": "done"})
        assert result.is_error
        assert "task 42 does not exist" in result.content[0].text
