import json
import time

import pytest
from fastapi.testclient import TestClient
from test_agent import FakeLLM, call, reply

from devpilot.api import create_app
from devpilot.db import TaskDB


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "api.db"
    db = TaskDB(db_path)
    db.create_task("precious work")
    config = {
        "servers": {
            "project": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "devpilot.servers.project_server"],
                "env": {"DEVPILOT_DB": str(db_path)},
            }
        }
    }
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(config))
    return path, db


def make_client(config_path, *responses):
    # Every session gets its own scripted LLM.
    return TestClient(create_app(str(config_path), client_factory=lambda: FakeLLM(*responses)))


def wait_for(client, sid, until, after=-1, timeout=30):
    """Poll the events endpoint like the UI does, until `until(events, busy)` holds."""
    deadline = time.monotonic() + timeout
    seen: list[dict] = []
    while time.monotonic() < deadline:
        data = client.get(f"/sessions/{sid}/events", params={"after": after}).json()
        seen.extend(data["events"])
        after = seen[-1]["id"] if seen else after
        if until(seen, data["busy"]):
            return seen
        time.sleep(0.05)
    raise AssertionError(f"timed out; events so far: {[e['type'] for e in seen]}")


def start(client, text="go"):
    sid = client.post("/sessions").json()["session_id"]
    assert client.post(f"/sessions/{sid}/messages", json={"text": text}).status_code == 202
    return sid


def test_servers_endpoint_lists_connected_servers_and_tools(env):
    config, _ = env
    with make_client(config) as client:
        [server] = client.get("/servers").json()
        assert server["status"] == "connected" and "delete_task" in server["tools"]
        assert server["prompts"] == ["plan_next_work"]


def test_a_turn_streams_user_tool_and_answer_events(env):
    config, _ = env
    with make_client(config, reply(None, call("c1", "project__list_tasks")), reply("You have one task.")) as client:
        sid = start(client, "what's on my list?")
        events = wait_for(client, sid, lambda ev, busy: not busy)
        assert [e["type"] for e in events] == ["user", "tool_call", "tool_result", "answer"]
        assert events[-1]["text"] == "You have one task."
        assert events[-1]["tool_calls"][0]["name"] == "project__list_tasks"


def approval_flow(config, decision):
    llm_script = (reply(None, call("c1", "project__delete_task", task_id=1)), reply("All done."))
    with make_client(config, *llm_script) as client:
        sid = start(client, "delete task 1")
        events = wait_for(client, sid, lambda ev, busy: any(e["type"] == "approval_request" for e in ev))
        request = events[-1]
        assert request["tool"] == "project__delete_task" and request["destructive"]
        assert client.get(f"/sessions/{sid}/events").json()["busy"]  # paused, waiting for the human

        ok = client.post(f"/sessions/{sid}/approvals/{request['approval_id']}", json=decision)
        assert ok.status_code == 200
        # An approval can only be answered once.
        again = client.post(f"/sessions/{sid}/approvals/{request['approval_id']}", json=decision)
        assert again.status_code == 404
        return wait_for(client, sid, lambda ev, busy: not busy, after=request["id"])


def test_approving_over_http_runs_the_tool(env):
    config, db = env
    events = approval_flow(config, {"approved": True})
    assert events[-1]["type"] == "answer" and not events[-1]["tool_calls"][0]["denied"]
    assert db.list_tasks(include_done=True) == []


def test_denying_over_http_leaves_data_untouched(env):
    config, db = env
    events = approval_flow(config, {"approved": False})
    assert events[-1]["tool_calls"][0]["denied"]
    assert [t["title"] for t in db.list_tasks()] == ["precious work"]


def test_bad_requests(env):
    config, _ = env
    with make_client(config, reply("hi")) as client:
        assert client.get("/sessions/nope/events").status_code == 404
        sid = client.post("/sessions").json()["session_id"]
        assert client.post(f"/sessions/{sid}/messages", json={"text": "  "}).status_code == 422


def test_second_question_while_busy_is_rejected(env):
    config, _ = env
    script = (reply(None, call("c1", "project__delete_task", task_id=1)), reply("ok"))
    with make_client(config, *script) as client:
        sid = start(client, "delete it")
        wait_for(client, sid, lambda ev, busy: any(e["type"] == "approval_request" for e in ev))
        assert client.post(f"/sessions/{sid}/messages", json={"text": "another"}).status_code == 409
