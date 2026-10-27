import asyncio
from types import SimpleNamespace as NS

import pytest
from mcp import types
from test_agent import FakeLLM, call, reply

from devpilot.agent import Agent
from devpilot.approval import (
    ApprovalDecision,
    ApprovalGate,
    approve_all,
    assess,
    deny_all,
)
from devpilot.client.manager import MCPClientManager, ToolRef
from devpilot.config import AppConfig
from devpilot.db import TaskDB


def tool(name="t", **hints) -> types.Tool:
    return types.Tool(name=name, input_schema={"type": "object"}, annotations=types.ToolAnnotations(**hints))


def unannotated(name="t") -> types.Tool:
    return types.Tool(name=name, input_schema={"type": "object"})


# ---- policy ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hints, needs, destructive",
    [
        (dict(read_only_hint=True), False, False),
        (dict(read_only_hint=True, open_world_hint=True), False, False),  # reading the web is fine
        (dict(read_only_hint=False, destructive_hint=True), True, True),
        (dict(read_only_hint=False), True, True),  # spec default: writes are destructive
        (dict(read_only_hint=False, destructive_hint=False, open_world_hint=True), True, False),
        (dict(read_only_hint=False, destructive_hint=False), True, False),  # spec default: open world
        (dict(read_only_hint=False, destructive_hint=False, open_world_hint=False), False, False),
    ],
)
def test_policy_follows_annotations_with_pessimistic_defaults(hints, needs, destructive):
    risk = assess(tool(**hints))
    assert (risk.requires_approval, risk.destructive) == (needs, destructive)


def test_unannotated_tool_is_treated_as_dangerous():
    risk = assess(unannotated())
    assert risk.requires_approval and risk.destructive


# ---- gate -----------------------------------------------------------------------------------


def stub_manager(**tools: types.Tool):
    refs = [ToolRef(f"srv__{n}", "srv", t) for n, t in tools.items()]
    return NS(tools=lambda: refs)


EXTERNAL = dict(read_only_hint=False, destructive_hint=False, open_world_hint=True)
DESTRUCTIVE = dict(read_only_hint=False, destructive_hint=True)


async def test_read_only_never_asks():
    asked = []

    async def approver(req):
        asked.append(req)
        return ApprovalDecision(True)

    gate = ApprovalGate(stub_manager(read=tool(read_only_hint=True)), approver)
    assert (await gate.check("srv__read", {})).allowed
    assert asked == []


async def test_request_carries_what_the_user_needs_to_decide():
    seen = []

    async def approver(req):
        seen.append(req)
        return ApprovalDecision(False, note="nope")

    gate = ApprovalGate(stub_manager(post=tool(**EXTERNAL)), approver)
    result = await gate.check("srv__post", {"title": "x"})
    assert (result.allowed, result.asked, result.note) == (False, True, "nope")
    assert (seen[0].tool_name, seen[0].server, seen[0].arguments) == ("srv__post", "srv", {"title": "x"})
    assert "outside DevPilot" in seen[0].reason and not seen[0].destructive


async def test_default_gate_denies_and_unknown_tools_pass_through():
    gate = ApprovalGate(stub_manager(post=tool(**EXTERNAL)))  # default approver = deny_all
    assert not (await gate.check("srv__post", {})).allowed
    assert (await gate.check("srv__missing", {})).allowed  # manager reports unknown tools itself


async def test_remember_applies_to_external_writes_but_never_to_destructive():
    asked = []

    async def approver(req):
        asked.append(req.tool_name)
        return ApprovalDecision(True, remember=True)

    gate = ApprovalGate(stub_manager(post=tool(**EXTERNAL), wipe=tool(**DESTRUCTIVE)), approver)
    for _ in range(3):
        assert (await gate.check("srv__post", {})).allowed
        assert (await gate.check("srv__wipe", {})).allowed
    assert asked == ["srv__post"] + ["srv__wipe"] * 3


async def test_approver_failure_fails_closed():
    async def broken(req):
        raise RuntimeError("UI crashed")

    gate = ApprovalGate(stub_manager(post=tool(**EXTERNAL)), broken)
    result = await gate.check("srv__post", {})
    assert not result.allowed and "RuntimeError" in result.note


async def test_parallel_requests_are_asked_one_at_a_time():
    active = peak = 0

    async def approver(req):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return ApprovalDecision(True)

    gate = ApprovalGate(stub_manager(wipe=tool(**DESTRUCTIVE)), approver)
    await asyncio.gather(*(gate.check("srv__wipe", {}) for _ in range(4)))
    assert peak == 1


# ---- end to end: model -> gate -> real project server -> SQLite -------------------------------


def project_manager(tmp_path) -> MCPClientManager:
    server = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "devpilot.servers.project_server"],
        "env": {"DEVPILOT_DB": str(tmp_path / "p.db")},
    }
    return MCPClientManager(AppConfig.model_validate({"servers": {"project": server}}))


def seeded_db(tmp_path) -> TaskDB:
    db = TaskDB(tmp_path / "p.db")
    db.create_task("precious work")
    return db


async def test_server_annotations_reach_the_host_policy(tmp_path):
    async with project_manager(tmp_path) as m:
        risk = {r.tool.name: assess(r.tool) for r in m.tools()}
        assert not risk["list_tasks"].requires_approval
        assert not risk["create_task"].requires_approval  # local + non-destructive
        assert risk["delete_task"].requires_approval and risk["delete_task"].destructive


async def test_denied_delete_does_not_touch_the_database_and_model_is_told(tmp_path):
    db = seeded_db(tmp_path)
    llm = FakeLLM(reply(None, call("c1", "project__delete_task", task_id=1)), reply("Okay, left it alone."))
    async with project_manager(tmp_path) as m:
        agent = Agent(m, llm, "test", gate=ApprovalGate(m, deny_all))
        result = await agent.ask("delete task 1")

    assert result.tool_calls[0].denied and result.tool_calls[0].is_error
    assert [t["title"] for t in db.list_tasks()] == ["precious work"]  # still there
    told = next(x for x in llm.requests[1]["messages"] if x["role"] == "tool")["content"]
    assert told.startswith("DENIED") and "NOT executed" in told


async def test_approved_delete_runs(tmp_path):
    db = seeded_db(tmp_path)
    llm = FakeLLM(reply(None, call("c1", "project__delete_task", task_id=1)), reply("Deleted."))
    async with project_manager(tmp_path) as m:
        result = await Agent(m, llm, "test", gate=ApprovalGate(m, approve_all)).ask("delete task 1")
    assert not result.tool_calls[0].denied
    assert db.list_tasks(include_done=True) == []


async def test_agent_default_gate_is_deny_all(tmp_path):
    db = seeded_db(tmp_path)
    llm = FakeLLM(reply(None, call("c1", "project__delete_task", task_id=1)), reply("ok"))
    async with project_manager(tmp_path) as m:
        result = await Agent(m, llm, "test").ask("delete task 1")  # no gate given
    assert result.tool_calls[0].denied
    assert len(db.list_tasks()) == 1


async def test_read_and_local_write_tools_need_no_approval(tmp_path):
    llm = FakeLLM(
        reply(None, call("c1", "project__create_task", title="new"), call("c2", "project__list_tasks")),
        reply("done"),
    )
    async with project_manager(tmp_path) as m:
        result = await Agent(m, llm, "test", gate=ApprovalGate(m, deny_all)).ask("add a task and list")
    assert not any(c.denied or c.is_error for c in result.tool_calls)
