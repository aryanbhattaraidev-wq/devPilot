"""HTTP API for DevPilot: the backend a web UI talks to.

    GET  /servers                              connected MCP servers and their tools
    POST /sessions                             start a conversation
    POST /sessions/{id}/messages               ask a question (runs in the background)
    GET  /sessions/{id}/events?after=N         what happened since event N (polling)
    POST /sessions/{id}/approvals/{approval}   approve / deny a pending sensitive action

One MCP client manager is shared by every session (servers connect once, at startup);
each session has its own agent, conversation history and approval gate.

Why polling events instead of a single response? A turn can pause midway waiting for a
human to approve an action, so the UI must see "approval needed" while the request that
caused it is still running. The events list is an append-only log; the UI just asks for
everything after the last event it has seen.

There is no authentication: it is meant to run on localhost for a single user.
"""

import asyncio
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from devpilot.agent import Agent, make_client
from devpilot.approval import ApprovalDecision, ApprovalGate, ApprovalRequest
from devpilot.client.manager import MCPClientManager, ServerStatus
from devpilot.config import Settings, load_config

APPROVAL_TIMEOUT_SECONDS = 300


class Session:
    def __init__(self, agent: Agent):
        self.id = uuid.uuid4().hex[:12]
        self.agent = agent
        self.events: list[dict[str, Any]] = []
        self.pending: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self.busy = False

    def emit(self, type: str, **data: Any) -> None:
        self.events.append({"id": len(self.events), "type": type, **data})

    async def approver(self, req: ApprovalRequest) -> ApprovalDecision:
        approval_id = uuid.uuid4().hex[:8]
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self.pending[approval_id] = future
        self.emit(
            "approval_request",
            approval_id=approval_id,
            tool=req.tool_name,
            server=req.server,
            arguments=req.arguments,
            reason=req.reason,
            destructive=req.destructive,
        )
        try:
            return await asyncio.wait_for(future, APPROVAL_TIMEOUT_SECONDS)
        except TimeoutError:
            return ApprovalDecision(False, note="the approval request timed out")
        finally:
            self.pending.pop(approval_id, None)

    async def run_turn(self, text: str) -> None:
        try:
            result = await self.agent.ask(text)
            self.emit(
                "answer",
                text=result.text,
                tool_calls=[
                    {"name": c.name, "arguments": c.arguments, "is_error": c.is_error, "denied": c.denied}
                    for c in result.tool_calls
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI instead of hanging it
            self.emit("error", text=f"{type(exc).__name__}: {exc}")
        finally:
            self.busy = False


class MessageIn(BaseModel):
    text: str


class ApprovalIn(BaseModel):
    approved: bool
    remember: bool = False


def create_app(
    config_path: str = "servers.json",
    client_factory: Callable[[], Any] | None = None,
) -> FastAPI:
    settings = Settings()
    make_llm = client_factory or (lambda: make_client(settings.llm_api_key, settings.llm_base_url))
    sessions: dict[str, Session] = {}
    background: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with MCPClientManager(load_config(config_path)) as manager:
            app.state.manager = manager
            yield

    app = FastAPI(title="DevPilot", lifespan=lifespan)

    def get_session(session_id: str) -> Session:
        if session_id not in sessions:
            raise HTTPException(404, "unknown session")
        return sessions[session_id]

    @app.get("/servers")
    async def servers() -> list[dict]:
        return [
            {
                "name": s.name,
                "transport": s.transport,
                "status": s.status.value,
                "error": s.error,
                "tools": [t.name for t in s.tools] if s.status is ServerStatus.CONNECTED else [],
                "resources": len(s.resources) + len(s.templates),
                "prompts": [p.name for p in s.prompts],
            }
            for s in app.state.manager.servers()
        ]

    @app.post("/sessions")
    async def create_session() -> dict:
        manager: MCPClientManager = app.state.manager
        holder: dict[str, Session] = {}

        async def approver(req: ApprovalRequest) -> ApprovalDecision:
            return await holder["session"].approver(req)

        agent = Agent(
            manager,
            make_llm(),
            settings.devpilot_model,
            default_repo=settings.github_repo,
            on_tool_call=lambda name, args: holder["session"].emit("tool_call", name=name, arguments=args),
            on_tool_result=lambda name, text, err: holder["session"].emit(
                "tool_result", name=name, chars=len(text), is_error=err
            ),
            gate=ApprovalGate(manager, approver),
        )
        session = holder["session"] = Session(agent)
        sessions[session.id] = session
        return {"session_id": session.id}

    @app.post("/sessions/{session_id}/messages", status_code=202)
    async def send_message(session_id: str, body: MessageIn) -> dict:
        session = get_session(session_id)
        if session.busy:
            raise HTTPException(409, "the previous question is still being answered")
        if not body.text.strip():
            raise HTTPException(422, "empty message")
        session.busy = True
        session.emit("user", text=body.text)
        task = asyncio.create_task(session.run_turn(body.text))
        background.add(task)  # keep a reference so the task isn't garbage-collected mid-run
        task.add_done_callback(background.discard)
        return {"status": "started"}

    @app.get("/sessions/{session_id}/events")
    async def events(session_id: str, after: int = -1) -> dict:
        session = get_session(session_id)
        return {
            "events": [e for e in session.events if e["id"] > after],
            "busy": session.busy,
        }

    @app.post("/sessions/{session_id}/approvals/{approval_id}")
    async def decide(session_id: str, approval_id: str, body: ApprovalIn) -> dict:
        future = get_session(session_id).pending.get(approval_id)
        if future is None or future.done():
            raise HTTPException(404, "no such pending approval (already answered or timed out)")
        future.set_result(
            ApprovalDecision(body.approved, remember=body.remember, note="" if body.approved else "the user declined")
        )
        return {"status": "recorded"}

    return app
