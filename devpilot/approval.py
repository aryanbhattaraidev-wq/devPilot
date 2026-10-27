"""Human approval for sensitive tool calls.

Who decides what needs approval? Not the server: a server could lie or be compromised.
Servers only *describe* each tool with annotations (``readOnlyHint``, ``destructiveHint``,
``openWorldHint``); the **host** turns those hints into policy and enforces it. The MCP
spec's defaults are deliberately pessimistic -- a tool that says nothing is assumed to be
destructive and to touch the outside world -- so forgetting an annotation fails safe.

Policy:
    read-only                          -> run
    destructive (or unannotated)       -> ask every time
    non-destructive write, external    -> ask (visible to others, e.g. creating a GitHub issue)
    non-destructive write, local only  -> run (e.g. adding a task to your own database)

The gate sits in the agent loop, between "the model asked for X" and "X runs", and it
fails closed: any error while asking counts as a denial.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mcp import types

from devpilot.client.manager import MCPClientManager


@dataclass(frozen=True)
class Risk:
    requires_approval: bool
    destructive: bool
    reason: str


def assess(tool: types.Tool) -> Risk:
    a = tool.annotations
    if a is None:
        return Risk(True, True, "the server gave no safety annotations, so it is treated as potentially destructive")
    if a.read_only_hint:
        return Risk(False, False, "read-only")
    if a.destructive_hint is None or a.destructive_hint:  # spec default for writes: destructive
        return Risk(True, True, "this action can destroy or overwrite data")
    if a.open_world_hint is None or a.open_world_hint:  # spec default: talks to the outside world
        return Risk(True, False, "this action changes something outside DevPilot that others can see")
    return Risk(False, False, "local, non-destructive change")


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str  # qualified, e.g. github__create_issue
    server: str
    arguments: dict
    reason: str
    destructive: bool


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    remember: bool = False  # "always allow this tool for the rest of the session"
    note: str = ""  # optional explanation passed back to the model


Approver = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


async def deny_all(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(False, note="no interactive approver is available")


async def approve_all(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(True)


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    asked: bool = False
    note: str = ""


class ApprovalGate:
    def __init__(self, manager: MCPClientManager, approver: Approver = deny_all):
        self.manager = manager
        self.approver = approver
        self._lock = asyncio.Lock()  # parallel tool calls must not stack up several prompts
        self._always: set[str] = set()

    async def check(self, qualified_name: str, arguments: dict) -> GateResult:
        ref = next((r for r in self.manager.tools() if r.qualified_name == qualified_name), None)
        if ref is None:
            return GateResult(True)  # unknown tool: the manager will report the error itself
        risk = assess(ref.tool)
        if not risk.requires_approval:
            return GateResult(True)

        async with self._lock:
            if qualified_name in self._always and not risk.destructive:
                return GateResult(True)
            request = ApprovalRequest(qualified_name, ref.server, arguments, risk.reason, risk.destructive)
            try:
                decision = await self.approver(request)
            except Exception as exc:  # noqa: BLE001 - fail closed
                return GateResult(False, True, f"approval failed ({type(exc).__name__})")
            if decision.approved and decision.remember and not risk.destructive:
                self._always.add(qualified_name)  # destructive tools are never remembered
            return GateResult(decision.approved, True, decision.note)
