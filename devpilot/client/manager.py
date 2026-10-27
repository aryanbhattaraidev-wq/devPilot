"""Multi-server MCP client manager.

Design notes (the interesting part of MCP client-side architecture):

* One ``Client`` == one connection to one server. A host application (DevPilot)
  keeps *many* of them and merges their tools into a single catalogue.
* Transports (stdio subprocess / streamable HTTP) are async context managers built
  on anyio task groups. Those must be entered and exited **in the same task**, so
  each server gets its own long-lived task that owns its connection. Callers talk
  to the server through that task's ``Client`` from anywhere.
* Failure isolation: a server that fails to start, times out, or crashes is
  recorded as FAILED with the reason; the other servers keep working.
* Tool names are namespaced as ``<server>__<tool>`` so two servers can both expose
  e.g. ``search`` without colliding, and so we can route a call back to its owner.
"""

import asyncio
import contextlib
import logging
import sys
from dataclasses import dataclass, field
from enum import StrEnum

import httpx2
from mcp import Client, types
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from devpilot.config import AppConfig, HttpServerConfig, ServerConfig, StdioServerConfig

log = logging.getLogger(__name__)

SEP = "__"


class ServerStatus(StrEnum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class ServerState:
    name: str
    transport: str
    status: ServerStatus = ServerStatus.CONNECTING
    error: str | None = None
    tools: list[types.Tool] = field(default_factory=list)


def _root_cause(exc: BaseException) -> str:
    """Unwrap ExceptionGroups (anyio wraps errors) into a readable one-liner."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _open_client(config: ServerConfig) -> Client:
    if isinstance(config, StdioServerConfig):
        # "python" means "the interpreter running DevPilot", so configs are portable.
        command = sys.executable if config.command == "python" else config.command
        return Client(
            StdioServerParameters(
                command=command, args=config.args, env=config.env or None, cwd=config.cwd
            )
        )
    assert isinstance(config, HttpServerConfig)
    http = httpx2.AsyncClient(headers=config.headers, timeout=30.0) if config.headers else None
    return Client(streamable_http_client(config.url, http_client=http))


class ServerConnection:
    """Owns the lifetime of a single server connection inside its own task."""

    def __init__(self, name: str, config: ServerConfig):
        self.config = config
        self.state = ServerState(name=name, transport=config.transport)
        self.client: Client | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> ServerState:
        self._task = asyncio.create_task(self._run(), name=f"mcp:{self.state.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), self.config.connect_timeout)
        except TimeoutError:
            self._fail(f"timed out after {self.config.connect_timeout:.0f}s")
            self._task.cancel()
        return self.state

    async def _run(self) -> None:
        try:
            async with _open_client(self.config) as client:
                tools = (await client.list_tools()).tools  # dynamic discovery
                self.client = client
                self.state.tools = tools
                self.state.status = ServerStatus.CONNECTED
                self._ready.set()
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - we want *every* failure recorded
            self._fail(_root_cause(exc))
        finally:
            self.client = None
            if self.state.status is ServerStatus.CONNECTED:
                self.state.status = ServerStatus.CLOSED
            self._ready.set()

    def _fail(self, reason: str) -> None:
        self.state.status = ServerStatus.FAILED
        self.state.error = reason
        self.state.tools = []
        self._ready.set()

    async def refresh_tools(self) -> None:
        """Re-run tools/list (servers may add/remove tools at runtime)."""
        if self.client:
            self.state.tools = (await self.client.list_tools()).tools

    async def close(self) -> None:
        self._stop.set()
        if self._task:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(self._task, 10)


@dataclass
class ToolRef:
    qualified_name: str
    server: str
    tool: types.Tool


class MCPClientManager:
    def __init__(self, config: AppConfig):
        self._connections = {n: ServerConnection(n, c) for n, c in config.servers.items()}

    async def __aenter__(self) -> "MCPClientManager":
        await self.connect_all()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def connect_all(self) -> None:
        # Concurrent: one slow/dead server must not delay the rest.
        await asyncio.gather(*(c.start() for c in self._connections.values()))

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in self._connections.values()))

    def servers(self) -> list[ServerState]:
        return [c.state for c in self._connections.values()]

    def tools(self) -> list[ToolRef]:
        return [
            ToolRef(f"{s.name}{SEP}{t.name}", s.name, t)
            for s in self.servers()
            if s.status is ServerStatus.CONNECTED
            for t in s.tools
        ]

    async def call_tool(
        self, qualified_name: str, arguments: dict | None = None
    ) -> types.CallToolResult:
        server, sep, tool = qualified_name.partition(SEP)
        conn = self._connections.get(server)
        if not sep or conn is None:
            raise KeyError(f"unknown tool {qualified_name!r}")
        if conn.client is None:
            raise ConnectionError(f"server {server!r} is {conn.state.status}: {conn.state.error}")
        try:
            return await conn.client.call_tool(tool, arguments or {})
        except Exception as exc:
            # Transport died mid-session (e.g. server crashed): record it, tell the caller.
            conn._fail(_root_cause(exc))
            raise ConnectionError(f"server {server!r} failed: {conn.state.error}") from exc
