"""DevPilot CLI. Step 1 commands: `servers` and `call`."""

import argparse
import asyncio
import json
import logging

from rich.console import Console
from rich.table import Table

from devpilot.client.manager import MCPClientManager, ServerStatus
from devpilot.config import load_config

console = Console()
STATUS_STYLE = {
    ServerStatus.CONNECTED: "green",
    ServerStatus.FAILED: "red",
    ServerStatus.CONNECTING: "yellow",
    ServerStatus.CLOSED: "dim",
}


def render(manager: MCPClientManager) -> None:
    table = Table(title="MCP servers")
    for col in ("Server", "Transport", "Status", "Tools / Error"):
        table.add_column(col)
    for s in manager.servers():
        style = STATUS_STYLE[s.status]
        detail = (
            ", ".join(t.name for t in s.tools)
            if s.status is ServerStatus.CONNECTED
            else (s.error or "")
        )
        table.add_row(s.name, s.transport, f"[{style}]{s.status}[/{style}]", detail)
    console.print(table)
    console.print(f"[bold]{len(manager.tools())}[/bold] tools available")


async def run(args: argparse.Namespace) -> None:
    async with MCPClientManager(load_config(args.config)) as manager:
        render(manager)
        if args.command == "call":
            result = await manager.call_tool(args.tool, json.loads(args.arguments))
            for block in result.content:
                console.print(getattr(block, "text", block))
            if result.is_error:
                console.print("[red]tool reported an error[/red]")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="devpilot")
    p.add_argument("--config", default="servers.json")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("servers", help="connect to all servers and list their tools")
    c = sub.add_parser("call", help="call one tool by qualified name (server__tool) with JSON args")
    c.add_argument("tool")
    c.add_argument("arguments", nargs="?", default="{}")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
