"""DevPilot CLI: `servers`, `call`, `ask`, `chat`."""

import argparse
import asyncio
import json
import logging

from rich.console import Console
from rich.table import Table

from devpilot.agent import Agent, format_args, make_client
from devpilot.client.manager import MCPClientManager, ServerStatus
from devpilot.config import Settings, load_config

console = Console()
STATUS_STYLE = {
    ServerStatus.CONNECTED: "green",
    ServerStatus.FAILED: "red",
    ServerStatus.CONNECTING: "yellow",
    ServerStatus.CLOSED: "dim",
}


def render(manager: MCPClientManager) -> None:
    table = Table(title="MCP servers")
    for col in ("Server", "Transport", "Status", "Res", "Prompts", "Tools / Error"):
        table.add_column(col)
    for s in manager.servers():
        style = STATUS_STYLE[s.status]
        up = s.status is ServerStatus.CONNECTED
        detail = ", ".join(t.name for t in s.tools) if up else (s.error or "")
        table.add_row(
            s.name,
            s.transport,
            f"[{style}]{s.status}[/{style}]",
            str(len(s.resources) + len(s.templates)) if up else "",
            str(len(s.prompts)) if up else "",
            detail,
        )
    console.print(table)
    console.print(f"[bold]{len(manager.tools())}[/bold] tools available")


def build_agent(manager: MCPClientManager) -> Agent:
    settings = Settings()
    if not settings.llm_api_key:
        raise SystemExit("No LLM key: set OPENROUTER_API_KEY (or LLM_API_KEY) in .env.")

    def show_call(name: str, arguments: dict) -> None:
        console.print(f"[cyan]> {name}[/cyan] [dim]{format_args(arguments)}[/dim]")

    def show_result(name: str, text: str, is_error: bool) -> None:
        style = "red" if is_error else "dim"
        console.print(f"[{style}]  {'error: ' if is_error else ''}{len(text)} chars from {name}[/{style}]")

    return Agent(
        manager,
        make_client(settings.llm_api_key, settings.llm_base_url),
        settings.devpilot_model,
        default_repo=settings.github_repo,
        on_tool_call=show_call,
        on_tool_result=show_result,
    )


async def run_index() -> None:
    from pathlib import Path

    from devpilot.rag.factory import build_store

    settings = Settings()
    if settings.embedder == "api" and not settings.llm_api_key:
        raise SystemExit("No LLM key: set OPENROUTER_API_KEY (or LLM_API_KEY) in .env.")
    store = build_store(settings)
    console.print(f"indexing {settings.project_root} with {store.embedder.name} ...")
    console.print(await store.sync(Path(settings.project_root)))


async def run(args: argparse.Namespace) -> None:
    if args.command == "index":
        return await run_index()
    async with MCPClientManager(load_config(args.config)) as manager:
        if args.command in ("ask", "chat"):
            agent = build_agent(manager)
            ok = [s for s in manager.servers() if s.status is ServerStatus.CONNECTED]
            console.print(f"[dim]{len(ok)} servers, {len(manager.tools())} tools, model {agent.model}[/dim]")
            if args.command == "ask":
                console.print((await agent.ask(args.question)).text, markup=False, highlight=False)
            else:
                console.print("[bold]DevPilot[/bold] - empty line or Ctrl+C to quit")
                while True:
                    try:
                        question = await asyncio.to_thread(
                            console.input, "[bold green]you>[/bold green] "
                        )
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not question.strip():
                        break
                    console.print()
                    console.print((await agent.ask(question)).text, markup=False, highlight=False)
            return
        render(manager)
        if args.command == "read":
            console.print(await manager.read_resource(args.server, args.uri))
        if args.command == "prompt":
            console.print(await manager.get_prompt(args.server, args.name, json.loads(args.arguments)))
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
    sub.add_parser("index", help="build or refresh the documentation search index")
    r = sub.add_parser("read", help="read an MCP resource, e.g. read project project://summary")
    r.add_argument("server")
    r.add_argument("uri")
    pr = sub.add_parser("prompt", help="render an MCP prompt template")
    pr.add_argument("server")
    pr.add_argument("name")
    pr.add_argument("arguments", nargs="?", default="{}")
    a = sub.add_parser("ask", help="ask one question; the AI chooses which tools to call")
    a.add_argument("question")
    sub.add_parser("chat", help="interactive conversation")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
