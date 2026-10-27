"""The DevPilot agent: an LLM that decides which MCP tools to call.

This is the "host" in MCP terms. The loop is the whole trick:

    1. Show the model every tool the connected MCP servers expose (names,
       descriptions, JSON Schemas -- discovered at runtime, nothing hard-coded).
    2. The model replies with text and/or ``tool_use`` blocks. *It* chooses
       which tools to call and with what arguments, based on the question.
    3. We route each call to the owning MCP server, and feed results back as
       ``tool_result`` blocks.
    4. Repeat until the model answers without asking for more tools.

The model never talks to MCP directly; the host translates in both directions.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import anthropic

from devpilot.client.manager import MCPClientManager

MAX_TOOL_RESULT_CHARS = 20_000
MAX_ITERATIONS = 12

SYSTEM_PROMPT = """\
You are DevPilot, an AI assistant that helps developers manage their software projects.

You have tools exposed by MCP servers (GitHub issues and pull requests, and later project \
tasks, files and documentation). Use them to ground your answers in real data instead of \
guessing. Call several independent tools at once when you can.

When asked what to work on next, look at the actual open issues and pull requests, weigh \
severity, user impact, blockers and effort, and give a short ranked recommendation with the \
reasoning behind it. Always cite issue/PR numbers. If a tool fails or returns nothing, say \
so plainly rather than inventing data.

Today's date is {today}. The default GitHub repository is {repo}.
"""

# Called as (tool_name, arguments) just before a tool runs, and (tool_name, text, is_error) after.
OnToolCall = Callable[[str, dict], Awaitable[None] | None]
OnToolResult = Callable[[str, str, bool], Awaitable[None] | None]


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result: str
    is_error: bool


@dataclass
class AgentResult:
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str | None = None


def to_anthropic_tools(manager: MCPClientManager) -> list[dict[str, Any]]:
    """MCP tool definitions -> Anthropic tool definitions.

    The two formats are nearly identical on purpose: both use JSON Schema. The only
    work is a rename (MCP ``input_schema`` is Anthropic's ``input_schema`` too) and
    using the namespaced name so a call can be routed back to its server.
    """
    return [
        {
            "name": ref.qualified_name,
            "description": ref.tool.description or "",
            "input_schema": ref.tool.input_schema,
        }
        for ref in manager.tools()
    ]


def _result_text(result: Any) -> str:
    parts = [getattr(block, "text", None) or str(block) for block in result.content]
    text = "\n".join(parts)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = (
            text[:MAX_TOOL_RESULT_CHARS]
            + f"\n[truncated: {len(text) - MAX_TOOL_RESULT_CHARS} more characters omitted]"
        )
    return text


async def _maybe_await(value: Awaitable[None] | None) -> None:
    if value is not None:
        await value


class Agent:
    def __init__(
        self,
        manager: MCPClientManager,
        client: Any,
        model: str,
        default_repo: str = "",
        on_tool_call: OnToolCall | None = None,
        on_tool_result: OnToolResult | None = None,
    ):
        self.manager = manager
        self.client = client
        self.model = model
        self.system = SYSTEM_PROMPT.format(today=date.today().isoformat(), repo=default_repo or "(none)")
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        # The API is stateless: this list *is* the conversation. Kept across ask() calls.
        self.messages: list[dict[str, Any]] = []

    async def _run_tool(self, block: Any) -> tuple[dict[str, Any], ToolCallRecord]:
        args = dict(block.input or {})
        if self.on_tool_call:
            await _maybe_await(self.on_tool_call(block.name, args))
        try:
            result = await self.manager.call_tool(block.name, args)
            text, is_error = _result_text(result), bool(result.is_error)
        except Exception as exc:  # unknown tool, dead server, ... -> let the model adapt
            text, is_error = f"{type(exc).__name__}: {exc}", True
        if self.on_tool_result:
            await _maybe_await(self.on_tool_result(block.name, text, is_error))
        message_block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.id,  # must echo the tool_use block's id
            "content": text or "(empty result)",
        }
        if is_error:
            message_block["is_error"] = True
        return message_block, ToolCallRecord(block.name, args, text, is_error)

    async def ask(self, question: str) -> AgentResult:
        self.messages.append({"role": "user", "content": question})
        tools = to_anthropic_tools(self.manager)
        calls: list[ToolCallRecord] = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=self.system,
                tools=tools,
                messages=self.messages,
            )
            # Keep the assistant turn exactly as returned (incl. thinking blocks).
            self.messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                text = "\n".join(b.text for b in response.content if b.type == "text")
                if response.stop_reason == "refusal":
                    text = text or "The model declined to answer this request."
                return AgentResult(text, calls, iteration, response.stop_reason)

            # Independent calls run concurrently; ALL results go back in ONE user message.
            outcomes = await asyncio.gather(*(self._run_tool(b) for b in tool_uses))
            self.messages.append({"role": "user", "content": [o[0] for o in outcomes]})
            calls.extend(o[1] for o in outcomes)

        return AgentResult(
            f"Stopped after {MAX_ITERATIONS} tool-calling rounds without a final answer.",
            calls,
            MAX_ITERATIONS,
            "max_iterations",
        )


def make_client(api_key: str) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=api_key or None)  # None -> SDK env/profile resolution


def format_args(args: dict) -> str:
    return json.dumps(args, separators=(",", ":"), ensure_ascii=False)
