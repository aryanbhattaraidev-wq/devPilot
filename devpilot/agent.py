"""The DevPilot agent: an LLM that decides which MCP tools to call.

This is the "host" in MCP terms. The loop is the whole trick:

    1. Show the model every tool the connected MCP servers expose (names,
       descriptions, JSON Schemas -- discovered at runtime, nothing hard-coded).
    2. The model replies with text and/or tool calls. *It* chooses which tools
       to call and with what arguments, based on the question.
    3. We route each call to the owning MCP server, and feed results back as
       ``tool`` messages.
    4. Repeat until the model answers without asking for more tools.

The model never talks to MCP directly; the host translates in both directions.
MCP is the protocol between host and tool servers; the host<->LLM leg uses whatever
the LLM vendor speaks -- here the OpenAI-compatible chat API (OpenRouter, OpenAI,
Ollama, ...), selected by ``LLM_BASE_URL``.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import openai

from devpilot.client.manager import MCPClientManager

MAX_TOOL_RESULT_CHARS = 20_000
MAX_ITERATIONS = 12
MAX_OUTPUT_TOKENS = 4096

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


def to_llm_tools(manager: MCPClientManager) -> list[dict[str, Any]]:
    """MCP tool definitions -> OpenAI-style function definitions.

    Both sides describe arguments with JSON Schema, so the work is re-wrapping:
    MCP ``input_schema`` becomes the function's ``parameters``. The name is the
    namespaced ``server__tool`` so a call can be routed back to its server.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": ref.qualified_name,
                "description": ref.tool.description or "",
                "parameters": ref.tool.input_schema,
            },
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

    async def _run_tool(self, call: Any) -> tuple[dict[str, Any], ToolCallRecord]:
        name = call.function.name
        args: dict = {}
        try:
            # Arguments arrive as a JSON *string* in this API and models sometimes botch it.
            args = json.loads(call.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            if self.on_tool_call:
                await _maybe_await(self.on_tool_call(name, args))
            result = await self.manager.call_tool(name, args)
            text, is_error = _result_text(result), bool(result.is_error)
        except Exception as exc:  # bad JSON, unknown tool, dead server... -> let the model adapt
            text, is_error = f"{type(exc).__name__}: {exc}", True
        if self.on_tool_result:
            await _maybe_await(self.on_tool_result(name, text, is_error))
        message = {"role": "tool", "tool_call_id": call.id, "content": text or "(empty result)"}
        return message, ToolCallRecord(name, args, text, is_error)

    async def ask(self, question: str) -> AgentResult:
        self.messages.append({"role": "user", "content": question})
        tools = to_llm_tools(self.manager)
        calls: list[ToolCallRecord] = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[{"role": "system", "content": self.system}, *self.messages],
                **({"tools": tools} if tools else {}),
            )
            choice = response.choices[0]
            msg = choice.message
            tool_calls = list(msg.tool_calls or [])

            assistant: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }
                    for c in tool_calls
                ]
            self.messages.append(assistant)

            if not tool_calls:
                text = msg.content or ""
                if choice.finish_reason == "length":
                    text += "\n[response cut off: output token limit reached]"
                elif choice.finish_reason == "content_filter" and not text:
                    text = "The model declined to answer this request."
                return AgentResult(text, calls, iteration, choice.finish_reason)

            # Independent calls run concurrently; each result is its own `tool` message,
            # appended in the same order as the calls.
            outcomes = await asyncio.gather(*(self._run_tool(c) for c in tool_calls))
            self.messages.extend(o[0] for o in outcomes)
            calls.extend(o[1] for o in outcomes)

        return AgentResult(
            f"Stopped after {MAX_ITERATIONS} tool-calling rounds without a final answer.",
            calls,
            MAX_ITERATIONS,
            "max_iterations",
        )


def make_client(api_key: str, base_url: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


def format_args(args: dict) -> str:
    return json.dumps(args, separators=(",", ":"), ensure_ascii=False)
