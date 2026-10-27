"""Agent loop tests: a scripted fake LLM driving real MCP servers over stdio."""

from types import SimpleNamespace as NS

from devpilot.agent import MAX_ITERATIONS, Agent, to_anthropic_tools
from devpilot.client.manager import MCPClientManager
from devpilot.config import AppConfig

DEMO = {
    "transport": "stdio",
    "command": "python",
    "args": ["-m", "devpilot.servers.demo_stdio"],
}


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, **input):
    return NS(type="tool_use", id=id, name=name, input=input)


def reply(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop)


class FakeLLM:
    """Returns scripted responses and records every request it received."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = self  # client.messages.create(...)

    async def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0) if self._responses else reply(tool_use("x", "demo__add", a=1, b=1), stop="tool_use")


def manager() -> MCPClientManager:
    return MCPClientManager(AppConfig.model_validate({"servers": {"demo": DEMO}}))


async def test_mcp_tools_are_translated_for_the_model():
    async with manager() as m:
        tools = {t["name"]: t for t in to_anthropic_tools(m)}
        assert set(tools) == {"demo__add", "demo__utc_now"}
        assert tools["demo__add"]["input_schema"]["properties"].keys() == {"a", "b"}
        assert "Add two integers" in tools["demo__add"]["description"]


async def test_model_chooses_tool_then_answers_from_the_result():
    llm = FakeLLM(
        reply(text("Let me add."), tool_use("t1", "demo__add", a=2, b=3), stop="tool_use"),
        reply(text("The sum is 5.")),
    )
    async with manager() as m:
        result = await Agent(m, llm, "test-model").ask("what is 2+3?")

    assert result.text == "The sum is 5."
    assert [(c.name, c.arguments, c.result) for c in result.tool_calls] == [("demo__add", {"a": 2, "b": 3}, "5")]
    # The tool result went back to the model, tied to the right tool_use id.
    followup = llm.requests[1]["messages"][-1]
    assert followup["role"] == "user"
    assert followup["content"][0]["tool_use_id"] == "t1"
    assert followup["content"][0]["content"] == "5"
    assert "demo__add" in {t["name"] for t in llm.requests[0]["tools"]}


async def test_parallel_tool_calls_return_in_one_message():
    llm = FakeLLM(
        reply(tool_use("a", "demo__add", a=1, b=1), tool_use("b", "demo__add", a=2, b=2), stop="tool_use"),
        reply(text("done")),
    )
    async with manager() as m:
        await Agent(m, llm, "test-model").ask("two sums")
    results = llm.requests[1]["messages"][-1]["content"]
    assert [(r["tool_use_id"], r["content"]) for r in results] == [("a", "2"), ("b", "4")]


async def test_tool_failures_are_reported_to_the_model_not_raised():
    llm = FakeLLM(
        reply(tool_use("t1", "nope__missing"), tool_use("t2", "demo__add", a="x", b=1), stop="tool_use"),
        reply(text("sorry")),
    )
    async with manager() as m:
        result = await Agent(m, llm, "test-model").ask("break things")
    assert result.text == "sorry"
    assert all(c.is_error for c in result.tool_calls)
    blocks = llm.requests[1]["messages"][-1]["content"]
    assert all(b.get("is_error") for b in blocks)


async def test_runaway_tool_loop_is_stopped():
    llm = FakeLLM()  # default response is an endless tool call
    async with manager() as m:
        result = await Agent(m, llm, "test-model").ask("loop forever")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == MAX_ITERATIONS


async def test_conversation_history_persists_between_questions():
    llm = FakeLLM(reply(text("first")), reply(text("second")))
    async with manager() as m:
        agent = Agent(m, llm, "test-model")
        await agent.ask("one")
        await agent.ask("two")
    roles = [msg["role"] for msg in llm.requests[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
