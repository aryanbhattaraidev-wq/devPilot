"""Agent loop tests: a scripted fake LLM driving real MCP servers over stdio."""

import json
from types import SimpleNamespace as NS

from devpilot.agent import MAX_ITERATIONS, Agent, to_llm_tools
from devpilot.client.manager import MCPClientManager
from devpilot.config import AppConfig

DEMO = {
    "transport": "stdio",
    "command": "python",
    "args": ["-m", "devpilot.servers.demo_stdio"],
}


def call(id, name, arguments="{}", **kwargs):
    if kwargs:
        arguments = json.dumps(kwargs)
    return NS(id=id, type="function", function=NS(name=name, arguments=arguments))


def reply(content=None, *tool_calls, finish=None):
    finish = finish or ("tool_calls" if tool_calls else "stop")
    msg = NS(content=content, tool_calls=list(tool_calls) or None)
    return NS(choices=[NS(message=msg, finish_reason=finish)])


class FakeLLM:
    """Returns scripted responses and records every request it received."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []
        self.chat = NS(completions=self)  # client.chat.completions.create(...)

    async def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if self._responses:
            return self._responses.pop(0)
        return reply(None, call("x", "demo__add", a=1, b=1))  # endless tool calls


def manager() -> MCPClientManager:
    return MCPClientManager(AppConfig.model_validate({"servers": {"demo": DEMO}}))


async def test_mcp_tools_are_translated_for_the_model():
    async with manager() as m:
        tools = {t["function"]["name"]: t["function"] for t in to_llm_tools(m)}
        assert set(tools) == {"demo__add", "demo__utc_now"}
        assert tools["demo__add"]["parameters"]["properties"].keys() == {"a", "b"}
        assert "Add two integers" in tools["demo__add"]["description"]


async def test_model_chooses_tool_then_answers_from_the_result():
    llm = FakeLLM(
        reply("Let me add.", call("t1", "demo__add", a=2, b=3)),
        reply("The sum is 5."),
    )
    async with manager() as m:
        result = await Agent(m, llm, "test-model").ask("what is 2+3?")

    assert result.text == "The sum is 5."
    assert [(c.name, c.arguments, c.result) for c in result.tool_calls] == [("demo__add", {"a": 2, "b": 3}, "5")]
    # The result went back to the model as a `tool` message tied to the right call id.
    sent = llm.requests[1]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[-2]["tool_calls"][0]["id"] == "t1"
    assert sent[-1] == {"role": "tool", "tool_call_id": "t1", "content": "5"}
    assert "demo__add" in {t["function"]["name"] for t in llm.requests[0]["tools"]}


async def test_parallel_tool_calls_each_get_a_result_in_order():
    llm = FakeLLM(
        reply(None, call("a", "demo__add", a=1, b=1), call("b", "demo__add", a=2, b=2)),
        reply("done"),
    )
    async with manager() as m:
        await Agent(m, llm, "test-model").ask("two sums")
    tool_msgs = [x for x in llm.requests[1]["messages"] if x["role"] == "tool"]
    assert [(x["tool_call_id"], x["content"]) for x in tool_msgs] == [("a", "2"), ("b", "4")]


async def test_tool_failures_are_reported_to_the_model_not_raised():
    llm = FakeLLM(
        reply(
            None,
            call("t1", "nope__missing"),
            call("t2", "demo__add", a="x", b=1),
            call("t3", "demo__add", arguments="{not json"),
        ),
        reply("sorry"),
    )
    async with manager() as m:
        result = await Agent(m, llm, "test-model").ask("break things")
    assert result.text == "sorry"
    assert len(result.tool_calls) == 3 and all(c.is_error for c in result.tool_calls)
    tool_msgs = [x for x in llm.requests[1]["messages"] if x["role"] == "tool"]
    assert [x["tool_call_id"] for x in tool_msgs] == ["t1", "t2", "t3"]


async def test_runaway_tool_loop_is_stopped():
    llm = FakeLLM()  # default response is an endless tool call
    async with manager() as m:
        result = await Agent(m, llm, "test-model").ask("loop forever")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == MAX_ITERATIONS


async def test_conversation_history_persists_between_questions():
    llm = FakeLLM(reply("first"), reply("second"))
    async with manager() as m:
        agent = Agent(m, llm, "test-model")
        await agent.ask("one")
        await agent.ask("two")
    roles = [msg["role"] for msg in llm.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
