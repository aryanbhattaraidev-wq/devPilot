import pytest

from devpilot.client.manager import MCPClientManager, ServerStatus
from devpilot.config import AppConfig

GOOD = {
    "transport": "stdio",
    "command": "python",
    "args": ["-m", "devpilot.servers.demo_stdio"],
}
BAD_COMMAND = {"transport": "stdio", "command": "definitely-not-a-real-command"}
HTTP_DOWN = {
    "transport": "streamable-http",
    "url": "http://127.0.0.1:9/mcp",  # nothing listens on the discard port
    "connect_timeout": 5,
}


def manager(**servers) -> MCPClientManager:
    return MCPClientManager(AppConfig.model_validate({"servers": servers}))


async def test_stdio_discovers_tools_and_calls_them():
    async with manager(demo=GOOD) as m:
        names = {t.qualified_name for t in m.tools()}
        assert names == {"demo__add", "demo__utc_now"}
        result = await m.call_tool("demo__add", {"a": 2, "b": 3})
        assert result.content[0].text == "5"


async def test_failing_servers_do_not_break_healthy_ones():
    async with manager(good=GOOD, bad=BAD_COMMAND, down=HTTP_DOWN) as m:
        status = {s.name: s for s in m.servers()}
        assert status["good"].status is ServerStatus.CONNECTED
        assert status["bad"].status is ServerStatus.FAILED and status["bad"].error
        assert status["down"].status is ServerStatus.FAILED and status["down"].error
        assert {t.server for t in m.tools()} == {"good"}


async def test_calling_failed_or_unknown_server_raises():
    async with manager(bad=BAD_COMMAND) as m:
        with pytest.raises(ConnectionError):
            await m.call_tool("bad__anything")
        with pytest.raises(KeyError):
            await m.call_tool("nope__x")
