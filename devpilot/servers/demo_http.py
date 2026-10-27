"""Tiny "remote" MCP server over streamable HTTP. Run it yourself:

    uv run python -m devpilot.servers.demo_http

Unlike stdio, this process is independent of the client: it is started
separately and can be reached by any number of clients at http://host:port/mcp.
"""

import random

from mcp.server import MCPServer

mcp = MCPServer("demo-http")


@mcp.tool()
def roll_dice(sides: int = 6) -> int:
    """Roll a die with the given number of sides."""
    return random.randint(1, sides)


@mcp.tool()
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8765)
