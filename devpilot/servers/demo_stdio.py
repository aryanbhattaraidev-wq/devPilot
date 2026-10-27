"""Tiny local MCP server over stdio. Launched by the client as a subprocess.

stdout is the protocol channel here -- never print() to it. Log to stderr.
"""

from datetime import datetime, timezone

from mcp.server import MCPServer

mcp = MCPServer("demo-stdio")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def utc_now() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    mcp.run(transport="stdio")
