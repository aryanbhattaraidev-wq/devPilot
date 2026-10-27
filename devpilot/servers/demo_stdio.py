"""Tiny local MCP server over stdio. Launched by the client as a subprocess.

stdout is the protocol channel here -- never print() to it. Log to stderr.
"""

from datetime import datetime, timezone

from mcp.server import MCPServer
from mcp_types import ToolAnnotations

mcp = MCPServer("demo-stdio")


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def utc_now() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    mcp.run(transport="stdio")
