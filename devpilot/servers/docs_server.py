"""Documentation MCP server (stdio): semantic search over the project's docs and code.

The model asks a natural-language question; we embed it, find the nearest chunks in the
Chroma index, and return them *with citations* (file + line range + heading). The model
is told to cite them, so every claim in an answer can be traced to a source.

Retrieval-augmented generation (RAG) in one sentence: instead of hoping the LLM remembers
your docs, fetch the relevant passages at question time and let it answer from them.
"""

import logging
import time
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from devpilot.config import Settings
from devpilot.rag.factory import build_store

for _name in ("httpx", "httpx2", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)  # stderr noise in the host's terminal

settings = Settings()
root = Path(settings.project_root)
store = build_store(settings)

SYNC_INTERVAL_SECONDS = 30
_last_sync = 0.0

mcp = MCPServer(
    "docs",
    instructions=(
        "Search the project's documentation and source code by meaning. Results include a "
        "`citation` (file:lines) for each passage; quote it when you use the passage."
    ),
)


async def _ensure_fresh() -> str | None:
    """Incremental sync (hash-based, so cheap when nothing changed), at most every 30s."""
    global _last_sync
    if store.count() and time.monotonic() - _last_sync < SYNC_INTERVAL_SECONDS:
        return None
    report = await store.sync(root)
    _last_sync = time.monotonic()
    return str(report)


@mcp.tool()
async def search_docs(query: str, limit: int = 5) -> dict:
    """Semantic (meaning-based) search over project docs and code.

    `query` must be a complete natural-language question or descriptive sentence, e.g. "why
    does each server connection run in its own asyncio task". Do NOT pass bare keywords like
    "asyncio task": they match unrelated text. Returns passages (best first) with citations;
    cosine scores below ~0.3 are only loosely related. If the first search misses, rephrase
    the whole question rather than shortening it.
    """
    try:
        await _ensure_fresh()
        hits = await store.search(query, limit)
    except Exception as exc:  # embedding API down, bad key, ... -> readable to the model
        raise ToolError(f"search failed: {type(exc).__name__}: {exc}") from exc
    return {"query": query, "results": hits}


@mcp.tool()
async def reindex_docs() -> dict:
    """Force a re-scan of the project; only changed files are re-embedded."""
    global _last_sync
    try:
        report = await store.sync(root)
    except Exception as exc:
        raise ToolError(f"indexing failed: {type(exc).__name__}: {exc}") from exc
    _last_sync = time.monotonic()
    return {"report": str(report), "chunks": report.chunks}


if __name__ == "__main__":
    mcp.run(transport="stdio")
