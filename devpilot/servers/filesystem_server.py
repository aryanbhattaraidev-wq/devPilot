"""Filesystem MCP server (stdio): read-only, sandboxed access to the project tree.

Semantic search (docs server) finds *where* to look; this server answers exact questions:
"show me lines 40-80 of agent.py", "where is `MAX_ITERATIONS` used?". It is deliberately
read-only and confined to the project root; secrets (.env etc.) are never reachable.
"""

import re
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from devpilot.config import Settings
from devpilot.fs import PathError, iter_project_files, rel_posix, resolve_in_root

root = Path(Settings().project_root).resolve()

MAX_READ_LINES = 400
MAX_READ_CHARS = 30_000

mcp = MCPServer(
    "filesystem",
    instructions="Read-only access to the project's files. Paths are relative to the project root.",
)


def _resolve(path: str) -> Path:
    try:
        return resolve_in_root(root, path)
    except PathError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def list_directory(path: str = ".") -> list[dict]:
    """List files and folders in a directory (relative to the project root)."""
    target = _resolve(path)
    if not target.is_dir():
        raise ToolError(f"not a directory: {path!r}")
    entries = []
    for entry in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        try:
            _resolve(rel_posix(root, entry))
        except ToolError:
            continue  # hidden: blocked or generated
        entries.append(
            {
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            }
        )
    return entries


@mcp.tool()
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> dict:
    """Read a text file, optionally a 1-based inclusive line range (max 400 lines per call)."""
    target = _resolve(path)
    if not target.is_file():
        raise ToolError(f"not a file: {path!r}")
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ToolError(f"{path!r} is not a UTF-8 text file") from exc
    start = max(start_line, 1)
    end = min(end_line or len(lines), len(lines), start + MAX_READ_LINES - 1)
    content = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))
    truncated = len(content) > MAX_READ_CHARS
    return {
        "path": rel_posix(root, target),
        "start_line": start,
        "end_line": end,
        "total_lines": len(lines),
        "truncated": truncated or end < min(end_line or len(lines), len(lines)),
        "content": content[:MAX_READ_CHARS],
    }


@mcp.tool()
def search_files(
    pattern: str, glob: str = "*", ignore_case: bool = True, max_results: int = 50
) -> list[dict]:
    """Regex search (like grep) across project text files. glob filters file names, e.g. '*.py'."""
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}") from exc
    results: list[dict] = []
    for path in iter_project_files(root):
        if not path.match(glob):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                results.append({"path": rel_posix(root, path), "line": n, "text": line.strip()[:200]})
                if len(results) >= max_results:
                    return results
    return results


if __name__ == "__main__":
    mcp.run(transport="stdio")
