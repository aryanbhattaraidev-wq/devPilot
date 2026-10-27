import json

import pytest

from devpilot.client.manager import MCPClientManager
from devpilot.config import AppConfig
from devpilot.fs import PathError, iter_project_files, resolve_in_root
from devpilot.rag.chunking import MAX_CHARS, chunk_text
from devpilot.rag.embeddings import HashEmbedder
from devpilot.rag.store import DocStore

# ---- chunking ---------------------------------------------------------------------------

MD = """# Guide

intro text

## Install

run the installer

```bash
# not a heading
pip install thing
```

## Usage

call the function
"""


def test_markdown_chunks_follow_headings_with_line_numbers():
    chunks = chunk_text(MD, "guide.md")
    by_heading = {c.heading: c for c in chunks}
    assert set(by_heading) == {"Guide", "Guide > Install", "Guide > Usage"}
    install = by_heading["Guide > Install"]
    assert "# not a heading" in install.text  # fenced '#' is not a heading
    assert (install.start_line, install.end_line) == (5, 12)
    assert MD.splitlines()[install.start_line - 1] == "## Install"


def test_long_text_is_split_with_overlap_and_no_line_is_lost():
    lines = [f"line number {i} with some padding text to take up room" for i in range(1, 200)]
    chunks = chunk_text("\n".join(lines), "big.py")
    assert len(chunks) > 3
    assert all(len(c.text) <= MAX_CHARS + 100 for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == len(lines)
    assert chunks[1].start_line <= chunks[0].end_line  # overlap
    covered = {n for c in chunks for n in range(c.start_line, c.end_line + 1)}
    assert covered == set(range(1, len(lines) + 1))


def test_giant_single_line_and_empty_files():
    chunks = chunk_text("x" * 5000, "min.js")
    assert len(chunks) >= 4 and all(len(c.text) <= MAX_CHARS for c in chunks)
    assert chunk_text("\n\n  \n", "empty.md") == []


# ---- path safety ------------------------------------------------------------------------


def test_sandbox_blocks_escape_and_secrets(tmp_path):
    (tmp_path / ".env").write_text("TOKEN=secret")
    (tmp_path / "ok.md").write_text("hi")
    (tmp_path / "data.db").write_bytes(b"x")
    with pytest.raises(PathError, match="escapes"):
        resolve_in_root(tmp_path, "../outside.txt")
    with pytest.raises(PathError, match="not allowed"):
        resolve_in_root(tmp_path, ".env")
    assert resolve_in_root(tmp_path, "ok.md").name == "ok.md"
    assert [p.name for p in iter_project_files(tmp_path)] == ["ok.md"]


# ---- store ------------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "auth.md").write_text("# Auth\n\nUsers log in with a password and receive a session token.\n")
    (root / "docs" / "deploy.md").write_text("# Deploy\n\nContainers are built by the pipeline and pushed to the registry.\n")
    (root / ".env").write_text("SECRET=hunter2")
    return root


async def test_search_returns_cited_relevant_chunk(project, tmp_path):
    store = DocStore(tmp_path / "chroma", HashEmbedder())
    report = await store.sync(project)
    assert (report.added, report.chunks) == (2, 2)

    [top, *_] = await store.search("how do users log in with a password", k=2)
    assert top["source"] == "docs/auth.md"
    assert top["citation"] == "docs/auth.md:1-3 (Auth)"
    assert "session token" in top["text"]
    assert not any("hunter2" in h["text"] for h in await store.search("SECRET hunter2", k=5))


async def test_sync_is_incremental(project, tmp_path):
    store = DocStore(tmp_path / "chroma", HashEmbedder())
    await store.sync(project)

    again = await store.sync(project)
    assert (again.unchanged, again.added, again.updated, again.removed) == (2, 0, 0, 0)

    (project / "docs" / "auth.md").write_text("# Auth\n\nNow we use single sign-on via an identity provider.\n")
    (project / "docs" / "deploy.md").unlink()
    (project / "docs" / "new.md").write_text("# New\n\nfresh page\n")
    changed = await store.sync(project)
    assert (changed.updated, changed.removed, changed.added) == (1, 1, 1)

    sources = {h["source"] for h in await store.search("identity provider single sign-on", k=5)}
    assert sources == {"docs/auth.md", "docs/new.md"}  # deploy.md is gone, old auth text replaced
    assert all("password" not in h["text"] for h in await store.search("password", k=5))


async def test_each_embedder_gets_its_own_collection(project, tmp_path):
    class Other(HashEmbedder):
        name = "other-model"

    a = DocStore(tmp_path / "chroma", HashEmbedder())
    await a.sync(project)
    assert DocStore(tmp_path / "chroma", Other()).count() == 0


# ---- the servers, over real MCP ---------------------------------------------------------


def items(result) -> list[dict]:
    """A tool returning a list arrives as one JSON text block per item."""
    return [json.loads(block.text) for block in result.content]


def manager(project, tmp_path) -> MCPClientManager:
    env = {
        "PROJECT_ROOT": str(project),
        "CHROMA_DIR": str(tmp_path / "chroma"),
        "EMBEDDER": "hash",
    }
    servers = {
        name: {"transport": "stdio", "command": "python", "args": ["-m", f"devpilot.servers.{name}_server"], "env": env}
        for name in ("docs", "filesystem")
    }
    return MCPClientManager(AppConfig.model_validate({"servers": servers}))


async def test_docs_server_indexes_lazily_and_searches(project, tmp_path):
    async with manager(project, tmp_path) as m:
        result = await m.call_tool("docs__search_docs", {"query": "container registry pipeline", "limit": 1})
        payload = json.loads(result.content[0].text)
        assert payload["results"][0]["citation"].startswith("docs/deploy.md:")


async def test_filesystem_server_reads_searches_and_refuses(project, tmp_path):
    async with manager(project, tmp_path) as m:
        read = json.loads((await m.call_tool("filesystem__read_file", {"path": "docs/auth.md", "start_line": 3})).content[0].text)
        assert read["content"].startswith("3: Users log in") and read["total_lines"] == 3

        found = items(await m.call_tool("filesystem__search_files", {"pattern": "registry", "glob": "*.md"}))
        assert [(f["path"], f["line"]) for f in found] == [("docs/deploy.md", 3)]

        for bad in ("../x", ".env"):
            denied = await m.call_tool("filesystem__read_file", {"path": bad})
            assert denied.is_error

        listing = items(await m.call_tool("filesystem__list_directory", {}))
        assert [e["name"] for e in listing] == ["docs"]  # .env hidden
