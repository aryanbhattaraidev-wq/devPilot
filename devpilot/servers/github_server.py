"""GitHub MCP server (stdio): issues, pull requests and repo info.

Tools are *model-controlled* actions, so their names, docstrings and parameter
types matter: MCPServer turns them into the JSON Schema the LLM sees. Write
docstrings for the model, not for humans.

Secrets: the token is read from the environment/.env inside this process. The
host never passes it over the protocol and it never appears in servers.json.
"""

import logging

from mcp.server import MCPServer
from mcp_types import ToolAnnotations
from mcp.server.mcpserver.exceptions import ToolError

from devpilot.github_api import GitHubClient, GitHubError, Settings

logging.getLogger("httpx").setLevel(logging.WARNING)  # stderr noise; URLs aren't useful to the host

settings = Settings()
gh = GitHubClient(settings.github_token, settings.github_repo)

mcp = MCPServer(
    "github",
    instructions=(
        "Read GitHub issues, pull requests and repository info. "
        "The `repo` argument is optional ('owner/name'); it defaults to the configured repository."
    ),
)


async def _guard(coro):
    """Turn API failures into tool errors the model can read and react to.

    The SDK deliberately masks unexpected exceptions (so internals don't leak to
    the client); only ToolError text is passed through to the model.
    """
    try:
        return await coro
    except GitHubError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def get_repo(repo: str | None = None) -> dict:
    """Get repository metadata: description, default branch, language, open issue count."""
    return await _guard(gh.get_repo(repo))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def list_issues(
    repo: str | None = None, state: str = "open", labels: str = "", limit: int = 30
) -> list[dict]:
    """List issues (not pull requests), most recently updated first.

    state: 'open', 'closed' or 'all'. labels: comma-separated label names to filter by.
    """
    return await _guard(gh.list_issues(repo, state, labels, limit))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def get_issue(number: int, repo: str | None = None) -> dict:
    """Get one issue with its full body and recent comments."""
    return await _guard(gh.get_issue(number, repo))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def list_pull_requests(
    repo: str | None = None, state: str = "open", limit: int = 30
) -> list[dict]:
    """List pull requests, most recently updated first. state: 'open', 'closed' or 'all'."""
    return await _guard(gh.list_pull_requests(repo, state, limit))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def get_pull_request(number: int, repo: str | None = None) -> dict:
    """Get one pull request with its body, merge status and diff size."""
    return await _guard(gh.get_pull_request(number, repo))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def search_issues(
    query: str, repo: str | None = None, limit: int = 20, kind: str = "issue"
) -> list[dict]:
    """Full-text search in the repository. kind: 'issue' (default) or 'pr' for pull requests."""
    return await _guard(gh.search_issues(query, repo, limit, kind))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True))
async def create_issue(
    title: str, body: str = "", labels: list[str] | None = None, repo: str | None = None
) -> dict:
    """Create a new issue. This WRITES to GitHub and is visible to others."""
    return await _guard(gh.create_issue(title, body, labels, repo))


if __name__ == "__main__":
    mcp.run(transport="stdio")
