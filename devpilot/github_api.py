"""Thin async wrapper over the GitHub REST API.

Kept separate from the MCP server so the API logic is testable without MCP,
and so the MCP layer stays a thin "expose these functions as tools" shell.
Responses are trimmed to the fields an LLM actually needs: smaller context,
fewer tokens, less noise.
"""

from pathlib import Path
from typing import Any

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Loaded from the environment or the repo-root .env (never from servers.json)."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    github_token: str = ""
    github_repo: str = ""  # default "owner/repo" when a tool call omits one


class GitHubError(Exception):
    pass


def _labels(item: dict) -> list[str]:
    return [label["name"] for label in item.get("labels", [])]


def _issue(i: dict, *, full: bool = False) -> dict[str, Any]:
    out = {
        "number": i["number"],
        "title": i["title"],
        "state": i["state"],
        "labels": _labels(i),
        "author": (i.get("user") or {}).get("login"),
        "assignees": [a["login"] for a in i.get("assignees", [])],
        "comments": i.get("comments", 0),
        "created_at": i["created_at"],
        "updated_at": i["updated_at"],
        "url": i["html_url"],
    }
    if full:
        out["body"] = i.get("body") or ""
    return out


def _pr(p: dict, *, full: bool = False) -> dict[str, Any]:
    out = {
        "number": p["number"],
        "title": p["title"],
        "state": p["state"],
        "draft": p.get("draft", False),
        "author": (p.get("user") or {}).get("login"),
        "labels": _labels(p),
        "head": (p.get("head") or {}).get("ref"),
        "base": (p.get("base") or {}).get("ref"),
        "created_at": p["created_at"],
        "updated_at": p["updated_at"],
        "url": p["html_url"],
    }
    if full:
        out.update(
            body=p.get("body") or "",
            merged=p.get("merged", False),
            mergeable=p.get("mergeable"),
            additions=p.get("additions"),
            deletions=p.get("deletions"),
            changed_files=p.get("changed_files"),
        )
    return out


class GitHubClient:
    def __init__(
        self,
        token: str,
        default_repo: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.AsyncClient(
            base_url="https://api.github.com", headers=headers, timeout=20, transport=transport
        )
        self.default_repo = default_repo

    def _repo(self, repo: str | None) -> str:
        repo = repo or self.default_repo
        if not repo or repo.count("/") != 1:
            raise GitHubError("repo must be given as 'owner/name' (and no default is configured)")
        return repo

    async def _request(self, method: str, path: str, **kw) -> Any:
        resp = await self._http.request(method, path, **kw)
        if resp.is_error:
            try:
                detail = resp.json().get("message", resp.text)
            except ValueError:
                detail = resp.text
            raise GitHubError(f"GitHub API {resp.status_code}: {detail}")
        return resp.json()

    async def get_repo(self, repo: str | None = None) -> dict:
        r = await self._request("GET", f"/repos/{self._repo(repo)}")
        return {
            "full_name": r["full_name"],
            "description": r["description"],
            "default_branch": r["default_branch"],
            "language": r["language"],
            "open_issues_count": r["open_issues_count"],
            "stars": r["stargazers_count"],
            "pushed_at": r["pushed_at"],
            "url": r["html_url"],
        }

    async def list_issues(
        self,
        repo: str | None = None,
        state: str = "open",
        labels: str = "",
        limit: int = 30,
    ) -> list[dict]:
        # GitHub's issues endpoint also returns PRs; drop them so "issues" means issues.
        params = {"state": state, "per_page": min(limit * 2, 100), "sort": "updated"}
        if labels:
            params["labels"] = labels
        items = await self._request("GET", f"/repos/{self._repo(repo)}/issues", params=params)
        return [_issue(i) for i in items if "pull_request" not in i][:limit]

    async def get_issue(self, number: int, repo: str | None = None) -> dict:
        repo = self._repo(repo)
        issue = await self._request("GET", f"/repos/{repo}/issues/{number}")
        comments = await self._request(
            "GET", f"/repos/{repo}/issues/{number}/comments", params={"per_page": 20}
        )
        out = _issue(issue, full=True)
        out["recent_comments"] = [
            {"author": (c.get("user") or {}).get("login"), "body": c["body"], "at": c["created_at"]}
            for c in comments
        ]
        return out

    async def list_pull_requests(
        self, repo: str | None = None, state: str = "open", limit: int = 30
    ) -> list[dict]:
        params = {"state": state, "per_page": min(limit, 100), "sort": "updated", "direction": "desc"}
        items = await self._request("GET", f"/repos/{self._repo(repo)}/pulls", params=params)
        return [_pr(p) for p in items]

    async def get_pull_request(self, number: int, repo: str | None = None) -> dict:
        return _pr(await self._request("GET", f"/repos/{self._repo(repo)}/pulls/{number}"), full=True)

    async def search_issues(self, query: str, repo: str | None = None, limit: int = 20) -> list[dict]:
        q = f"{query} repo:{self._repo(repo)}"
        r = await self._request("GET", "/search/issues", params={"q": q, "per_page": min(limit, 100)})
        return [{**_issue(i), "is_pull_request": "pull_request" in i} for i in r["items"]]

    async def create_issue(
        self, title: str, body: str = "", labels: list[str] | None = None, repo: str | None = None
    ) -> dict:
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        return _issue(
            await self._request("POST", f"/repos/{self._repo(repo)}/issues", json=payload), full=True
        )

    async def aclose(self) -> None:
        await self._http.aclose()
