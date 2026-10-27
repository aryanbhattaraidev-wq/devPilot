import httpx
import pytest

from devpilot.github_api import GitHubClient, GitHubError


def client(handler) -> GitHubClient:
    return GitHubClient("tok", "o/r", transport=httpx.MockTransport(handler))


def issue(n, title="t", pr=False):
    d = {
        "number": n, "title": title, "state": "open", "labels": [{"name": "bug"}],
        "user": {"login": "me"}, "assignees": [], "comments": 0,
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z",
        "html_url": f"https://github.com/o/r/issues/{n}", "body": "details",
    }
    if pr:
        d["pull_request"] = {}
    return d


async def test_list_issues_filters_out_pull_requests_and_sends_auth():
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["authorization"]
        seen["path"] = req.url.path
        return httpx.Response(200, json=[issue(1), issue(2, pr=True), issue(3)])

    out = await client(handler).list_issues()
    assert [i["number"] for i in out] == [1, 3]
    assert out[0]["labels"] == ["bug"]
    assert seen == {"auth": "Bearer tok", "path": "/repos/o/r/issues"}


async def test_api_errors_become_github_error():
    def handler(req):
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GitHubError, match="404.*Not Found"):
        await client(handler).get_issue(99)


async def test_repo_is_required_when_no_default():
    gh = GitHubClient("tok", "", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(GitHubError, match="owner/name"):
        await gh.get_repo()


async def test_search_always_sends_a_type_qualifier():
    queries = []

    def handler(req: httpx.Request):
        queries.append(req.url.params["q"])
        return httpx.Response(200, json={"items": [issue(1)]})

    gh = client(handler)
    await gh.search_issues("dark mode")
    await gh.search_issues("fix", kind="pr")
    assert queries == ["dark mode repo:o/r is:issue", "fix repo:o/r is:pull-request"]
    with pytest.raises(GitHubError, match="kind"):
        await gh.search_issues("x", kind="repo")
