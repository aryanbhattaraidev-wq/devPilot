"""Create a handful of realistic issues in the test repo so DevPilot has data to reason about.

    uv run python scripts/seed_github.py

Idempotent-ish: skips titles that already exist. Uses GITHUB_TOKEN / GITHUB_REPO from .env.
"""

import asyncio

from devpilot.github_api import GitHubClient, Settings

ISSUES = [
    ("Login fails with 500 when password contains unicode", "Reproduce: sign up with 'pässwörd'. "
     "Server logs show UnicodeEncodeError in auth.hash(). Blocks some users from logging in.", ["bug", "critical"]),
    ("Add rate limiting to public API endpoints", "No throttling today; one client can saturate the service. "
     "Proposal: token bucket per API key, 100 req/min.", ["enhancement", "security"]),
    ("Update README with setup instructions", "New contributors can't get the project running. "
     "Document Python version, env vars and how to run tests.", ["documentation"]),
    ("Dashboard chart flickers on window resize", "Minor visual glitch in Chrome; no data loss.", ["bug", "low-priority"]),
    ("Migrate CI from Travis to GitHub Actions", "Travis builds are queueing for 30+ minutes. "
     "Blocking release work for everyone.", ["infrastructure", "high-priority"]),
]


async def main() -> None:
    s = Settings()
    gh = GitHubClient(s.github_token, s.github_repo)
    try:
        existing = {i["title"] for i in await gh.list_issues(state="all", limit=100)}
        for title, body, labels in ISSUES:
            if title in existing:
                print("skip   ", title)
                continue
            created = await gh.create_issue(title, body, labels)
            print(f"created #{created['number']} {title}")
    finally:
        await gh.aclose()


if __name__ == "__main__":
    asyncio.run(main())
