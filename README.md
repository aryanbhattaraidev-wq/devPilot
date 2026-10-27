# DevPilot

An AI developer assistant built on the Model Context Protocol (MCP). Ask it things like
"what are the highest-priority unresolved issues, and which should I work on next?" and it
connects to GitHub, your task database, and your project files and docs through MCP servers,
then answers from real data.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and run `uv sync`.
2. Copy `.env.example` to `.env` and fill in:
   - `GITHUB_TOKEN`: a fine-grained personal access token. Read access to issues and pull
     requests is enough; add Issues write only if you want DevPilot to create issues.
   - `GITHUB_REPO`: the default repository, as `owner/name`.
   - `OPENROUTER_API_KEY`: used for the chat model and for document embeddings.
3. Optionally start the remote demo server: `uv run python -m devpilot.servers.demo_http`.

## Usage

- `uv run devpilot servers` connects to every configured server and shows its tools,
  resources and prompts. A server that fails to start is reported, and the rest keep working.
- `uv run devpilot ask "your question"` asks one question. The model decides which tools to call.
- `uv run devpilot chat` starts an interactive conversation that remembers earlier turns.
- `uv run devpilot index` builds or refreshes the documentation search index.
- `uv run devpilot call github__list_issues '{"state":"open"}'` calls one tool directly.
- `uv run devpilot read project project://summary` reads an MCP resource.

## Configuration

Servers are listed in `servers.json`. A server is either `stdio` (DevPilot starts it as a
subprocess) or `streamable-http` (it is already running at a URL). Secrets are never put in
`servers.json`; each server reads them from `.env` itself.

Set `DEVPILOT_MODEL` to change the chat model. It must support tool calling. The default is
`openai/gpt-4o-mini`; a stronger model gives better prioritisation advice.

## Tests

Run `uv run pytest`. The tests use a scripted fake LLM and an offline hash embedder, so they
need no network or API keys.
