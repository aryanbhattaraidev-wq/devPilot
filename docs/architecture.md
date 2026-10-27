# DevPilot architecture

DevPilot follows the standard MCP shape: one **host** application, several **servers**, and
one **client connection** per server.

## Roles

- **Host**: the DevPilot application. It owns the chat interface, talks to the LLM, and decides
  what the model is allowed to do. The model never connects to a server directly.
- **Client**: a connection object inside the host, one per server. `MCPClientManager` in
  `devpilot/client/manager.py` holds all of them.
- **Server**: a separate program that exposes capabilities over MCP. DevPilot has servers for
  GitHub, the project task database, the filesystem and documentation search.

## Transports

MCP messages are JSON-RPC. Two transports are used:

- **stdio**: the host starts the server as a subprocess and exchanges messages over its
  stdin and stdout. This is used for local servers. Because stdout carries the protocol, a
  stdio server must never print to it; logs go to stderr.
- **Streamable HTTP**: the server is a long-running web service at a URL such as
  `http://127.0.0.1:8765/mcp`. This is used for remote servers and can serve many clients.

## Connection lifecycle and failure handling

Each server connection lives in its own asyncio task, because the transport's cancel scopes
must be entered and exited in the same task. On connect the client sends `initialize`, then
lists tools, resources and prompts. Servers connect concurrently. If one fails to spawn, times
out, or crashes mid-session, it is marked `failed` with the reason and the others keep working.

Tool names are namespaced as `server__tool`, for example `github__list_issues`, so two servers
can expose the same tool name and a call can be routed to its owner.

## The three server primitives

- **Tools** are actions chosen by the model: listing issues, creating a task, searching docs.
- **Resources** are read-only data chosen by the application, such as `project://summary`.
- **Prompts** are reusable templates chosen by the user, such as `plan_next_work`.

## The agent loop

`devpilot/agent.py` converts MCP tool definitions to the LLM's function-calling format (both use
JSON Schema), sends the conversation to the model, and executes any tool calls the model requests
by routing them to the owning server. Independent calls run concurrently. Results, including
errors, are fed back until the model produces a final answer. A cap of 12 rounds stops runaway loops.

## Human approval for sensitive actions

Servers describe each tool with annotations: `readOnlyHint`, `destructiveHint` and
`openWorldHint`. The host, not the server, turns those hints into policy, because a server could
misreport itself. The MCP defaults are pessimistic: a write is assumed destructive and a tool is
assumed to reach the outside world, so a missing annotation fails safe.

`ApprovalGate` in `devpilot/approval.py` runs inside the agent loop, between the model asking for
a tool and the tool running. Read-only tools and local non-destructive writes run immediately.
Destructive tools always ask, and are never remembered. External non-destructive writes ask, and
the user may allow them for the rest of the session. The gate fails closed: if the approver
errors, the action is denied. Parallel tool calls are asked one at a time. A denial is returned to
the model as a tool result saying the action was not executed, so it can explain and offer
alternatives instead of retrying.

## HTTP API and web UI

`devpilot/api.py` is a FastAPI backend and `ui/app.py` is a Streamlit chat front end. One shared
client manager serves all sessions, and each session has its own agent, history and gate. A turn
can pause while it waits for a human, so the UI polls an append-only event log (user message, tool
call, tool result, approval request, answer) instead of waiting on a single response. Approvals are
answered by a separate request that resolves the waiting future.

## Documentation search (RAG)

Retrieval-augmented generation fetches relevant passages at question time instead of relying on
what the model remembers.

1. **Ingestion**: files under the project root are read, skipping secrets such as `.env`, build
   output and files over 200 KB.
2. **Chunking**: Markdown is split on headings so each chunk covers one topic; long sections are
   packed to about 1200 characters with a small overlap. Every chunk records its line range.
3. **Embedding**: each chunk becomes a vector via an embedding model.
4. **Storage**: vectors and metadata are stored in a Chroma collection using cosine similarity.
5. **Retrieval**: the question is embedded the same way and the nearest chunks are returned
   with a citation in the form `path:start-end (heading)`.

Re-indexing is incremental. A file is re-embedded only when its content hash changes, and chunks
from deleted files are removed. A different embedding model gets its own collection, because
vectors from different models cannot be compared.

## Data layer

- SQLite (`devpilot.db`) stores tasks and notes for the project server.
- The GitHub REST API supplies issues and pull requests.
- Local repository files are served by the filesystem server.
- Chroma (`.chroma/`) stores document embeddings.
