"""DevPilot chat UI (Streamlit). Talks to the FastAPI backend; run both:

    uv run devpilot serve
    uv run streamlit run ui/app.py

Streamlit re-runs this whole script on every interaction, so conversation state lives in
st.session_state and the script ends by re-running itself while a turn is in progress.
"""

import os
import time

import httpx
import streamlit as st

API = os.environ.get("DEVPILOT_API", "http://127.0.0.1:8000")
POLL_SECONDS = 0.5

st.set_page_config(page_title="DevPilot", page_icon=":compass:", layout="wide")


def api(method: str, path: str, **kwargs):
    response = httpx.request(method, f"{API}{path}", timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def reset_conversation() -> None:
    st.session_state.session_id = api("POST", "/sessions")["session_id"]
    st.session_state.transcript = []
    st.session_state.cursor = -1
    st.session_state.busy = False


def absorb(events: list[dict]) -> None:
    """Fold backend events into the transcript the page renders."""
    transcript = st.session_state.transcript
    for e in events:
        st.session_state.cursor = e["id"]
        kind = e["type"]
        if kind == "user":
            transcript.append({"kind": "user", "text": e["text"]})
        elif kind == "tool_call":
            transcript.append({"kind": "tool", "name": e["name"], "arguments": e["arguments"], "done": False})
        elif kind == "tool_result":
            running = next((t for t in transcript if t["kind"] == "tool" and t["name"] == e["name"] and not t["done"]), None)
            if running:
                running.update(done=True, chars=e["chars"], is_error=e["is_error"])
            else:  # denied calls report a result without a preceding tool_call
                transcript.append({"kind": "tool", "name": e["name"], "arguments": {}, "done": True,
                                   "chars": e["chars"], "is_error": e["is_error"]})
        elif kind == "approval_request":
            transcript.append({"kind": "approval", "status": "pending", **e})
        elif kind in ("answer", "error"):
            transcript.append({"kind": kind, "text": e["text"]})


def decide(item: dict, approved: bool, remember: bool = False) -> None:
    try:
        api("POST", f"/sessions/{st.session_state.session_id}/approvals/{item['approval_id']}",
            json={"approved": approved, "remember": remember})
        item["status"] = "approved" if approved else "denied"
    except httpx.HTTPStatusError:
        item["status"] = "expired"


# ---- connect -------------------------------------------------------------------------------

try:
    if "session_id" not in st.session_state:
        reset_conversation()
    servers = api("GET", "/servers")
except httpx.HTTPError as exc:
    st.error(f"Cannot reach the DevPilot API at {API}. Start it with `uv run devpilot serve`.\n\n{exc}")
    st.stop()

# ---- sidebar -------------------------------------------------------------------------------

with st.sidebar:
    st.header("DevPilot")
    if st.button("New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()
    st.subheader("MCP servers")
    for s in servers:
        up = s["status"] == "connected"
        st.markdown(f"{':green_circle:' if up else ':red_circle:'} **{s['name']}** `{s['transport']}`")
        if up:
            st.caption(f"{len(s['tools'])} tools, {s['resources']} resources, {len(s['prompts'])} prompts")
        else:
            st.caption(s["error"] or s["status"])

# ---- transcript ----------------------------------------------------------------------------

st.title("DevPilot")
if not st.session_state.transcript:
    st.caption("Try: *What are the highest-priority unresolved issues, and which should I work on next?*")

for item in st.session_state.transcript:
    kind = item["kind"]
    if kind == "user":
        st.chat_message("user").write(item["text"])
    elif kind == "tool":
        with st.chat_message("assistant", avatar=":material/build:"):
            state = "running..." if not item["done"] else ("error" if item["is_error"] else f"{item['chars']} chars")
            args = ", ".join(f"{k}={v!r}" for k, v in item["arguments"].items())
            st.caption(f"`{item['name']}({args})` - {state}")
    elif kind == "approval":
        with st.chat_message("assistant", avatar=":material/shield:"):
            box = st.error if item["destructive"] else st.warning
            box(f"**Approval needed:** `{item['tool']}`\n\n`{item['arguments']}`\n\n{item['reason']}")
            if item["status"] == "pending":
                cols = st.columns(3 if not item["destructive"] else 2)
                if cols[0].button("Approve", key=f"ok-{item['approval_id']}", type="primary"):
                    decide(item, True)
                    st.rerun()
                if cols[1].button("Deny", key=f"no-{item['approval_id']}"):
                    decide(item, False)
                    st.rerun()
                if not item["destructive"] and cols[2].button("Always allow this session", key=f"al-{item['approval_id']}"):
                    decide(item, True, remember=True)
                    st.rerun()
            else:
                st.caption(f"You {item['status']} this request.")
    elif kind == "answer":
        st.chat_message("assistant").markdown(item["text"])
    elif kind == "error":
        st.chat_message("assistant").error(item["text"])

# ---- input + polling -----------------------------------------------------------------------

prompt = st.chat_input("Ask DevPilot...", disabled=st.session_state.busy)
if prompt:
    try:
        api("POST", f"/sessions/{st.session_state.session_id}/messages", json={"text": prompt})
        st.session_state.busy = True
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.text)
    st.rerun()

if st.session_state.busy:
    waiting_for_human = any(t["kind"] == "approval" and t["status"] == "pending" for t in st.session_state.transcript)
    data = api("GET", f"/sessions/{st.session_state.session_id}/events", params={"after": st.session_state.cursor})
    before = len(st.session_state.transcript)
    absorb(data["events"])
    st.session_state.busy = data["busy"]
    if len(st.session_state.transcript) != before or not data["busy"]:
        st.rerun()  # new content to draw (or the turn finished)
    if not waiting_for_human:
        time.sleep(POLL_SECONDS)
        st.rerun()
