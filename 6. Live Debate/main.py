"""
main.py — FastAPI: a WebSocket for the live debate + REST for approval.

WHY A WEBSOCKET AND NOT SSE HERE: this connection carries tokens DOWN
(server → client) and nothing meaningful UP. SSE would genuinely be the
simpler correct choice for that. WebSocket is used because the next
project scales this across processes with a Redis pub/sub backplane, and
having the bidirectional channel already in place makes that step
smaller. Worth knowing you had a cheaper option.
"""

import json
import os
import uuid
from typing import Literal

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import close_graph, get_graph, get_state, graph_ascii, resume_debate, stream_debate


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise the graph (and its checkpointer DB connection) HERE, on
    # the app's event loop. See close_graph()'s note in agent.py for why
    # lazy initialisation from an arbitrary loop is a trap.
    await get_graph()
    yield
    await close_graph()


app = FastAPI(title="Project 6 — Live Debate", lifespan=lifespan)


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]


class StartRequest(BaseModel):
    topic: str = Field(..., min_length=10, max_length=300)


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "llm_configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.get("/api/graph")
async def show_graph():
    return {"diagram": graph_ascii()}


@app.post("/api/debates")
async def create_debate(body: StartRequest):
    """
    Mint a thread_id. The debate itself runs over the WebSocket — this
    just hands you the key.
    """
    return {"thread_id": f"d-{uuid.uuid4().hex[:10]}", "topic": body.topic}


@app.websocket("/ws/debate/{thread_id}")
async def debate_ws(ws: WebSocket, thread_id: str):
    """
    Streams the whole debate: tokens, turn boundaries, verdict, and the
    interrupt if one fires.

    NOTE the client sends the topic as its first message. That's the one
    thing flowing upward on this socket — see the WebSocket-vs-SSE note
    at the top of the file.
    """
    await ws.accept()
    try:
        topic = (await ws.receive_text()).strip()
        if len(topic) < 10:
            await ws.send_json({"type": "error", "message": "Topic too short"})
            await ws.close()
            return
        if not os.getenv("OPENAI_API_KEY"):
            await ws.send_json({"type": "error", "message": "OPENAI_API_KEY is not set"})
            await ws.close()
            return

        async for event in stream_debate(topic, thread_id):
            await ws.send_json(event)

        await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        # The client closed the tab. The debate's STATE is already in the
        # checkpointer, so nothing is lost — reconnect and read it back.
        pass
    except Exception as e:
        print(f"[ws] {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)[:200]})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/api/debates/{thread_id}")
async def read_debate(thread_id: str):
    """
    Read a debate's saved state — including whether it's paused.

    This is how a reloaded page recovers: the turns, verdict, and pause
    status all come from the checkpoint database, not from the browser
    and not from whatever process ran the debate.
    """
    state = await get_state(thread_id)
    if not state["exists"]:
        raise HTTPException(404, "No such debate")
    return state


@app.post("/api/debates/{thread_id}/decide")
async def decide(thread_id: str, body: ApprovalRequest):
    """
    ⭐ THE RESUME ENDPOINT.

    Notice how little this does: it doesn't wake up a specific process,
    doesn't look for an in-memory Event, doesn't care which process
    originally paused the graph — or whether that process still exists.

    It loads the checkpoint, injects the decision, and continues. That's
    what makes the pause durable rather than merely convenient.
    """
    state = await get_state(thread_id)
    if not state["exists"]:
        raise HTTPException(404, "No such debate")
    if not state["paused_at"]:
        # A state-machine guard: approving something that isn't waiting
        # is a client bug. Failing loudly surfaces it instead of silently
        # doing nothing.
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "This debate is not waiting for approval")
    try:
        return await resume_debate(thread_id, body.decision)
    except Exception as e:
        print(f"[resume] {e}")
        raise HTTPException(500, f"Resume failed: {str(e)[:200]}")