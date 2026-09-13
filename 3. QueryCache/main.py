"""
main.py — FastAPI: chat endpoint with rate limiting, thread management.
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import store
from agent import chat as run_chat
from agent import get_history
from migrations import upgrade


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MIGRATIONS RUN ON STARTUP, not create_all().
    # The app should never serve traffic against a schema that doesn't
    # match what the code expects — so "container healthy" implies
    # "schema current".
    print("Running migrations...")
    await upgrade()
    yield


app = FastAPI(title="Project 3 — QueryCache", lifespan=lifespan)


# ============================================================
# DTOs
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    thread_id: str | None = None       # None → start a new conversation


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    trace: list[dict]
    summarized: bool
    message_count: int
    summary: str


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    """
    Reports Redis separately from the app. The app is USABLE without
    Redis (cache misses, rate limiting fails open) — so Redis being down
    is degraded, not dead. Health checks should say which.
    """
    redis_ok = False
    try:
        await store.get_redis().ping()
        redis_ok = True
    except Exception:
        pass
    return {"status": "ok", "redis": "up" if redis_ok else "down (degraded)"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set")

    # ---- RATE LIMIT ----
    # Keyed by IP here for simplicity. In a real app: the authenticated
    # user id, so one user on many devices shares one budget and a shared
    # office IP doesn't throttle everyone.
    client = request.client.host if request.client else "anon"
    if not await store.check_rate_limit(client, limit=20, window=60):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded — 20 messages/minute",
            headers={"Retry-After": "60"},   # tell the client WHEN to retry
        )

    # ---- THREAD ----
    # No thread_id means a new conversation. The id is the memory key —
    # generating a fresh one is how you "start over".
    thread_id = body.thread_id or f"t-{uuid.uuid4().hex[:12]}"
    await store.get_or_create_conversation(thread_id)

    try:
        result = await run_chat(body.message, thread_id)
    except Exception as e:
        print(f"[agent] {e}")
        raise HTTPException(500, f"Agent failed: {str(e)[:200]}")

    await store.set_title(thread_id, body.message)
    return ChatResponse(thread_id=thread_id, **result)


@app.get("/api/threads")
async def threads():
    convs = await store.list_conversations()
    return [
        {"thread_id": c.thread_id, "title": c.title,
         "created_at": c.created_at.isoformat() if c.created_at else None}
        for c in convs
    ]


@app.get("/api/threads/{thread_id}/history")
async def history(thread_id: str):
    """
    Loads a conversation from the CHECKPOINT DB.

    This is the proof of persistence: the messages aren't in the browser
    and aren't in this process's memory. Restart the server, reload the
    page, the conversation is still there.
    """
    return {"thread_id": thread_id, "messages": await get_history(thread_id)}