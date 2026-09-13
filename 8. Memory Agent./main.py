"""
main.py — FastAPI: chat with two memory scopes, plus memory management.
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import memory
from agent import chat as run_chat
from agent import close_graph, get_graph, get_history, graph_ascii


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise on the app's loop — see close_graph()'s note.
    await get_graph()
    yield
    await close_graph()


app = FastAPI(title="Project 8 — Memory Agent", lifespan=lifespan)


class ChatReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(..., min_length=1, max_length=50)
    thread_id: str | None = None


class MemoryReq(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=50)
    fact: str = Field(..., min_length=3, max_length=300)
    category: str = "general"


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "memory_backend": memory.backend_name(),
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/api/graph")
async def show_graph():
    return {"diagram": graph_ascii()}


@app.post("/api/chat")
async def chat(body: ChatReq):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set")

    # No thread_id = a brand new conversation. The user_id stays the
    # same, which is what makes "fresh chat, but it still knows me"
    # possible — that's the whole point of the project.
    thread_id = body.thread_id or f"t-{uuid.uuid4().hex[:10]}"
    try:
        result = await run_chat(body.message, body.user_id, thread_id)
    except Exception as e:
        print(f"[chat] {e}")
        raise HTTPException(500, f"Chat failed: {str(e)[:200]}")
    return {"thread_id": thread_id, **result}


@app.get("/api/history/{thread_id}")
async def history(thread_id: str):
    """STM — this conversation only, from the checkpointer."""
    return {"messages": await get_history(thread_id)}


@app.get("/api/memories/{user_id}")
async def memories(user_id: str):
    """LTM — this user, across every conversation they've ever had."""
    return await memory.all_memories(user_id)


@app.post("/api/memories", status_code=status.HTTP_201_CREATED)
async def add_memory(body: MemoryReq):
    """Manually teach the agent a fact, bypassing extraction."""
    return await memory.save_memory(body.user_id, body.fact, body.category)


@app.delete("/api/memories/{user_id}/{mem_id}", status_code=status.HTTP_204_NO_CONTENT)
async def forget(user_id: str, mem_id: str):
    """
    The right to be forgotten, as an endpoint.

    Worth noticing: this is trivial with LTM because memories are
    discrete, addressable records. Deleting a fact from STM would mean
    surgically editing a conversation history — far messier. Structured
    memory is easier to govern than raw transcripts.
    """
    await memory.delete_memory(user_id, mem_id)