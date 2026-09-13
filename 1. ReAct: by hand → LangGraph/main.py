"""
main.py — FastAPI app.

The BACKEND half of Project 1. No database yet (that's Project 2) — this
covers the layer that sits in front of it: routing, DTOs, validation,
config, health checks, error handling.
"""

import os
import time

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent_graph import graph_ascii, run_graph_agent
from agent_manual import run_manual_agent

# ============================================================
# CONFIG — from environment, never hardcoded
# ============================================================
# The 12-factor rule: anything that differs between your laptop and
# production is an env var. Hardcoding an API key means it ends up in git.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
APP_NAME = "Project 1 — ReAct: By Hand vs LangGraph"

app = FastAPI(title=APP_NAME)


# ============================================================
# DTOs — Data Transfer Objects (Pydantic models)
# ============================================================
# A DTO defines the shape of data crossing your API boundary. Two jobs:
#
#   1. VALIDATION — bad input is rejected with a 422 before your code runs.
#      You never write `if not question: raise ...` by hand.
#   2. CONTRACT — the response model controls exactly what leaves your
#      system. In Project 2, when models have a `password_hash` column,
#      this is what stops it reaching the client.
#
# Request DTO ≠ Response DTO, deliberately. The client sends a question;
# it receives an answer, a trace, timing, and which engine ran. Different
# shapes for different directions.

class AskRequest(BaseModel):
    """What the client SENDS."""
    question: str = Field(
        ...,                          # ... means required
        min_length=3,
        max_length=500,               # bounds input cost — every char is tokens
        examples=["What's the weather in Delhi?"],
    )


class TraceStep(BaseModel):
    """One step of the agent's reasoning. Nested inside AskResponse."""
    step: int
    type: str                         # tool_call | observation | answer
    tool: str | None = None
    args: dict | None = None
    result: str | dict | None = None
    content: str | None = None


class AskResponse(BaseModel):
    """What the client RECEIVES."""
    engine: str                       # "manual" | "langgraph"
    answer: str
    steps: int
    elapsed_ms: int
    trace: list[TraceStep]
    budget_exceeded: bool = False


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def ui():
    """Serve the comparison UI."""
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    """
    Health check. A load balancer pings this to decide whether to send
    traffic here — if it fails, this instance is pulled from rotation.
    Cheap by design: no DB call, no LLM call, just "am I up and configured".
    """
    return {"status": "ok", "llm_configured": bool(OPENAI_API_KEY)}


@app.get("/api/graph")
async def show_graph():
    """The LangGraph structure, drawn as ASCII."""
    return {"diagram": graph_ascii()}


async def _run(engine: str, body: AskRequest) -> AskResponse:
    """Shared logic for both endpoints — one place to change timing/errors."""
    if not OPENAI_API_KEY:
        # 503, not 500: the service is fine, it's just not configured.
        # Status codes are part of your API contract — pick them deliberately.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not set",
        )

    t0 = time.perf_counter()
    try:
        runner = run_manual_agent if engine == "manual" else run_graph_agent
        result = await runner(body.question)
    except Exception as e:
        # Never leak a raw stack trace to a client — it exposes internals
        # and is useless to them. Log the detail, return something clean.
        print(f"[{engine}] failed: {e}")
        raise HTTPException(500, detail=f"Agent failed: {str(e)[:200]}")

    return AskResponse(
        engine=engine,
        answer=result["answer"],
        steps=result["steps"],
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        trace=result["trace"],
        budget_exceeded=result.get("budget_exceeded", False),
    )


# `async def` because everything inside awaits on the network (the LLM API).
# While one request waits, the event loop serves others on the same worker.
# A blocking call here would freeze every concurrent request — that's the
# subject of Project 4.

@app.post("/api/ask/manual", response_model=AskResponse)
async def ask_manual(body: AskRequest):
    """Run the hand-rolled ReAct loop."""
    return await _run("manual", body)


@app.post("/api/ask/graph", response_model=AskResponse)
async def ask_graph(body: AskRequest):
    """Run the LangGraph version."""
    return await _run("langgraph", body)