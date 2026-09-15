"""
main.py — FastAPI: the planner, plus a ReAct baseline to compare against.
"""

import os

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from planner import TOOL_DOCS, graph_ascii, run
from react_baseline import run_react

app = FastAPI(title="Project 10 — Planner Agent")


class GoalReq(BaseModel):
    goal: str = Field(..., min_length=10, max_length=300)
    mode: str = Field("planner", pattern="^(planner|react)$")


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "llm_configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.get("/api/graph")
async def show_graph():
    return {"diagram": graph_ascii(), "tools": TOOL_DOCS}


@app.post("/api/run")
async def run_goal(body: GoalReq):
    """
    Same goal, two agent architectures.

    The comparison is the point: watch the step counts and LLM call
    counts diverge, and notice that only the planner shows you its
    intentions BEFORE acting.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set")
    try:
        if body.mode == "planner":
            return {"mode": "planner", **await run(body.goal)}
        return {"mode": "react", **await run_react(body.goal)}
    except Exception as e:
        print(f"[run] {e}")
        raise HTTPException(500, f"Agent failed: {str(e)[:200]}")