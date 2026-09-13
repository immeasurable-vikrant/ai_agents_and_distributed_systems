"""
main.py — FastAPI: the solve endpoint + a raw sandbox endpoint.
"""

import os

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import graph_ascii, solve
from sandbox import DOCKER_AVAILABLE, backend_name, run_code

app = FastAPI(title="Project 5 — SandboxExec")


class SolveRequest(BaseModel):
    request: str = Field(..., min_length=5, max_length=300)


class RunRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=5000)


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    """
    Reports which sandbox backend is active — because the security
    guarantees are genuinely different, and you should never be unsure
    which one you're running.
    """
    return {
        "status": "ok",
        "sandbox_backend": backend_name(),
        "docker_available": DOCKER_AVAILABLE,
        "warning": None if DOCKER_AVAILABLE else
                   "Docker not found — using the weaker subprocess fallback",
    }


@app.get("/api/graph")
async def show_graph():
    parent, sub = graph_ascii()
    return {"parent": parent, "subgraph": sub}


@app.post("/api/solve")
async def solve_endpoint(body: SolveRequest):
    """Full agent: understand → [write → run → fix] → explain."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set")
    try:
        return await solve(body.request)
    except Exception as e:
        print(f"[solve] {e}")
        raise HTTPException(500, f"Agent failed: {str(e)[:200]}")


@app.post("/api/run")
async def run_endpoint(body: RunRequest):
    """
    Run code directly, no LLM. This is the "attack your own sandbox"
    endpoint — paste hostile code and watch what the limits do.

    Notice it needs no API key: the sandbox is a backend feature,
    independent of the agent that happens to feed it.
    """
    return await run_code(body.code)