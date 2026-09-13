"""
main.py — FastAPI: the research endpoints + the concurrency lab endpoints.
"""

import asyncio
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import concurrency
from agent import graph_ascii, run_research, run_research_sequential

app = FastAPI(title="Project 4 — Research Fanout")


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=300)
    mode: str = Field("parallel", pattern="^(parallel|sequential)$")


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "cpus": os.cpu_count()}


@app.get("/api/graph")
async def show_graph():
    return {"diagram": graph_ascii()}


@app.post("/api/research")
async def research(body: ResearchRequest):
    """
    Same work, two schedulings. Run both and compare total_ms.

    NOTE this endpoint is `async def` with everything awaited inside —
    so while one user's research is waiting on the LLM, this worker
    happily serves others. Experiment 5 in the lab shows what happens
    when you get that wrong.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set")
    try:
        if body.mode == "parallel":
            return {"mode": "parallel", **await run_research(body.question)}
        return {"mode": "sequential", **await run_research_sequential(body.question)}
    except Exception as e:
        print(f"[research] {e}")
        raise HTTPException(500, f"Research failed: {str(e)[:200]}")


# ============================================================
# THE CONCURRENCY LAB — no LLM needed, runs instantly
# ============================================================

@app.post("/api/lab/{experiment}")
async def lab(experiment: str):
    """
    Run one experiment from concurrency.py and return its numbers.

    ⚠️ exp2 (the GIL demo) spawns processes and burns CPU for a couple of
    seconds. It's `def`-free on purpose — `run_in_executor` would be the
    cleaner production pattern, but here we WANT to see the raw timings.
    """
    runners = {
        "1": concurrency.exp1_sequential_vs_gather,
        "2": lambda: asyncio.to_thread(concurrency.exp2_gil),
        "3": lambda: asyncio.to_thread(concurrency.exp3_threads_on_io),
        "4": concurrency.exp4_bounded,
        "5": concurrency.exp5_blocking_loop,
    }
    if experiment not in runners:
        raise HTTPException(404, "No such experiment (1-5)")

    result = runners[experiment]()
    return await result if asyncio.iscoroutine(result) else result