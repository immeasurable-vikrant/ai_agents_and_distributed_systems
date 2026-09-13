"""
main.py — FastAPI: run suites, compare versions, browse the dataset.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dataset import DATASET, PROMPTS
from runner import USE_REAL_AGENT, run_comparison, run_suite

app = FastAPI(title="Project 9 — Eval Harness")


class RunReq(BaseModel):
    version: str = Field("v1", pattern="^(v1|v2)$")
    use_judge: bool = False
    real: bool | None = None      # None = auto (real if a key is present)


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "real_agent_available": USE_REAL_AGENT,
        "note": None if USE_REAL_AGENT else
                "No OPENAI_API_KEY — evals run against a deterministic fake agent",
    }


@app.get("/api/dataset")
async def dataset():
    return {"cases": DATASET, "prompts": PROMPTS}


@app.post("/api/run")
async def run(body: RunReq):
    """Run one prompt version through the whole dataset."""
    if body.use_judge and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "LLM judge requires OPENAI_API_KEY")
    try:
        return await run_suite(body.version, body.use_judge, body.real)
    except Exception as e:
        print(f"[run] {e}")
        raise HTTPException(500, f"Suite failed: {str(e)[:200]}")


@app.post("/api/compare")
async def compare(body: RunReq):
    """
    Run BOTH versions and diff them per case.

    This is the endpoint you'd wire into CI: fail the build when
    `comparison.regressions` is non-empty, regardless of whether the
    overall mean went up.
    """
    if body.use_judge and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "LLM judge requires OPENAI_API_KEY")
    try:
        return await run_comparison(body.use_judge, body.real)
    except Exception as e:
        print(f"[compare] {e}")
        raise HTTPException(500, f"Comparison failed: {str(e)[:200]}")