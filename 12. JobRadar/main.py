"""
main.py — FastAPI: the dashboard API.
"""

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select

import agents
import bus
import store
import workers
from store import (BAND_ORDER, MASTER_RESUME, SessionLocal, Job, JobOut,
                   PasteJob, classify_company, dedupe_key, init_db, tz_overlap)

_tasks = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    _tasks.extend(workers.start_workers())
    print(f"JobRadar ready · event bus: {bus.backend_name()}")
    yield
    for t in _tasks:
        t.cancel()
    await bus.close()


app = FastAPI(title="Project 12 — JobRadar", lifespan=lifespan)


@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "event_bus": bus.backend_name(),
        "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
        "bar": {"target_base_lpa": store.TARGET_BASE_LPA,
                "floor_base_lpa": store.FLOOR_BASE_LPA},
    }


@app.get("/api/metrics")
async def metrics():
    """Pipeline counters + queue depths — backpressure, made visible."""
    return {"counters": workers.METRICS, "bus": bus.stats()}


@app.post("/api/ingest")
async def ingest():
    """
    Pull from all sources. Returns as soon as jobs are INSERTED — the
    agents enrich them asynchronously off the bus. You see new rows
    immediately; bands and tiers fill in over the next minute.
    """
    return await workers.ingest_once()


@app.get("/api/jobs", response_model=list[JobOut])
async def list_jobs(
    band: str = Query(None),
    tier: str = Query(None),
    relocation: str = Query(None),
    min_fit: int = Query(0),
    include_red: bool = Query(True),
    limit: int = Query(100, le=300),
):
    """
    Ordering: band first (green → yellow → unknown → red), then fit score.

    RED IS NEVER FILTERED OUT — just sorted last. A company below your
    number can still be worth a conversation, and hiding it entirely
    makes the tool decide something you should decide.
    """
    async with SessionLocal() as db:
        q = select(Job).where(Job.status.in_(["classified", "enriched", "approved",
                                              "package_ready", "applied"]))
        if band:
            q = q.where(Job.band == band)
        if tier:
            q = q.where(Job.company_tier == tier)
        if relocation:
            q = q.where(Job.relocation_tier == relocation)
        if min_fit:
            q = q.where(Job.fit_score >= min_fit)
        rows = list((await db.execute(q.limit(limit))).scalars().all())

    if not include_red:
        rows = [r for r in rows if r.band != "RED"]

    rows.sort(key=lambda j: (BAND_ORDER.get(j.band, 9), -j.fit_score))
    return rows


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: int):
    async with SessionLocal() as db:
        job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Not found")
    return {c.name: getattr(job, c.name) for c in job.__table__.columns}


@app.post("/api/jobs/{job_id}/approve")
async def approve(job_id: int):
    """
    ⭐ APPROVAL → PACKAGE. This is the ONLY place resume tailoring runs.

    Not on discovery. ~5 tailoring calls a week instead of 200 — and
    fewer divergent versions of your career floating around.

    This does NOT submit anything. It prepares the materials and hands
    you the apply link. Auto-submitting violates ATS terms, gets you
    flagged as a bulk applicant, and can't be un-sent.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set")

    async with SessionLocal() as db:
        job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
        if not job:
            raise HTTPException(404, "Not found")
        title, company, desc = job.title, job.company, job.description
        missing = json.loads(job.missing_skills or "[]")

    tailored = await agents.tailor_resume(MASTER_RESUME, title, company, desc, missing)

    async with SessionLocal() as db:
        job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
        job.tailored_resume = tailored["tailored_resume"]
        job.cover_letter = tailored["cover_letter"]
        job.changes_summary = tailored["changes_summary"]
        job.status = "package_ready"
        await db.commit()
        url = job.apply_url

    return {"status": "package_ready", "apply_url": url,
            "changes_summary": tailored["changes_summary"]}


@app.post("/api/jobs/{job_id}/status")
async def set_status(job_id: int, value: str = Query(..., pattern="^(rejected|applied)$")):
    async with SessionLocal() as db:
        job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
        if not job:
            raise HTTPException(404, "Not found")
        job.status = value
        await db.commit()
    return {"status": value}


@app.post("/api/paste")
async def paste(body: PasteJob):
    """
    For jobs you found yourself on LinkedIn or Wellfound.

    You do the browsing; the agents still do the work — classification,
    fit scoring, salary research, banding. Same pipeline, entered
    manually at the front.
    """
    async with SessionLocal() as db:
        job = Job(source="manual", dedupe_key=dedupe_key(body.company, body.title),
                  company=body.company, title=body.title, location=body.location,
                  description=body.description, apply_url=body.apply_url,
                  company_tier=classify_company(body.company),
                  tz_overlap=tz_overlap(body.location), status="discovered")
        db.add(job)
        try:
            await db.commit()
            await db.refresh(job)
        except Exception:
            await db.rollback()
            raise HTTPException(409, "This job is already in your feed")

    await bus.publish(bus.TOPIC_DISCOVERED, {"job_id": job.id})
    return {"job_id": job.id, "status": "queued for enrichment"}


@app.get("/api/resume")
async def resume():
    return {"master_resume": MASTER_RESUME, "variants": store.RESUME_VARIANTS}