"""
workers.py — the multi-agent pipeline, wired over the event bus.

⭐ THIS FILE IS WHERE THE MULTI-AGENT ARCHITECTURE LIVES.

    ingest ──→ [jobs.discovered] ─┬→ classifier_worker  (group: classify)
                                  └→ metrics_worker     (group: metrics)
    classifier ─→ [jobs.classified] ──→ salary_worker   (group: salary)
    salary ─────→ [jobs.enriched]   ──→ dashboard

TWO PATTERNS IN ONE PICTURE:

  PUB/SUB FAN-OUT — classifier and metrics read the SAME topic with
  DIFFERENT group ids, so both see every job. Adding a third observer
  (an alerting agent, say) requires touching nothing that already exists.
  The producer doesn't know or care who's listening.

  QUEUE HANDOFF — classifier publishes to jobs.classified; the salary
  worker consumes it in a SEPARATE PROCESS. Fully decoupled: the salary
  agent can be slow, crash, or be restarted without ingestion noticing.
  Run two salary workers with the same group id and they split the load.

WHY NOT JUST CALL THE FUNCTIONS DIRECTLY: because then a slow LLM call
blocks ingestion, a crash loses the job, and scaling one stage means
scaling all of them. The log between stages is what buys independence.
"""

import asyncio
import json

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

import agents
import bus
import store
from store import Job, SessionLocal, band_salary, classify_company, dedupe_key, tz_overlap

PROFILE = {
    "seniority": "senior",
    "years": 6,
    "roles": ["Senior Backend Engineer", "AI Engineer", "Agentic Systems Engineer",
              "Platform Engineer"],
    "skills": ["Python", "FastAPI", "LangGraph", "PostgreSQL", "Redis", "Kafka",
               "Docker", "RAG", "multi-agent systems", "system design"],
}

# Counters the dashboard reads — a crude metrics agent's output.
METRICS = {"discovered": 0, "classified": 0, "enriched": 0,
           "green": 0, "sponsorship": 0, "skipped_low_fit": 0}


# ============================================================
# INGEST — pull from sources, dedupe, publish
# ============================================================

async def ingest_once() -> dict:
    """
    Fetch every source concurrently, dedupe, insert, publish.

    Ingestion does NO LLM work. It's fast and free, so it never has to
    wait on the expensive stages. That separation is the entire reason
    for the bus.
    """
    from sources import fetch_all

    raw_jobs, source_stats = await fetch_all()
    inserted = duplicates = 0

    for rj in raw_jobs:
        if not rj.company or not rj.title:
            continue

        async with SessionLocal() as db:
            job = Job(
                source=rj.source,
                dedupe_key=dedupe_key(rj.company, rj.title),
                company=rj.company, title=rj.title, location=rj.location,
                description=rj.description, apply_url=rj.apply_url,
                salary_raw=rj.salary_raw,
                # cheap, non-LLM enrichment done inline
                company_tier=classify_company(rj.company),
                tz_overlap=tz_overlap(rj.location),
                status="discovered",
            )
            db.add(job)
            try:
                await db.commit()
                await db.refresh(job)
                inserted += 1
            except IntegrityError:
                # THE DEDUPE, firing. The same job arrived from another
                # source. Expected, not an error.
                await db.rollback()
                duplicates += 1
                continue

        await bus.publish(bus.TOPIC_DISCOVERED, {"job_id": job.id})
        METRICS["discovered"] += 1

    return {"fetched": len(raw_jobs), "inserted": inserted,
            "duplicates": duplicates, "by_source": source_stats}


# ============================================================
# WORKER 1 — CLASSIFIER  (group: classify)
# ============================================================

async def classifier_worker():
    """Consumes jobs.discovered → classifies → publishes jobs.classified."""
    print("[classifier] started (group=classify)")
    async for event in bus.consume(bus.TOPIC_DISCOVERED, "classify"):
        try:
            job_id = event["job_id"]
            async with SessionLocal() as db:
                job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
                if not job:
                    continue
                title, company, loc, desc = job.title, job.company, job.location, job.description

            cls = await agents.classify(title, company, loc, desc)

            async with SessionLocal() as db:
                job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
                if not job:
                    continue
                job.employment_type = cls["employment_type"]
                job.relocation_tier = cls["relocation_tier"]
                job.visa_evidence = cls["visa_evidence"]
                job.work_region = cls["work_region"]
                job.archetype = cls["archetype"]
                if cls["work_region"]:
                    job.tz_overlap = tz_overlap(cls["work_region"])

                fit = await agents.score_fit(title, desc, PROFILE)
                job.fit_score = fit["fit_score"]
                job.fit_reason = fit["reason"]
                job.missing_skills = json.dumps(fit["missing_skills"])
                job.status = "classified"
                await db.commit()

            METRICS["classified"] += 1
            if cls["relocation_tier"] == "VISA_SPONSORSHIP":
                METRICS["sponsorship"] += 1

            await bus.publish(bus.TOPIC_CLASSIFIED,
                              {"job_id": job_id, "fit_score": fit["fit_score"]})
        except Exception as e:
            # A poison message must not kill the worker. In production this
            # would go to a dead-letter topic after N retries.
            print(f"[classifier] error: {e}")


# ============================================================
# WORKER 2 — METRICS  (group: metrics)  ← THE FAN-OUT PROOF
# ============================================================

async def metrics_worker():
    """
    Consumes the SAME topic as the classifier, with a DIFFERENT group id.

    Both workers receive EVERY job. That's pub/sub fan-out, and it's the
    reason you can bolt on a new observer without touching the producer
    or the existing consumer.

    Deliberately trivial — the architecture is the lesson, not the code.
    """
    print("[metrics] started (group=metrics)")
    async for event in bus.consume(bus.TOPIC_DISCOVERED, "metrics"):
        print(f"[metrics] saw job {event['job_id']} "
              f"(total discovered: {METRICS['discovered']})")


# ============================================================
# WORKER 3 — SALARY  (group: salary)  ← THE QUEUE HANDOFF
# ============================================================

async def salary_worker():
    """
    Consumes jobs.classified — a DIFFERENT topic, published by another
    worker. That's the handoff: two agents, two processes, no direct call.

    ⭐ THE COST GATE. Salary research is the strong model plus web
    searching — by far the most expensive step. Running it on a job that
    scored 20/100 is money burned on an application you'd never send.
    """
    print("[salary] started (group=salary)")
    async for event in bus.consume(bus.TOPIC_CLASSIFIED, "salary"):
        try:
            job_id, fit_score = event["job_id"], event.get("fit_score", 0)

            if fit_score < agents.FIT_GATE:
                METRICS["skipped_low_fit"] += 1
                async with SessionLocal() as db:
                    job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
                    if job:
                        job.status = "enriched"
                        job.band = "UNKNOWN"
                        job.salary_note = f"Skipped comp research (fit {fit_score} < {agents.FIT_GATE})"
                        await db.commit()
                continue

            async with SessionLocal() as db:
                job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
                if not job:
                    continue
                company, title, loc, raw = job.company, job.title, job.location, job.salary_raw

            sal = await agents.research_salary(company, title, loc, raw)
            band, warning = band_salary(sal["base_lpa"], sal["total_lpa"],
                                        sal["base_confidence"])

            async with SessionLocal() as db:
                job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
                if not job:
                    continue
                job.base_lpa = sal["base_lpa"]
                job.total_lpa = sal["total_lpa"]
                job.base_confidence = sal["base_confidence"]
                job.salary_sources = json.dumps(sal["sources"])
                job.salary_note = warning or sal.get("note", "")
                job.band = band
                job.status = "enriched"
                await db.commit()

            METRICS["enriched"] += 1
            if band == "GREEN":
                METRICS["green"] += 1

            await bus.publish(bus.TOPIC_ENRICHED, {"job_id": job_id, "band": band})
        except Exception as e:
            print(f"[salary] error: {e}")


# ============================================================
# STARTUP
# ============================================================

def start_workers() -> list[asyncio.Task]:
    """
    Three concurrent workers.

    In DEVELOPMENT these are asyncio tasks in one process — simple, and
    fine because every stage is I/O-bound.

    In PRODUCTION each would be its own container, scaled independently:
    the salary worker is the slow one, so you'd run four of those and one
    classifier. Same group id means they split the load automatically.
    That's what the bus buys you.
    """
    # Pre-register groups so the memory backend doesn't drop early events.
    bus.subscribe_group(bus.TOPIC_DISCOVERED, "classify")
    bus.subscribe_group(bus.TOPIC_DISCOVERED, "metrics")
    bus.subscribe_group(bus.TOPIC_CLASSIFIED, "salary")

    return [
        asyncio.create_task(classifier_worker()),
        asyncio.create_task(metrics_worker()),
        asyncio.create_task(salary_worker()),
    ]