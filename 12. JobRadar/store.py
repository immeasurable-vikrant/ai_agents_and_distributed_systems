"""
store.py — models, DTOs, salary banding, and the resume layers.
"""

import hashlib
import json
import os
import re
from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./jobradar.db")
engine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ============================================================
# YOUR BAR
# ============================================================
TARGET_BASE_LPA = 40        # 🟢 at or above
FLOOR_BASE_LPA = 35         # 🔴 below this
CURRENT_CTC_LPA = 29


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(40), index=True)
    dedupe_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    #   ⭐ THE DEDUPE MECHANISM — a DB unique constraint, not app logic.
    #   The same job on Greenhouse, Adzuna and RemoteOK produces the same
    #   key; inserts 2 and 3 raise IntegrityError and we skip them. The
    #   database enforces "one row per real job" so no code path can forget.

    company: Mapped[str] = mapped_column(String(200), index=True)
    title: Mapped[str] = mapped_column(String(300))
    location: Mapped[str] = mapped_column(String(300), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    apply_url: Mapped[str] = mapped_column(Text, default="")
    salary_raw: Mapped[str] = mapped_column(String(300), default="")

    # --- classifier agent ---
    employment_type: Mapped[str] = mapped_column(String(30), default="unknown")
    relocation_tier: Mapped[str] = mapped_column(String(30), default="unknown", index=True)
    visa_evidence: Mapped[str] = mapped_column(Text, default="")
    work_region: Mapped[str] = mapped_column(String(60), default="")
    tz_overlap: Mapped[int] = mapped_column(Integer, default=0)
    archetype: Mapped[str] = mapped_column(String(30), default="backend")
    #   ↑ which resume VARIANT this job needs — see RESUME_VARIANTS below

    # --- fit agent ---
    fit_score: Mapped[int] = mapped_column(Integer, default=0, index=True)
    fit_reason: Mapped[str] = mapped_column(Text, default="")
    missing_skills: Mapped[str] = mapped_column(Text, default="[]")

    # --- salary agent: base and total kept SEPARATE, deliberately ---
    base_lpa: Mapped[int] = mapped_column(Integer, default=0)
    total_lpa: Mapped[int] = mapped_column(Integer, default=0)
    base_confidence: Mapped[str] = mapped_column(String(20), default="unknown")
    salary_note: Mapped[str] = mapped_column(Text, default="")
    salary_sources: Mapped[str] = mapped_column(Text, default="[]")
    band: Mapped[str] = mapped_column(String(24), default="UNKNOWN", index=True)

    company_tier: Mapped[str] = mapped_column(String(24), default="STARTUP", index=True)
    status: Mapped[str] = mapped_column(String(24), default="new", index=True)
    tailored_resume: Mapped[str] = mapped_column(Text, default="")
    cover_letter: Mapped[str] = mapped_column(Text, default="")
    changes_summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now())


# ============================================================
# SALARY BANDING — the part that needed the most care
# ============================================================

def band_salary(base_lpa: int, total_lpa: int, base_confidence: str) -> tuple[str, str]:
    """
    Returns (band, warning).

    THE PROBLEM THIS SOLVES: most public comp data reports TOTAL, not
    base. A "₹58L" figure is often ₹34L base + ₹18L ESOPs + ₹6L bonus —
    which FAILS a ₹40L base bar while looking like it clears it easily.

    So base and total are never conflated. When base can't be confirmed,
    the job is surfaced with an explicit warning rather than being
    silently scored on a number that doesn't mean what it appears to.

      🟢 GREEN            base confirmed ≥ 40
      🟡 YELLOW_HIGH      base confirmed 35-40   (below target, above floor)
      🟡 YELLOW_UNCONFIRMED  total ≥ 40 but base unknown  ← the dangerous case
      🔴 RED              base or total < 35
      ⚪ UNKNOWN          no data at all

    NOTHING is filtered out. Red is sorted last and dimmed, because a
    company below your number can still be worth a conversation.
    """
    confirmed = base_confidence in ("high", "medium")

    if base_lpa and confirmed:
        if base_lpa >= TARGET_BASE_LPA:
            return "GREEN", ""
        if base_lpa >= FLOOR_BASE_LPA:
            return "YELLOW_HIGH", f"Base ₹{base_lpa}L — above your ₹{FLOOR_BASE_LPA}L floor, below your ₹{TARGET_BASE_LPA}L target"
        return "RED", f"Base ₹{base_lpa}L is below your ₹{FLOOR_BASE_LPA}L floor"

    if total_lpa >= TARGET_BASE_LPA:
        return "YELLOW_UNCONFIRMED", (
            f"⚠️ Base NOT confirmed. Total comp ~₹{total_lpa}L — but that may "
            f"include ESOPs/bonus. Ask for the base split in the screening call."
        )

    if total_lpa and total_lpa < FLOOR_BASE_LPA:
        return "RED", f"Total comp ~₹{total_lpa}L is below your floor"

    return "UNKNOWN", "No comp data found — ask about base in the first call"


BAND_ORDER = {"GREEN": 0, "YELLOW_HIGH": 1, "YELLOW_UNCONFIRMED": 2,
              "UNKNOWN": 3, "RED": 4}


# ============================================================
# COMPANY TIERS — scoped to your ₹40L+ bar
# ============================================================
# These are companies that realistically pay ₹40L+ base at 6 YOE in
# India, or pay in USD/EUR for remote roles.
#
# ⚠️ Hand-maintained and WILL go stale. Treat as a starting point.

TIERS = {
    "FAANG": {"google", "microsoft", "amazon", "meta", "apple", "netflix"},
    "TOP_TECH": {"stripe", "databricks", "atlassian", "uber", "confluent",
                 "rubrik", "salesforce", "adobe", "nvidia", "linkedin",
                 "airbnb", "dropbox", "twilio", "datadog", "snowflake",
                 "openai", "anthropic", "figma", "canva", "coinbase"},
    "HIGH_PAY_INDIA": {"razorpay", "cred", "zerodha", "postman", "navi",
                       "zepto", "sprinklr", "browserstack", "freshworks",
                       "phonepe", "groww", "meesho", "swiggy", "flipkart"},
    "YC": {"vercel", "supabase", "retool", "posthog", "replit", "deel",
           "modal", "hex", "linear", "clickhouse"},
}

TIER_COLORS = {"FAANG": "#7c3aed", "TOP_TECH": "#2563eb",
               "HIGH_PAY_INDIA": "#16a34a", "YC": "#ea580c", "STARTUP": "#6b7280"}


def classify_company(name: str) -> str:
    """A dict lookup, not an LLM call — company tier is a stable fact."""
    tokens = {name.lower().strip()} | set(name.lower().split())
    for tier, members in TIERS.items():
        if tokens & members:
            return tier
    return "STARTUP"


# ============================================================
# TIMEZONE OVERLAP (from IST)
# ============================================================
UTC_OFFSETS = {
    "india": 5.5, "bangalore": 5.5, "gurgaon": 5.5, "delhi": 5.5, "mumbai": 5.5,
    "singapore": 8.0, "germany": 1.0, "berlin": 1.0, "netherlands": 1.0,
    "amsterdam": 1.0, "europe": 1.0, "emea": 1.0, "eu": 1.0,
    "uk": 0.0, "london": 0.0, "ireland": 0.0, "dublin": 0.0,
    "us east": -5.0, "new york": -5.0, "us": -6.0, "usa": -6.0,
    "united states": -6.0, "us west": -8.0, "san francisco": -8.0,
    "canada": -5.0, "worldwide": 5.5, "global": 5.5, "anywhere": 5.5,
    #   ⚠️ NOTE "remote" is deliberately NOT a key. It appears inside
    #   region-specific strings like "Remote - US", where longest-match
    #   would otherwise pick it and score a US role as same-timezone.
    #   A bare "remote" tells you nothing about a timezone; "worldwide"
    #   does (they hire anywhere, so they're async-friendly).
}


def tz_overlap(location: str) -> int:
    """Hours of 9-6 overlap with IST. 9 = same zone, 0-2 = night shift."""
    if not location:
        return 0
    t = location.lower()
    off = next((UTC_OFFSETS[r] for r in sorted(UTC_OFFSETS, key=len, reverse=True)
                if r in t), None)
    if off is None:
        return 0
    ist_s, ist_e = 9 - 5.5, 18 - 5.5
    job_s, job_e = 9 - off, 18 - off
    return max(0, int(round(min(ist_e, job_e) - max(ist_s, job_s))))


# ============================================================
# RESUME — three layers
# ============================================================
# LAYER 1: master (you write once, never modified)
# LAYER 2: variants by archetype (generated once, REUSED)
# LAYER 3: per-JD tailoring — only on APPROVAL, not on discovery
#
# WHY NOT REWRITE PER JOB: 200 discovered jobs × a strong-model rewrite
# is real money for applications you'll never send — and 200 slightly
# different versions of your career is a consistency risk if two reach
# the same company. Tailoring happens ~5×/week, not 200×.

RESUME_VARIANTS = ["backend", "ai_agentic", "platform"]

MASTER_RESUME = """# Singh
Senior Software Engineer · Gurgaon, India (open to relocation: Europe / US / Singapore)

## Summary
Engineer with 6+ years across product and services companies, moving from
frontend/fullstack into backend and AI/agentic systems. Built production AI
agents handling lead generation, ticket triage and review response. Strong on
distributed-systems fundamentals: concurrency, caching, messaging, multi-tenancy.

## Experience

### Birdeye — Software Engineer (1.7 years)
- Built production AI agents for lead generation, ticketing and review response
- Designed multi-agent workflows with tool calling and structured output
- Worked with LangGraph, LangChain and LangSmith for orchestration and tracing
- Shipped fullstack features across React frontends and Python/Node services

### Bangalore product startup — Software Engineer (8 months)
- Owned frontend architecture and contributed to backend API design
- Improved page performance and reduced time-to-interactive on core flows

### Gurgaon services company — Software Engineer (3.8 years)
- Delivered fullstack applications for multiple enterprise clients
- Built reusable component libraries and shared internal tooling
- Worked across the stack: React, Node.js, REST APIs, relational databases

### Bangalore startup — Software Engineer (3 months)
- Early-stage product work across frontend and API integration

## Independent projects (self-directed, 2025-2026)
- **Agent platform**: microservices with an API gateway, RS256 service-to-service
  auth, database-per-service, and an MCP server exposing agent tools
- **Multi-agent systems**: supervisor/specialist delegation, plan-and-execute with
  replanning, parallel fan-out with LangGraph
- **RAG**: naive → corrective → self-RAG pipelines with grading, query rewriting,
  grounding and usefulness checks; multi-tenant isolation with defense in depth
- **Agent memory**: checkpointed short-term memory plus long-term memory across
  conversations (LangGraph Store, mem0)
- **Evals**: heuristic, LLM-judge, trajectory and regression evaluators with
  per-case regression detection
- **Infrastructure**: Kafka event bus, Redis caching and rate limiting (Lua
  atomicity), Docker-sandboxed code execution, WebSocket streaming, durable
  human-in-the-loop

## Skills
Python · FastAPI · LangGraph · LangChain · LangSmith · PostgreSQL · Redis · Kafka
Docker · SQLAlchemy · async concurrency · RAG · multi-agent orchestration · MCP
React · TypeScript · Node.js · system design

## Education
B.Tech, Computer Science
"""


# ============================================================
# DTOs
# ============================================================

class JobOut(BaseModel):
    id: int
    source: str
    company: str
    title: str
    location: str
    apply_url: str
    employment_type: str
    relocation_tier: str
    visa_evidence: str
    tz_overlap: int
    archetype: str
    company_tier: str
    fit_score: int
    fit_reason: str
    base_lpa: int
    total_lpa: int
    base_confidence: str
    salary_note: str
    band: str
    status: str
    model_config = {"from_attributes": True}


class PasteJob(BaseModel):
    company: str = Field(..., min_length=1, max_length=200)
    title: str = Field(..., min_length=1, max_length=300)
    location: str = ""
    apply_url: str = ""
    description: str = Field(..., min_length=50, max_length=20000)


def dedupe_key(company: str, title: str) -> str:
    """
    Normalized company+title hash.

    Imperfect on purpose: a company posting "Senior Backend Engineer" for
    two different teams collapses into one row. Seeing the same job four
    times is a bigger annoyance than missing a near-duplicate, so the
    trade leans this way.
    """
    def norm(s: str) -> str:
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        s = re.sub(r"\s+", " ", s)
        for noise in (" inc", " ltd", " llc", " technologies", " labs", " india",
                      " remote", " full time", " fulltime", " m f d", " f m x"):
            s = s.replace(noise, "")
        return s.strip()
    return hashlib.sha256(f"{norm(company)}::{norm(title)}".encode()).hexdigest()[:40]


async def init_db():
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)