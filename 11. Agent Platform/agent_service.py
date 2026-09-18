"""
agent_service.py — the AGENT microservice.

Runs on :8002, owns its OWN database, and verifies tokens using BOTH
strategies so you can compare them.

Contains the SUPERVISOR pattern: an orchestrator that delegates to
specialist subagents instead of doing the work itself.
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal, TypedDict

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from jose import JWTError, jwt
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import operator

AUTH_URL = os.getenv("AUTH_URL", "http://localhost:8001")
VERIFY_MODE = os.getenv("VERIFY_MODE", "local")     # local | remote
MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")

# ⭐ A SECOND, SEPARATE DATABASE.
DATABASE_URL = os.getenv("AGENT_DB_URL", "sqlite+aiosqlite:///./agent.db")
engine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)

    org_id: Mapped[int] = mapped_column(index=True)
    #   ⚠️ NO FOREIGN KEY to organizations — and this is THE cost of
    #   database-per-service, not an oversight.
    #
    #   The orgs table lives in the auth service's database. A foreign
    #   key cannot span two databases. So referential integrity that the
    #   database enforced for free in Project 7 is now YOUR problem:
    #   nothing stops a row here referencing an org that was deleted.
    #
    #   You trade guaranteed consistency for independent deployability.
    #   That is the microservices bargain, stated honestly.

    user_id: Mapped[int] = mapped_column(index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    trace_id: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())


# ============================================================
# TOKEN VERIFICATION — both strategies, side by side
# ============================================================

_cached_public_key: str | None = None


async def get_public_key() -> str:
    """
    Fetched ONCE, then cached for the process lifetime.

    This one cache is what decouples the services: after this call, the
    agent service can verify tokens forever without the auth service
    existing. Exercise 3 proves it by killing auth.
    """
    global _cached_public_key
    if _cached_public_key is None:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{AUTH_URL}/public-key")
            r.raise_for_status()
            _cached_public_key = r.json()["public_key"]
    return _cached_public_key


class Caller:
    def __init__(self, user_id: int, org_id: int, role: str, mode: str):
        self.user_id, self.org_id, self.role, self.mode = user_id, org_id, role, mode


async def verify_local(token: str) -> Caller:
    """Fast path: verify the signature ourselves. No network call."""
    key = await get_public_key()
    try:
        c = jwt.decode(token, key, algorithms=["RS256"])
    except JWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}")
    return Caller(int(c["sub"]), int(c["org_id"]), c.get("role", "member"), "local")


async def verify_remote(token: str) -> Caller:
    """Ask the auth service on EVERY request. Slower, but revocation is instant."""
    async with httpx.AsyncClient(timeout=5) as cl:
        try:
            r = await cl.post(f"{AUTH_URL}/verify",
                              headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            # The coupling made visible: auth being down means WE are down.
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "Auth service unreachable")
    if r.status_code != 200:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token rejected by auth service")
    d = r.json()
    return Caller(d["user_id"], d["org_id"], d["role"], "remote")


async def current_user(authorization: str = Header(None)) -> Caller:
    """
    ⭐ THIS SERVICE ENFORCES AUTH ITSELF.

    Not "the gateway already checked." If the only auth check lives at
    the gateway, then anyone who reaches :8002 directly — another pod, a
    misconfigured network policy, a developer on the host — bypasses it
    entirely. Exercise 5 has you do exactly that.

    The gateway is a convenience layer, never the security boundary.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    token = authorization[7:]
    return await (verify_remote(token) if VERIFY_MODE == "remote" else verify_local(token))


# ============================================================
# THE SUPERVISOR GRAPH
# ============================================================
# An orchestrator that DELEGATES rather than doing the work.
#
#                    ┌→ researcher ─┐
#   START → supervisor┼→ analyst   ─┼→ synthesize → END
#                    └→ writer     ─┘
#
# WHY THIS SHAPE: each specialist gets a narrow prompt and (in a fuller
# build) a narrow toolset. A generalist agent with 20 tools picks badly;
# three specialists with 3 tools each pick well. The supervisor's only
# job is routing — deciding WHO should work, not doing the work.

SPECIALISTS = {
    "researcher": "You research factual background. Give 2-3 concrete findings.",
    "analyst": "You analyse trade-offs and risks. Give 2-3 specific points.",
    "writer": "You produce clear summaries for a non-expert audience.",
}


class SupervisorState(TypedDict):
    question: str
    assigned: list[str]
    findings: Annotated[list[dict], operator.add]   # reducer: parallel writes
    answer: str


class Assignment(TypedDict):
    specialists: list[Literal["researcher", "analyst", "writer"]]
    reason: str


def llm(temp: float = 0):
    return ChatOpenAI(model=MODEL, temperature=temp)


async def supervisor(state: SupervisorState) -> dict:
    """Decide WHICH specialists this question needs. Routing, not answering."""
    s = llm().with_structured_output(Assignment)
    out = await s.ainvoke([HumanMessage(content=(
        "You supervise a team of specialists:\n"
        + "\n".join(f"- {k}: {v}" for k, v in SPECIALISTS.items())
        + f"\n\nQuestion: {state['question']}\n\n"
        "Assign ONLY the specialists genuinely needed. Assigning all "
        "three when one would do wastes time and money."
    ))])
    return {"assigned": out["specialists"] or ["researcher"]}


def delegate(state: SupervisorState):
    """Dynamic fan-out — the specialist count comes from the supervisor."""
    return [Send("specialist", {"question": state["question"], "role": r})
            for r in state["assigned"]]


async def specialist(payload: dict) -> dict:
    role = payload["role"]
    r = await llm(0.3).ainvoke([HumanMessage(content=(
        f"{SPECIALISTS[role]}\n\nQuestion: {payload['question']}"
    ))])
    return {"findings": [{"role": role, "content": r.content}]}


async def synthesize(state: SupervisorState) -> dict:
    parts = "\n\n".join(f"[{f['role']}] {f['content']}" for f in state["findings"])
    r = await llm(0.3).ainvoke([HumanMessage(content=(
        f"Question: {state['question']}\n\nSpecialist input:\n{parts}\n\n"
        "Write one coherent answer, 3-4 sentences."
    ))])
    return {"answer": r.content}


def build_graph():
    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor)
    g.add_node("specialist", specialist)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", delegate, ["specialist"])
    g.add_edge("specialist", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


graph = build_graph()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="Agent Service", lifespan=lifespan)


class AskReq(BaseModel):
    question: str = Field(..., min_length=5, max_length=300)


@app.get("/health")
async def health():
    return {"service": "agent", "status": "ok", "verify_mode": VERIFY_MODE}


@app.post("/ask")
async def ask(body: AskReq, me: Caller = Depends(current_user),
              x_trace_id: str = Header(None)):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set")

    result = await graph.ainvoke({
        "question": body.question, "assigned": [], "findings": [], "answer": "",
    })

    async with SessionLocal() as db:
        job = Job(org_id=me.org_id, user_id=me.user_id, question=body.question,
                  answer=result["answer"], status="done", trace_id=x_trace_id or "")
        db.add(job)
        await db.commit()
        await db.refresh(job)

    return {
        "job_id": job.id,
        "answer": result["answer"],
        "specialists": result["assigned"],
        "findings": result["findings"],
        "verified_via": me.mode,
        "trace_id": x_trace_id,
    }


@app.get("/jobs")
async def jobs(me: Caller = Depends(current_user)):
    """Org-scoped, exactly as in Project 7 — the boundary still applies."""
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(Job).where(Job.org_id == me.org_id).order_by(Job.id.desc()).limit(20)
        )).scalars().all()
    return [{"id": j.id, "question": j.question, "answer": j.answer[:200],
             "org_id": j.org_id, "trace_id": j.trace_id} for j in rows]