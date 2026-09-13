"""
store.py — multi-tenant data + auth + vector search.

THE STAKES ARE DIFFERENT HERE. In earlier projects a bug meant a wrong
answer. Here a bug means Org A reads Org B's confidential documents.
So isolation gets built in TWO layers, and the second one exists purely
to catch the first one failing.
"""

import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt
from sqlalchemy import ForeignKey, String, Text, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = "sqlite+aiosqlite:///./docs.db"
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-prod")
JWT_ALGO = "HS256"
TOKEN_MINUTES = 60

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
# NOTE: using the `bcrypt` library DIRECTLY rather than passlib.
# passlib 1.7.x is incompatible with bcrypt 4.x — it trips an internal
# version-detection path and raises
#   "password cannot be longer than 72 bytes"
# on any call. One fewer dependency, and no broken compatibility shim.


class Base(DeclarativeBase):
    pass


# ============================================================
# MODELS — normalized, with org_id as the isolation key
# ============================================================

class Org(Base):
    __tablename__ = "orgs"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"), index=True)


class Doc(Base):
    __tablename__ = "docs"
    id: Mapped[int] = mapped_column(primary_key=True)

    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"), index=True)
    #   ⭐ THE ISOLATION COLUMN. Every retrieval query filters on it.
    #   index=True because literally every query does — without it,
    #   every search is a full table scan.

    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[str] = mapped_column(Text, default="[]")
    #   JSON-encoded float list. A real deployment uses pgvector with an
    #   ANN index — brute-force cosine over every row is O(n) and fine at
    #   demo scale, fatal at 100k docs. Noted so you know the corner cut.

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


# ============================================================
# AUTH
# ============================================================

def hash_pw(p: str) -> str:
    # bcrypt has a hard 72-byte input limit — truncate explicitly rather
    # than letting the library raise on long passwords.
    return bcrypt.hashpw(p.encode()[:72], bcrypt.gensalt()).decode()


def verify_pw(p: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(p.encode()[:72], h.encode())
    except ValueError:
        return False


def make_token(user_id: int, org_id: int) -> str:
    """
    org_id goes IN the token. That means every request carries its own
    tenant context — no DB lookup just to answer "which org is this?"

    The trade-off: if a user's org ever changed, their existing token
    would carry the stale value until it expires. Fine when org
    membership is effectively permanent; if it isn't, you need short
    TTLs and a refresh flow that re-checks.
    """
    payload = {
        "sub": str(user_id),
        "org_id": org_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def read_token(token: str) -> dict | None:
    try:
        p = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return {"user_id": int(p["sub"]), "org_id": int(p["org_id"])}
    except (JWTError, KeyError, ValueError):
        return None


# ============================================================
# EMBEDDINGS
# ============================================================

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def embed(text: str) -> list[float]:
    """
    Real embeddings via OpenAI when a key is present; otherwise a
    deterministic hash-based fake so the whole project is still runnable
    (and testable) offline. The fake has no semantic meaning — it exists
    to keep the plumbing exercisable, not to give good results.
    """
    if os.getenv("OPENAI_API_KEY"):
        from openai import AsyncOpenAI
        r = await AsyncOpenAI().embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return r.data[0].embedding

    h = hashlib.sha256(text.lower().encode()).digest()
    return [(b - 128) / 128 for b in h]     # 32 dims, deterministic


# ============================================================
# RETRIEVAL — LAYER 1 of isolation
# ============================================================

async def search(org_id: int, query: str, k: int = 4) -> list[dict]:
    """
    ✅ Note the `.where(Doc.org_id == org_id)`. That ONE clause is the
    primary tenant boundary. Everything downstream — ranking, the LLM
    prompt, the answer — only ever sees rows that passed it.
    """
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(Doc).where(Doc.org_id == org_id)
        )).scalars().all()

    if not rows:
        return []

    qv = await embed(query)
    scored = [
        {"id": d.id, "org_id": d.org_id, "title": d.title,
         "content": d.content, "score": round(cosine(qv, json.loads(d.embedding)), 4)}
        for d in rows
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ============================================================
# ❌ THE BUGGY VERSION — for exercise 1. NEVER CALLED.
# ============================================================
# Read it; don't wire it in unless you're deliberately breaking things.
#
# async def search_BUGGY(org_id: int, query: str, k: int = 4):
#     """
#     The org filter is gone. Retrieves across EVERY tenant, ranked purely
#     by similarity — so Org A's question can surface Org B's confidential
#     document if it happens to be the best semantic match.
#
#     This is not a contrived bug. It's what happens when someone
#     "simplifies" a query during a refactor, or copies a snippet from a
#     single-tenant tutorial. Tests may even pass, if the test data only
#     has one org.
#     """
#     async with SessionLocal() as db:
#         rows = (await db.execute(select(Doc))).scalars().all()   # NO FILTER
#     ...


# ============================================================
# LAYER 2 — defense in depth
# ============================================================

class TenantLeak(Exception):
    pass


def assert_same_org(docs: list[dict], org_id: int):
    """
    An INDEPENDENT check that every retrieved doc really belongs to the
    caller. If Layer 1 is correct this never fires.

    WHY BOTHER THEN: because Layer 1 is one line of code, and code gets
    refactored. This is a tripwire for a future bug, placed where it can
    still stop the leak.

    WHY RAISE instead of filtering-and-continuing: if this fires,
    cross-tenant data was about to leave the system. Refusing to answer
    is recoverable; leaking is not. Fail loud, fail closed.
    """
    for d in docs:
        if d["org_id"] != org_id:
            raise TenantLeak(
                f"Doc {d['id']} belongs to org {d['org_id']}, "
                f"caller is org {org_id}. Refusing to answer."
            )


# ============================================================
# SETUP / SEED
# ============================================================

async def init_db():
    async with engine.begin() as c:
        await c.run_sync(Base.metadata.create_all)

    async with SessionLocal() as db:
        if (await db.execute(select(Org))).scalars().first():
            return

        acme = Org(name="Acme")
        globex = Org(name="Globex")
        db.add_all([acme, globex])
        await db.flush()

        db.add_all([
            User(email="alice@acme.com", password_hash=hash_pw("password123"), org_id=acme.id),
            User(email="bob@globex.com", password_hash=hash_pw("password123"), org_id=globex.id),
        ])

        seed = [
            (acme.id, "Acme Refund Policy",
             "Acme offers full refunds within 30 days of purchase. After 30 days, "
             "store credit only. Enterprise contracts have custom refund terms "
             "negotiated per deal."),
            (acme.id, "Acme Q4 Revenue",
             "Acme Q4 revenue was 4.2 million dollars, up 18 percent year over "
             "year. This figure is confidential and internal only."),
            (acme.id, "Acme Onboarding",
             "New Acme employees complete a two week onboarding. Week one is "
             "product training, week two is shadowing a senior engineer."),
            (globex.id, "Globex Refund Policy",
             "Globex does not offer refunds. All sales are final. Customers may "
             "exchange defective items within 14 days."),
            (globex.id, "Globex Hiring Plan",
             "Globex plans to hire 40 engineers in Q1, focused on the platform "
             "team. Confidential until announced."),
        ]
        for org_id, title, content in seed:
            db.add(Doc(org_id=org_id, title=title, content=content,
                       embedding=json.dumps(await embed(f"{title} {content}"))))
        await db.commit()


async def login(email: str, password: str) -> dict | None:
    async with SessionLocal() as db:
        u = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    # Same error for "no such user" and "wrong password" — don't help an
    # attacker enumerate valid emails.
    if not u or not verify_pw(password, u.password_hash):
        return None
    return {"token": make_token(u.id, u.org_id), "org_id": u.org_id, "email": u.email}


async def list_docs(org_id: int) -> list[dict]:
    async with SessionLocal() as db:
        rows = (await db.execute(select(Doc).where(Doc.org_id == org_id))).scalars().all()
    return [{"id": d.id, "title": d.title, "org_id": d.org_id} for d in rows]


async def add_doc(org_id: int, title: str, content: str) -> int:
    """org_id comes from the TOKEN, never from the client. You can't even
    create a document in someone else's org."""
    async with SessionLocal() as db:
        d = Doc(org_id=org_id, title=title, content=content,
                embedding=json.dumps(await embed(f"{title} {content}")))
        db.add(d)
        await db.commit()
        await db.refresh(d)
        return d.id