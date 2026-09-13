"""
store.py — persistence + Redis.

Three things live here, and they're all about DURABILITY:

  1. The DB schema, now owned by ALEMBIC MIGRATIONS (not create_all)
  2. A Redis CACHE in front of an expensive call
  3. A Redis RATE LIMITER, with the race condition made visible

Project 2 used create_all() — fine for a fresh dev database, useless the
moment you have real data and need to CHANGE the schema. This project
fixes that properly.
"""

import hashlib
import json
import os
import time
from datetime import datetime

import redis.asyncio as aioredis
from sqlalchemy import String, Text, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./chat.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ============================================================
# THE MODEL — and the migration story behind it
# ============================================================
# This class reflects the schema AFTER all three migrations have run.
# The migrations themselves (migrations.py) tell the story of how it got
# here — which is the part create_all() can never express.
#
#   0001  create conversations(id, thread_id, title, created_at)
#   0002  ADD nullable column `summary`              ← EXPAND
#   0003  backfill it, then make it NOT NULL         ← MIGRATE + CONTRACT
#
# That EXPAND → MIGRATE → CONTRACT sequence is how you add a required
# column to a table that ALREADY HAS ROWS without breaking anything.
# Adding NOT NULL in one step would fail instantly on existing data.

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)

    thread_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    #   ⭐ THE LINK TO LANGGRAPH. The checkpointer stores the agent's
    #   message history keyed by this same thread_id. This table holds
    #   the app's metadata about the conversation; LangGraph holds the
    #   conversation itself. Two stores, one key.

    title: Mapped[str] = mapped_column(String(200), default="New chat")
    summary: Mapped[str] = mapped_column(Text, default="")   # added by 0002/0003
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


# ============================================================
# REDIS
# ============================================================
_redis = None


def get_redis():
    """Lazy, so the app still starts (and /health still answers) without Redis."""
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


# ---------- CACHE (cache-aside) ----------
# The pattern:
#   1. look in cache  → HIT? return it, done
#   2. MISS → do the expensive thing → store it with a TTL → return
#
# WHY a TTL and not "cache forever": the underlying answer can change.
# The TTL bounds how stale you can be. It does not PREVENT staleness —
# exercise 4 makes you feel that difference.

async def cache_get(key: str) -> dict | None:
    try:
        raw = await get_redis().get(f"cache:{key}")
        return json.loads(raw) if raw else None
    except Exception:
        return None      # Redis down → treat as a miss, don't break the request


async def cache_set(key: str, value: dict, ttl: int = 300):
    try:
        await get_redis().set(f"cache:{key}", json.dumps(value), ex=ttl)
    except Exception:
        pass             # caching is an optimization; failing to cache is survivable


def cache_key(text: str) -> str:
    """
    Hash + normalize. The normalize step matters more than it looks:
    "Hello " and "hello" are the same question to a human but would hash
    to different keys without it — a guaranteed cache miss on a hit.
    """
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:32]


# ---------- RATE LIMITER ----------
#
# ❌ THE NAIVE VERSION — a genuine race condition.
#
#     count = await redis.get(key)        # READ
#     if int(count) < limit:              # CHECK
#         await redis.incr(key)           # WRITE
#         return True
#
# Two requests arrive together. Both READ count=9 (limit 10). Both see
# 9 < 10. Both INCR. Eleven requests got through a limit of ten.
# The gap between READ and WRITE is the vulnerability.
#
# ✅ THE FIX — a Lua script. Redis runs it as ONE atomic operation; no
# other command can interleave. The gap simply doesn't exist.
#
# This is the same idea as SELECT FOR UPDATE, as compare-and-swap, as
# SET NX: make read-modify-write inseparable.

RATE_LIMIT_LUA = """
local key   = KEYS[1]
local limit = tonumber(ARGV[1])
local window= tonumber(ARGV[2])

local count = redis.call('INCR', key)
if count == 1 then
    redis.call('EXPIRE', key, window)   -- start the window on first hit
end
if count > limit then
    return 0
end
return 1
"""


async def check_rate_limit(user: str, limit: int = 10, window: int = 60) -> bool:
    """True = allowed. False = too many requests."""
    try:
        allowed = await get_redis().eval(
            RATE_LIMIT_LUA, 1, f"rl:{user}", limit, window
        )
        return allowed == 1
    except Exception:
        # FAIL-OPEN: Redis being down shouldn't take the whole app down.
        # For a chat app, "briefly unprotected from abuse" beats "totally
        # unavailable". A payments endpoint would likely choose the
        # opposite. This is a decision, not a default.
        return True


# ============================================================
# CONVERSATION HELPERS
# ============================================================

async def get_or_create_conversation(thread_id: str) -> Conversation:
    async with SessionLocal() as db:
        q = select(Conversation).where(Conversation.thread_id == thread_id)
        conv = (await db.execute(q)).scalar_one_or_none()
        if conv is None:
            conv = Conversation(thread_id=thread_id)
            db.add(conv)
            await db.commit()
            await db.refresh(conv)
        return conv


async def list_conversations() -> list[Conversation]:
    async with SessionLocal() as db:
        q = select(Conversation).order_by(Conversation.id.desc()).limit(50)
        return list((await db.execute(q)).scalars().all())


async def set_title(thread_id: str, title: str):
    async with SessionLocal() as db:
        q = select(Conversation).where(Conversation.thread_id == thread_id)
        conv = (await db.execute(q)).scalar_one_or_none()
        if conv and conv.title == "New chat":
            conv.title = title[:200]
            await db.commit()