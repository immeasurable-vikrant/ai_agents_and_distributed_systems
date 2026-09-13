"""
memory.py — LONG-TERM memory, with two interchangeable backends.

THE DISTINCTION THAT MATTERS (this is the whole project):

    SHORT-TERM MEMORY (STM)          LONG-TERM MEMORY (LTM)
    ─────────────────────────        ──────────────────────────
    the checkpointer (Project 3)     the Store (this project)
    scoped to ONE thread_id          scoped to a USER, across threads
    "what did we say in this chat"   "what do I know about this person"
    full message history             distilled FACTS
    grows every turn, gets trimmed   grows slowly, curated
    cleared when the chat ends       survives forever

Start a brand new conversation and STM is empty — but LTM still knows
you're vegetarian. That's the difference, and you can feel it in the UI.

TWO BACKENDS:
  langgraph  — LangGraph's built-in Store. Zero extra deps. You write the
               extraction and search logic yourself, so you can SEE it.
  mem0       — a dedicated memory service. It handles extraction,
               deduplication, conflict resolution and decay for you.

Build #1 to understand the machinery; reach for #2 when you want it
maintained by someone else.
"""

import json
import os
import uuid
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
BACKEND = os.getenv("MEMORY_BACKEND", "langgraph").lower()


# ============================================================
# BACKEND 1 — LangGraph Store
# ============================================================
# A namespaced key-value store with optional semantic search.
#
# NAMESPACE = the isolation key. ("memories", user_id) means one user's
# facts are structurally unreachable from another's — the same idea as
# Project 7's org_id filter, but built into the store's API rather than
# something you must remember to add to every query.
#
# ⚠️ InMemoryStore is for development — it dies with the process. Swap
# for PostgresStore in production; the interface is identical. This is
# the exact same "SQLite → Postgres" swap as the checkpointer in P3.

_store = InMemoryStore()


def _ns(user_id: str) -> tuple:
    return ("memories", user_id)


async def lg_save(user_id: str, fact: str, category: str = "general") -> dict:
    mem_id = str(uuid.uuid4())[:8]
    value = {
        "fact": fact,
        "category": category,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await _store.aput(_ns(user_id), mem_id, value)
    return {"id": mem_id, **value}


async def lg_all(user_id: str) -> list[dict]:
    items = await _store.asearch(_ns(user_id), limit=100)
    return [{"id": i.key, **i.value} for i in items]


async def lg_search(user_id: str, query: str, k: int = 5) -> list[dict]:
    """
    ⚠️ HONEST LIMITATION, verified by testing: with no embedding index
    configured, InMemoryStore IGNORES `query=` entirely and returns every
    item in the namespace. Not substring matching — no filtering at all:

        query='vegetarian'    -> ['vegetarian','lives in Gurgaon','loves Kafka']
        query='zzzz-nonsense' -> ['vegetarian','lives in Gurgaon','loves Kafka']

    So this function is really "fetch all, capped at k". The semantic
    selection happens in `relevant_memories()` via an LLM instead.

    Configure the store with an embedding index and `asearch(query=...)`
    becomes real vector search — at which point you can drop the LLM
    filtering step.
    """
    items = await _store.asearch(_ns(user_id), query=query, limit=k)
    return [{"id": i.key, **i.value} for i in items]


async def lg_delete(user_id: str, mem_id: str):
    await _store.adelete(_ns(user_id), mem_id)


# ============================================================
# BACKEND 2 — mem0
# ============================================================
# A dedicated memory layer. The pitch: it does extraction, dedup,
# conflict resolution ("I love pizza" → later → "I don't eat pizza") and
# decay for you, instead of you hand-rolling each one.
#
# Imported lazily so the project runs with zero extra dependencies when
# you don't want it.

_mem0 = None


def _get_mem0():
    global _mem0
    if _mem0 is None:
        from mem0 import Memory        # pip install mem0ai
        _mem0 = Memory()               # reads OPENAI_API_KEY from env
    return _mem0


async def mem0_save(user_id: str, fact: str, category: str = "general") -> dict:
    m = _get_mem0()
    # mem0 takes MESSAGES, not facts — it runs its own extraction pass
    # to decide what's worth remembering. That's the core difference
    # from the Store: extraction is the service's job, not yours.
    result = m.add(messages=[{"role": "user", "content": fact}], user_id=user_id)
    return {"id": str(result), "fact": fact, "category": category,
            "created_at": datetime.now(timezone.utc).isoformat()}


async def mem0_all(user_id: str) -> list[dict]:
    res = _get_mem0().get_all(user_id=user_id)
    items = res.get("results", res) if isinstance(res, dict) else res
    return [{"id": i.get("id", "?"), "fact": i.get("memory", ""),
             "category": "mem0", "created_at": i.get("created_at", "")}
            for i in items]


async def mem0_search(user_id: str, query: str, k: int = 5) -> list[dict]:
    res = _get_mem0().search(query=query, user_id=user_id, limit=k)
    items = res.get("results", res) if isinstance(res, dict) else res
    return [{"id": i.get("id", "?"), "fact": i.get("memory", ""),
             "category": "mem0", "created_at": i.get("created_at", "")}
            for i in items]


async def mem0_delete(user_id: str, mem_id: str):
    _get_mem0().delete(memory_id=mem_id)


# ============================================================
# ONE INTERFACE, EITHER BACKEND
# ============================================================
# The agent calls these four functions and never knows which backend is
# behind them. That's the point of the abstraction — you can switch with
# an env var and compare behaviour on identical inputs.

async def save_memory(user_id: str, fact: str, category: str = "general") -> dict:
    return await (mem0_save if BACKEND == "mem0" else lg_save)(user_id, fact, category)


async def all_memories(user_id: str) -> list[dict]:
    return await (mem0_all if BACKEND == "mem0" else lg_all)(user_id)


async def search_memories(user_id: str, query: str, k: int = 5) -> list[dict]:
    return await (mem0_search if BACKEND == "mem0" else lg_search)(user_id, query, k)


async def delete_memory(user_id: str, mem_id: str):
    return await (mem0_delete if BACKEND == "mem0" else lg_delete)(user_id, mem_id)


def backend_name() -> str:
    return BACKEND


# ============================================================
# EXTRACTION — deciding what is worth remembering
# ============================================================
# With the LangGraph Store this is YOUR job. (mem0 does it internally,
# which is exactly what you're paying for.)

EXTRACTION_PROMPT = """Extract DURABLE facts about the user worth remembering \
across future conversations.

REMEMBER: preferences, constraints, goals, relationships, recurring \
context, stable personal details.
DO NOT REMEMBER: one-off questions, small talk, anything already obvious \
from context, or transient state ("I'm tired today").

Return [] if nothing is worth keeping. Most messages are worth nothing — \
be strict. A memory store full of noise is worse than an empty one, \
because it crowds out the facts that matter."""


class ExtractedFacts(dict):
    pass


async def extract_facts(message: str, existing: list[str]) -> list[dict]:
    """
    One LLM call: does this message contain anything durable?

    `existing` is passed in so the model can avoid re-saving a fact we
    already have. That's a crude form of DEDUPLICATION — mem0 does this
    properly with embedding similarity. Without any dedup, "I'm
    vegetarian" said three times becomes three memories.
    """
    from typing import Literal, TypedDict

    class Fact(TypedDict):
        fact: str
        category: Literal["preference", "constraint", "goal", "personal", "general"]

    class Result(TypedDict):
        facts: list[Fact]

    llm = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(Result)
    known = "\n".join(f"- {e}" for e in existing) or "(none yet)"

    out = await llm.ainvoke([HumanMessage(content=(
        f"{EXTRACTION_PROMPT}\n\nAlready known about this user:\n{known}\n\n"
        f"New message: {message}\n\n"
        "Extract only NEW durable facts not already covered above."
    ))])
    return out["facts"]


async def relevant_memories(user_id: str, question: str, k: int = 5) -> list[dict]:
    """
    Pick which stored memories matter for THIS question.

    WHY an LLM filter rather than raw search: see lg_search's note — the
    dev Store does substring matching, so "what should I cook?" misses
    "user is vegetarian" entirely. The LLM closes that semantic gap.

    WHY NOT just inject every memory into every prompt: it works at 10
    memories and breaks at 500 — you'd blow the context window and pay
    for irrelevant tokens on every single turn. Filtering is what makes
    LTM scale.
    """
    mems = await all_memories(user_id)
    if not mems:
        return []
    if len(mems) <= 3:
        return mems                    # too few to bother filtering

    from typing import TypedDict

    class Picked(TypedDict):
        ids: list[str]

    llm = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(Picked)
    listing = "\n".join(f"{m['id']}: {m['fact']}" for m in mems)
    out = await llm.ainvoke([HumanMessage(content=(
        f"Stored memories:\n{listing}\n\nQuestion: {question}\n\n"
        "Return the ids of memories genuinely useful for answering this. "
        "Return an empty list if none are."
    ))])
    chosen = set(out["ids"])
    return [m for m in mems if m["id"] in chosen][:k]