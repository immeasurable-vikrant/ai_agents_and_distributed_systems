"""
agent.py — an agent with BOTH kinds of memory at once.

    STM  checkpointer, keyed by thread_id   "what we said in THIS chat"
    LTM  store, keyed by user_id            "what I know about YOU"

The graph:

    START → recall ──→ respond ──→ remember → END
            (read LTM)  (uses both)  (write LTM)

`recall` runs BEFORE responding, `remember` runs AFTER. That ordering
matters: you want relevant facts in the prompt, and you want to save new
ones only once you've seen the whole exchange.
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from memory import all_memories, extract_facts, relevant_memories, save_memory

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
KEEP_LAST = 8


class State(TypedDict):
    messages: Annotated[list, add_messages]   # ← STM lives here
    user_id: str
    recalled: list[dict]                       # ← LTM pulled in for this turn
    saved: list[dict]                          # ← LTM written this turn


# ============================================================
# NODE 1 — RECALL (read LTM)
# ============================================================

async def recall(state: State) -> dict:
    """
    Pull the memories relevant to THIS question, before answering.

    WHY filter instead of injecting everything: at 10 memories you could
    dump them all in the prompt. At 500 you'd blow the context window and
    pay for irrelevant tokens every single turn. Selection is what lets
    LTM grow without the cost growing with it.
    """
    last = state["messages"][-1].content
    mems = await relevant_memories(state["user_id"], last, k=5)
    return {"recalled": mems}


# ============================================================
# NODE 2 — RESPOND (uses STM + LTM together)
# ============================================================

async def respond(state: State) -> dict:
    system = ("You are a helpful assistant with memory of this user.")

    if state["recalled"]:
        facts = "\n".join(f"- {m['fact']}" for m in state["recalled"])
        system += (f"\n\nThings you know about this user:\n{facts}\n\n"
                   "Use these naturally. Don't announce that you remembered.")
        #   ^ that last instruction matters: an assistant that says
        #   "As I recall, you're vegetarian!" every turn is exhausting.
        #   Memory should feel like continuity, not a party trick.

    # STM: the recent turns of THIS conversation, from the checkpointer.
    # Trimmed, because every message is re-sent on every call — a long
    # chat otherwise costs more and more per reply until it breaks.
    history = state["messages"][-KEEP_LAST:]

    llm = ChatOpenAI(model=MODEL, temperature=0.4)
    reply = await llm.ainvoke([HumanMessage(content=system)] + history)
    return {"messages": [reply]}


# ============================================================
# NODE 3 — REMEMBER (write LTM)
# ============================================================

async def remember(state: State) -> dict:
    """
    Decide whether anything in this exchange is worth keeping forever.

    Runs AFTER responding so the user isn't waiting on it, and so the
    extractor sees the full exchange.

    Most turns save NOTHING — that's correct. A memory store full of
    "user asked about the weather" is worse than an empty one, because
    noise crowds out the facts that matter at recall time.
    """
    user_msg = next((m.content for m in reversed(state["messages"])
                     if isinstance(m, HumanMessage)), "")
    if not user_msg:
        return {"saved": []}

    existing = [m["fact"] for m in await all_memories(state["user_id"])]
    facts = await extract_facts(user_msg, existing)

    saved = [await save_memory(state["user_id"], f["fact"], f["category"])
             for f in facts]
    return {"saved": saved}


# ============================================================
# THE GRAPH
# ============================================================

_graph = None
_saver_cm = None


async def get_graph():
    global _graph, _saver_cm
    if _graph is None:
        _saver_cm = AsyncSqliteSaver.from_conn_string("./chat.db")
        checkpointer = await _saver_cm.__aenter__()

        g = StateGraph(State)
        g.add_node("recall", recall)
        g.add_node("respond", respond)
        g.add_node("remember", remember)
        g.add_edge(START, "recall")
        g.add_edge("recall", "respond")
        g.add_edge("respond", "remember")
        g.add_edge("remember", END)

        # checkpointer = STM. The Store (LTM) is accessed directly by the
        # nodes rather than passed in here — keeping the two mechanisms
        # visibly separate is the point of this project.
        _graph = g.compile(checkpointer=checkpointer)
    return _graph


async def close_graph():
    """Tied to the app lifespan — the aiosqlite connection is bound to
    the event loop that created it (the Project 6 lesson)."""
    global _graph, _saver_cm
    if _saver_cm is not None:
        await _saver_cm.__aexit__(None, None, None)
        _saver_cm = None
        _graph = None


async def chat(message: str, user_id: str, thread_id: str) -> dict:
    """
    TWO KEYS, TWO MEMORIES:
      thread_id → which conversation (STM)
      user_id   → which person       (LTM)

    New thread_id, same user_id = fresh conversation, but the agent
    still knows who you are. That's the whole demo.
    """
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message)],
         "user_id": user_id, "recalled": [], "saved": []},
        config,
    )

    answer = next((m.content for m in reversed(result["messages"])
                   if isinstance(m, AIMessage) and m.content), "(no answer)")

    return {
        "answer": answer,
        "recalled": result.get("recalled", []),
        "saved": result.get("saved", []),
        "stm_size": len(result["messages"]),
    }


async def get_history(thread_id: str) -> list[dict]:
    graph = await get_graph()
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    out = []
    for m in (snap.values.get("messages", []) if snap.values else []):
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            out.append({"role": "assistant", "content": m.content})
    return out


def graph_ascii() -> str:
    try:
        import asyncio
        g = asyncio.get_event_loop().run_until_complete(get_graph())
        return g.get_graph().draw_ascii()
    except Exception:
        return "START -> recall -> respond -> remember -> END"