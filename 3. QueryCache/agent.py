"""
agent.py — a LangGraph agent that REMEMBERS.

THE BIG NEW IDEA: PERSISTENCE.

Projects 1 and 2 were amnesiacs. Every request built a fresh message list,
ran the graph, threw the state away. Ask "what's the weather in Delhi?"
then "what about Mumbai?" and the second question meant nothing.

A CHECKPOINTER changes that. After every node, LangGraph saves the State.
Pass the same thread_id next time and it LOADS that state back before
running. That's short-term memory (STM) — and because it's saved to a
database, it survives a process restart.

    Project 1/2:  request → [fresh state] → graph → discard
    Project 3:    request → [load state by thread_id] → graph → SAVE
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from store import cache_get, cache_key, cache_set

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")

# When the history grows past this, we summarize the old part away.
# WHY a limit at all: every message is re-sent on EVERY call. A 200-turn
# conversation means paying for 200 messages of tokens per reply, until
# you eventually blow the context window entirely.
KEEP_LAST_N = 6


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str        # compressed memory of everything trimmed away


# ============================================================
# TOOLS
# ============================================================

@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # CACHE-ASIDE, inside a tool. Weather for the same city, asked twice
    # in a minute, shouldn't cost two API calls.
    key = cache_key(f"weather:{city}")
    if hit := await cache_get(key):
        return f"{hit['text']} (cached)"

    fake = {"delhi": 34, "mumbai": 31, "bangalore": 26, "london": 12}
    text = f"{city}: {fake.get(city.lower(), 20)}°C"
    await cache_set(key, {"text": text}, ttl=120)
    return text


@tool
async def remember_fact(fact: str) -> str:
    """Store a fact the user wants you to remember for this conversation."""
    # NOTE: this writes into the conversation, so the checkpointer persists
    # it — but only for THIS thread. Cross-conversation memory (LTM) is
    # Project 8's subject. This is still short-term memory.
    return f"Noted: {fact}"


TOOLS = [get_weather, remember_fact]

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)
    return _llm


# ============================================================
# NODES
# ============================================================

async def agent(state: AgentState) -> dict:
    system = "You are a helpful assistant with memory of this conversation."
    if state.get("summary"):
        # The summary is injected as context so the model still "knows"
        # what happened in the trimmed-away portion.
        system += f"\n\nEarlier in this conversation: {state['summary']}"

    response = await get_llm().ainvoke(
        [HumanMessage(content=system)] + state["messages"]
    )
    return {"messages": [response]}


act = ToolNode(TOOLS)


async def summarize(state: AgentState) -> dict:
    """
    SHORT-TERM MEMORY MANAGEMENT.

    When history gets long: ask the model to summarize the old messages,
    store that summary in State, and DELETE the originals.

    RemoveMessage is how you delete from a list guarded by the
    add_messages reducer. You can't just return a shorter list — the
    reducer APPENDS, so that would do nothing. You return an explicit
    "remove this id" instruction instead.

    The trade-off is real: you trade fidelity for cost. The summary is
    lossy. Tune KEEP_LAST_N based on how much your conversations rely on
    exact earlier wording.
    """
    messages = state["messages"]
    if len(messages) <= KEEP_LAST_N:
        return {}

    old = messages[:-KEEP_LAST_N]
    transcript = "\n".join(
        f"{type(m).__name__}: {getattr(m, 'content', '')}" for m in old
    )

    plain = ChatOpenAI(model=MODEL, temperature=0)
    result = await plain.ainvoke([HumanMessage(content=(
        f"Previous summary: {state.get('summary', '(none)')}\n\n"
        f"New messages to fold in:\n{transcript}\n\n"
        "Write an updated 2-3 sentence summary of the whole conversation. "
        "Keep any facts, names, or preferences the user mentioned."
    ))])

    return {
        "summary": result.content,
        "messages": [RemoveMessage(id=m.id) for m in old],   # ← the deletion
    }


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "act"
    # No tools left to run. Trim if we've grown too long, else finish.
    return "summarize" if len(state["messages"]) > KEEP_LAST_N else END


# ============================================================
# THE GRAPH — now with a checkpointer
# ============================================================
#
#   START → agent ⇄ act
#             │
#             ├→ summarize → END   (when history is long)
#             └→ END               (normally)
#
# compile(checkpointer=...) is the entire persistence feature. One
# argument. Everything else — loading prior state, saving after each
# node, keying by thread_id — is handled.

_graph = None
_saver_cm = None


async def get_graph():
    """
    Built once, lazily. AsyncSqliteSaver is an async context manager, so
    we enter it manually and keep it open for the app's lifetime.

    PRODUCTION NOTE: swap AsyncSqliteSaver for AsyncPostgresSaver and
    nothing else changes. SQLite is single-file and single-writer —
    genuinely fine for one process, wrong the moment you run several.
    """
    global _graph, _saver_cm
    if _graph is None:
        _saver_cm = AsyncSqliteSaver.from_conn_string("./checkpoints.db")
        checkpointer = await _saver_cm.__aenter__()

        g = StateGraph(AgentState)
        g.add_node("agent", agent)
        g.add_node("act", act)
        g.add_node("summarize", summarize)

        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", should_continue,
                                {"act": "act", "summarize": "summarize", END: END})
        g.add_edge("act", "agent")
        g.add_edge("summarize", END)

        _graph = g.compile(checkpointer=checkpointer)
    return _graph


# ============================================================
# RUN
# ============================================================

async def chat(message: str, thread_id: str) -> dict:
    """
    thread_id IS the memory key.

    Same thread_id  → the agent picks up where it left off.
    New thread_id   → a completely fresh conversation.

    Nothing about the message list is managed here. We send ONE new
    message; LangGraph loads the rest from the checkpoint.
    """
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    trace, answer, summarized = [], None, False

    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        config,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node == "summarize" and update.get("summary"):
                summarized = True
                trace.append({"type": "summarize", "content": update["summary"]})
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            trace.append({"type": "tool_call", "tool": tc["name"],
                                          "args": tc["args"]})
                    elif msg.content:
                        answer = msg.content
                        trace.append({"type": "answer", "content": msg.content})
                elif isinstance(msg, ToolMessage):
                    trace.append({"type": "observation", "tool": msg.name,
                                  "result": str(msg.content)[:200]})

    state = await graph.aget_state(config)
    return {
        "answer": answer or "(no answer)",
        "trace": trace,
        "summarized": summarized,
        "message_count": len(state.values.get("messages", [])),
        "summary": state.values.get("summary", ""),
    }


async def get_history(thread_id: str) -> list[dict]:
    """
    Read a thread's state WITHOUT running the graph.

    This is what makes "reload the page and your chat is still there"
    work — the history lives in the checkpoint database, not in the
    browser and not in this process's memory.
    """
    graph = await get_graph()
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    out = []
    for m in state.values.get("messages", []):
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            out.append({"role": "assistant", "content": m.content})
    return out


def graph_ascii() -> str:
    try:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(get_graph()).get_graph().draw_ascii()
    except Exception:
        return "START -> agent <-> act -> (summarize) -> END"