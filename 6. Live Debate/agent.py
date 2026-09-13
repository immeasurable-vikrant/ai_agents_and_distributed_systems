"""
agent.py — two agents debate, streamed live, and PAUSE for a human.

TWO NEW IDEAS:

1. STREAMING (stream_mode="messages")
   Projects 1-5 used stream_mode="updates" — one event per NODE. That's
   fine for a trace, but the user stares at nothing while a node runs.
   "messages" mode streams individual TOKENS as the model generates them.
   That's the difference between a spinner and watching text appear.

2. INTERRUPT (human-in-the-loop)
   `interrupt()` STOPS the graph mid-run, saves everything to the
   checkpointer, and returns control to you. Later — a second, an hour,
   after a server restart — you resume with Command(resume=value) and it
   continues from exactly that spot.

   THE CRITICAL PART: this only works because of the checkpointer. The
   pause isn't an `asyncio.Event` sitting in this process's memory (which
   would evaporate on restart, stranding the run forever). It's a saved
   checkpoint in a database. That's what makes it DURABLE.
"""

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
ROUNDS = 2


def llm(temp: float = 0.7, streaming: bool = True):
    # streaming=True matters: it makes the underlying client request a
    # token stream, which is what stream_mode="messages" forwards.
    return ChatOpenAI(model=MODEL, temperature=temp, streaming=streaming)


class DebateState(TypedDict):
    topic: str
    turns: Annotated[list[dict], lambda a, b: a + b]   # reducer: accumulate
    round: int
    verdict: str
    approved: bool


ADVOCATE = ("You are the ADVOCATE. Argue FOR the proposal with concrete, "
            "specific reasoning. 2-3 sentences. If the Skeptic has spoken, "
            "rebut their strongest point directly.")

SKEPTIC = ("You are the SKEPTIC. Find the strongest real risk or flaw in "
           "the Advocate's argument. 2-3 sentences. Be specific, not "
           "generically cautious.")


def _history(state: DebateState) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['text']}" for t in state["turns"])


# ============================================================
# THE DEBATE NODES
# ============================================================

async def advocate(state: DebateState) -> dict:
    msg = await llm().ainvoke([HumanMessage(content=(
        f"{ADVOCATE}\n\nTopic: {state['topic']}\n\n"
        f"Debate so far:\n{_history(state) or '(you open)'}"
    ))])
    return {"turns": [{"role": "advocate", "text": msg.content,
                       "round": state["round"] + 1}]}


async def skeptic(state: DebateState) -> dict:
    msg = await llm().ainvoke([HumanMessage(content=(
        f"{SKEPTIC}\n\nTopic: {state['topic']}\n\n"
        f"Debate so far:\n{_history(state)}"
    ))])
    return {"turns": [{"role": "skeptic", "text": msg.content,
                       "round": state["round"] + 1}],
            "round": state["round"] + 1}


def more_rounds(state: DebateState) -> str:
    return "advocate" if state["round"] < ROUNDS else "judge"


# ============================================================
# THE JUDGE — decides whether a human is needed
# ============================================================

class Verdict(TypedDict):
    verdict: str
    risk: Literal["low", "high"]


async def judge(state: DebateState) -> dict:
    """
    Synthesize a verdict AND rate its risk. The risk rating is what
    decides whether the graph pauses for a human.

    WHY not always ask a human: a human rubber-stamping 100% of decisions
    isn't oversight, it's friction. HITL is only valuable if it fires on
    the things that actually deserve a second look.
    """
    j = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(Verdict)
    result = await j.ainvoke([HumanMessage(content=(
        f"Topic: {state['topic']}\n\nDebate:\n{_history(state)}\n\n"
        "Give a 2-3 sentence verdict. Rate risk 'high' if acting on this "
        "would mean significant irreversible spend, legal exposure, or if "
        "the Skeptic raised an objection that was never resolved. "
        "Otherwise 'low'."
    ))])
    return {"verdict": result["verdict"], "approved": result["risk"] == "low"}
    #   approved=True on low risk means we skip the human entirely.


def needs_human(state: DebateState) -> str:
    return END if state["approved"] else "review"


# ============================================================
# ⭐ THE INTERRUPT NODE
# ============================================================

async def review(state: DebateState) -> dict:
    """
    interrupt() throws a special exception that LangGraph catches. It:
      1. saves the current state to the checkpointer
      2. stops execution here
      3. surfaces the payload below to whoever called astream()

    The graph is now FROZEN at this exact point — in the database, not in
    memory. Kill the server. Deploy new code. Come back tomorrow. Then:

        graph.ainvoke(Command(resume="approve"), config)

    ...and `human_decision` below receives "approve" and execution
    continues from right here.

    ⚠️ CONTRAST WITH THE NAIVE APPROACH:
        event = asyncio.Event()
        PENDING[thread_id] = event
        await event.wait()          # ❌
    That looks fine and is fatally fragile. The dict and the Event live
    in THIS process's RAM. A restart destroys them, and nothing is left
    listening — the run is stranded forever, with no error to tell you.
    """
    human_decision = interrupt({
        "reason": "High-risk verdict needs human approval",
        "verdict": state["verdict"],
        "topic": state["topic"],
    })

    # Execution RESUMES here on Command(resume=...)
    if human_decision == "approve":
        return {"approved": True}
    return {"approved": False,
            "verdict": state["verdict"] + "\n\n[REJECTED by human reviewer]"}


# ============================================================
# THE GRAPH
# ============================================================
#
#   START → advocate → skeptic ─(more rounds?)→ advocate
#                          │
#                          └─→ judge ─(low risk)──────────→ END
#                                   └─(high risk)→ review → END
#                                                    ⏸ interrupt

_graph = None
_saver_cm = None


async def get_graph():
    global _graph, _saver_cm
    if _graph is None:
        _saver_cm = AsyncSqliteSaver.from_conn_string("./debates.db")
        checkpointer = await _saver_cm.__aenter__()

        g = StateGraph(DebateState)
        g.add_node("advocate", advocate)
        g.add_node("skeptic", skeptic)
        g.add_node("judge", judge)
        g.add_node("review", review)

        g.add_edge(START, "advocate")
        g.add_edge("advocate", "skeptic")
        g.add_conditional_edges("skeptic", more_rounds,
                                {"advocate": "advocate", "judge": "judge"})
        g.add_conditional_edges("judge", needs_human,
                                {"review": "review", END: END})
        g.add_edge("review", END)

        # interrupt() REQUIRES a checkpointer. Without one there's nowhere
        # to save the paused state, and LangGraph will refuse.
        _graph = g.compile(checkpointer=checkpointer)
    return _graph


# ============================================================
# STREAMING RUN
# ============================================================

async def stream_debate(topic: str, thread_id: str):
    """
    Async generator of events for the WebSocket.

    stream_mode=["messages", "updates"] gives BOTH:
      "messages" → individual tokens, as generated  (for live typing)
      "updates"  → whole-node results               (for structure/state)

    You need both. Tokens alone can't tell you which agent is speaking or
    that an interrupt happened.
    """
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    async for mode, chunk in graph.astream(
        {"topic": topic, "turns": [], "round": 0, "verdict": "", "approved": False},
        config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if getattr(msg, "content", None):
                # node name tells the UI which bubble to append to
                yield {"type": "token", "node": meta.get("langgraph_node", "?"),
                       "text": msg.content}

        elif mode == "updates":
            for node, update in chunk.items():
                if node == "__interrupt__":
                    # ⏸ The graph has PAUSED and saved itself.
                    payload = update[0].value if isinstance(update, tuple) else update
                    yield {"type": "interrupt", "payload": payload}
                    # ⚠️ DO NOT `return` here.
                    # Returning from inside an astream loop closes the
                    # async generator, which cancels the underlying stream
                    # — and the checkpoint write can be cut off mid-flight.
                    # (Found the hard way: state came back with 1 turn
                    # instead of 4 and an empty verdict.) Let the loop end
                    # naturally; astream stops on its own after interrupt.
                    continue
                if node in ("advocate", "skeptic") and update.get("turns"):
                    t = update["turns"][-1]
                    yield {"type": "turn_end", "node": node,
                           "role": t["role"], "round": t["round"]}
                elif node == "judge":
                    yield {"type": "verdict", "verdict": update["verdict"],
                           "approved": update["approved"]}


async def resume_debate(thread_id: str, decision: str):
    """
    Resume a paused graph. Works from ANY process — the state came from
    the database, not from whatever process called interrupt().
    """
    from langgraph.types import Command
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(Command(resume=decision), config)
    return {"approved": result["approved"], "verdict": result["verdict"]}


async def get_state(thread_id: str) -> dict:
    """Read a debate's state without running it — incl. whether it's paused."""
    graph = await get_graph()
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap or not snap.values:
        return {"exists": False}
    return {
        "exists": True,
        "topic": snap.values.get("topic", ""),
        "turns": snap.values.get("turns", []),
        "verdict": snap.values.get("verdict", ""),
        "approved": snap.values.get("approved", False),
        # `next` is non-empty when the graph is mid-run — i.e. PAUSED.
        "paused_at": list(snap.next) if snap.next else [],
    }


async def close_graph():
    """
    Close the checkpointer's DB connection.

    ⚠️ LOOP-BINDING GOTCHA (found by testing): the aiosqlite connection
    is bound to the EVENT LOOP that created it. Build the graph in one
    loop and use it from another and you get
    `ValueError: no active connection`.

    In production this is a non-issue — uvicorn runs one loop for the
    process lifetime. But it means the graph must be initialised from
    the app's lifespan (see main.py), not lazily from whichever loop
    happens to call first.
    """
    global _graph, _saver_cm
    if _saver_cm is not None:
        await _saver_cm.__aexit__(None, None, None)
        _saver_cm = None
        _graph = None


def graph_ascii() -> str:
    try:
        import asyncio
        g = asyncio.get_event_loop().run_until_complete(get_graph())
        return g.get_graph().draw_ascii()
    except Exception:
        return "START -> advocate -> skeptic -> judge -> (review) -> END"