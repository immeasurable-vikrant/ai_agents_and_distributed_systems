"""
react_baseline.py — the SAME tools, driven by a ReAct loop.

This exists purely for comparison. Run the same goal through both and
the difference between the two architectures becomes visible rather
than theoretical.

    ReAct                             PLAN-AND-EXECUTE
    ────────────────────────────      ──────────────────────────────
    one LLM call PER STEP             one planning call, then execute
    no plan exists anywhere           the plan is inspectable state
    steps are strictly sequential     independent steps run in PARALLEL
    adapts implicitly every turn      adapts via explicit REPLANNING
    can't show intent in advance      can show intent before acting
    great for short/exploratory       great for long/dependent work

Neither wins outright. ReAct is simpler and adapts without ceremony.
Planning costs a planning call up front and buys you visibility,
parallelism, and a plan a human can approve BEFORE anything runs.
"""

import os
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict

from planner import calculate, get_metrics, query_db, search_web

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
MAX_STEPS = 8


# Wrap the SAME underlying functions as LangChain tools, so the two
# architectures are genuinely comparable — same capabilities, different
# control flow. If the tools differed, the comparison would be meaningless.

@tool
async def web(query: str) -> str:
    """General web research on a topic."""
    return await search_web(query)


@tool
async def metrics(metric: str) -> str:
    """Internal metrics. Only: revenue, churn, headcount, runway."""
    return await get_metrics(metric)


@tool
async def database(table: str) -> str:
    """Query a database table. Only: customers, orders."""
    return await query_db(table)


@tool
async def math(expression: str) -> str:
    """Evaluate plain arithmetic, e.g. '4.2 * 1.18'."""
    return await calculate(expression)


TOOLS = [web, metrics, database, math]


class ReactState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int


async def agent_node(state: ReactState) -> dict:
    llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)
    system = HumanMessage(content=(
        "You are a research assistant. Use the tools to answer the user's "
        "goal. If a tool returns an ERROR, read it and try different "
        "arguments. Be concise."
    ))
    reply = await llm.ainvoke([system] + state["messages"])
    return {"messages": [reply], "steps": state["steps"] + 1}


def should_continue(state: ReactState) -> str:
    # The step budget — ReAct's only guard against wandering forever.
    # Note the planner needs TWO bounds (MAX_STEPS and MAX_REPLANS)
    # because it has two ways to loop.
    if state["steps"] >= MAX_STEPS:
        return END
    last = state["messages"][-1]
    return "act" if getattr(last, "tool_calls", None) else END


def build_graph():
    g = StateGraph(ReactState)
    g.add_node("agent", agent_node)
    g.add_node("act", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"act": "act", END: END})
    g.add_edge("act", "agent")
    return g.compile()


graph = build_graph()


async def run_react(goal: str) -> dict:
    """Same return shape as planner.run(), so the UI can render either."""
    t0 = time.perf_counter()
    trace = []
    llm_calls = 0

    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=goal)], "steps": 0},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            for m in update.get("messages", []):
                if isinstance(m, AIMessage):
                    llm_calls += 1
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            trace.append({"node": "act", "log": [
                                f"call {tc['name']}({list(tc['args'].values())[0] if tc['args'] else ''})"]})
                    elif m.content:
                        trace.append({"node": "answer", "answer": m.content})
                elif isinstance(m, ToolMessage):
                    ok = not str(m.content).startswith("ERROR")
                    trace.append({"node": "observe", "log": [
                        f"  {'done' if ok else 'failed'} → {str(m.content)[:90]}"]})

    answer = next((t["answer"] for t in reversed(trace) if "answer" in t), "(none)")

    return {
        "goal": goal,
        "answer": answer,
        "replans": 0,               # ReAct has no concept of replanning —
                                    # it re-decides implicitly every turn
        "llm_calls": llm_calls,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        "trace": trace,
    }