"""
agent_graph.py — the SAME agent, built with LangGraph.

Read agent_manual.py first, then this. Compare them directly:

    MANUAL                              LANGGRAPH
    ──────────────────────────          ────────────────────────────
    `for step in range(MAX_STEPS)`      the graph's edges create the loop
    if/else on msg.tool_calls           a conditional edge function
    messages.append(...) by hand        State + a reducer do it
    try/except around every tool        ToolNode handles it
    manual trace list                   stream_mode="updates"

Same behaviour, different ownership. You describe the SHAPE of the
computation; LangGraph runs it.
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
MAX_STEPS = 5


# ============================================================
# 1. STATE — what flows between nodes
# ============================================================
class AgentState(TypedDict):
    """
    Every node receives the State and returns a PARTIAL update. LangGraph
    merges that update in for you.

    `Annotated[list, add_messages]` is a REDUCER. Without it, each node's
    `{"messages": [...]}` would REPLACE the list and the agent would lose
    its memory every step. The reducer says "append, don't overwrite" —
    it's doing the job of `messages.append(...)` in agent_manual.py.
    """
    messages: Annotated[list, add_messages]
    steps: int


# ============================================================
# 2. TOOLS — same functions, now with the @tool decorator
# ============================================================
# @tool builds the JSON schema from the signature and docstring. Compare
# this to TOOL_SCHEMAS in agent_manual.py — ~30 lines of hand-written JSON
# replaced by a decorator. The docstring IS the description the model reads,
# so it still matters just as much.

@tool
def get_weather(city: str) -> dict:
    """Get the current temperature for a city."""
    fake = {"delhi": 34, "mumbai": 31, "bangalore": 26, "london": 12}
    return {"city": city, "temp_c": fake.get(city.lower(), 20)}


@tool
def calculate(expression: str) -> dict:
    """Evaluate a basic arithmetic expression, e.g. '3 * (4 + 1)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return {"error": "only basic arithmetic is allowed"}
    return {"expression": expression, "result": eval(expression)}


TOOLS = [get_weather, calculate]

# Lazy, same reasoning as agent_manual.py — the app must start without a key.
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)
    return _llm


# ============================================================
# 3. NODES — plain functions: State in, partial State out
# ============================================================

async def reason(state: AgentState) -> dict:
    """The LLM node. Mirrors the `client.chat.completions.create` call."""
    response = await get_llm().ainvoke(state["messages"])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


# ToolNode reads the last message, runs every tool call in it, and appends
# the results as ToolMessages — including error handling. This one line
# replaces the whole try/except block in agent_manual.py.
act = ToolNode(TOOLS)


# ============================================================
# 4. EDGES — the routing logic
# ============================================================

def should_continue(state: AgentState) -> str:
    """
    This IS the `if not msg.tool_calls: return` check from the manual
    version, lifted out into its own function so the graph can route on it.
    """
    if state.get("steps", 0) >= MAX_STEPS:
        return END                     # the step budget, as an edge
    last = state["messages"][-1]
    return "act" if getattr(last, "tool_calls", None) else END


# ============================================================
# 5. BUILD THE GRAPH
# ============================================================
#
#        START → reason → (tool calls?) ─yes→ act ─┐
#                   ↑                              │
#                   └──────────────────────────────┘
#                            │no
#                            ▼
#                           END
#
# The `act → reason` edge is what creates the loop. In the manual version
# that loop was a `for` statement. Here it's a line in a graph definition —
# which is why LangGraph can checkpoint it, visualise it, and pause it
# mid-run (all covered in projects 3 and 6).

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("reason", reason)
    g.add_node("act", act)
    g.add_edge(START, "reason")
    g.add_conditional_edges("reason", should_continue, {"act": "act", END: END})
    g.add_edge("act", "reason")        # ← the loop
    return g.compile()


graph = build_graph()


# ============================================================
# 6. RUN — stream updates to build the same trace shape
# ============================================================

async def run_graph_agent(question: str) -> dict:
    """Returns the same {"answer", "trace", "steps"} shape as the manual agent."""
    trace = []
    final_answer = None
    steps = 0

    # stream_mode="updates" yields {node_name: partial_state} after each node.
    # This gives us the trace for free — the manual version built it by hand.
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=question)], "steps": 0},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            steps += 1
                            trace.append({"step": steps, "type": "tool_call",
                                          "tool": tc["name"], "args": tc["args"]})
                    elif msg.content:
                        steps += 1
                        final_answer = msg.content
                        trace.append({"step": steps, "type": "answer",
                                      "content": msg.content})
                elif isinstance(msg, ToolMessage):
                    trace.append({"step": steps, "type": "observation",
                                  "tool": msg.name, "result": msg.content[:300]})

    return {"answer": final_answer or "(no answer)", "trace": trace, "steps": steps}


def graph_ascii() -> str:
    """LangGraph can draw itself — something the manual loop can't do."""
    try:
        return graph.get_graph().draw_ascii()
    except Exception:
        return "START -> reason -> (tools? act -> reason) -> END"