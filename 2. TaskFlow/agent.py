"""
agent.py — a LangGraph agent whose tools read and write your database.

NEW IN THIS PROJECT (vs Project 1):
  • Tools with SIDE EFFECTS — P1's tools were read-only fakes. These
    create and modify real rows.
  • A CONDITIONAL workflow — a router node sends work down different
    paths instead of one straight line.
  • An ITERATIVE workflow — the agent loops until a goal is met, not
    just until it stops calling tools.
  • Tools that share the API's query layer, so both stay consistent.
"""

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import models
from models import Project, SessionLocal, Task

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
MAX_STEPS = 6


# ============================================================
# STATE
# ============================================================
class AgentState(TypedDict):
    """
    New vs P1: `intent` and `steps` alongside messages.

    Not everything belongs in the message list. `intent` is a routing
    decision the graph needs but the LLM shouldn't have to re-derive at
    every node. State is for anything a node downstream needs to know.
    """
    messages: Annotated[list, add_messages]
    intent: str
    steps: int


# ============================================================
# TOOLS — these touch a real database
# ============================================================
# Each opens its OWN short-lived session. They can't use FastAPI's
# Depends() because they aren't running inside a request — the graph
# invokes them. Keeping session lifetime inside the tool also means
# these same tools work unchanged from a background worker later.

@tool
async def list_tasks(status: Literal["all", "done", "pending"] = "all") -> str:
    """List the user's tasks. status can be 'all', 'done', or 'pending'."""
    async with SessionLocal() as db:
        done = None if status == "all" else (status == "done")
        tasks = await models.list_tasks(db, done=done)
        if not tasks:
            return "No tasks found."
        return "\n".join(
            f"#{t.id} [{'x' if t.done else ' '}] {t.title} ({t.project.name})"
            for t in tasks
        )


@tool
async def create_task(title: str, project_name: str = "Work") -> str:
    """Create a new task. project_name must be an existing project."""
    async with SessionLocal() as db:
        project = await models.find_project_by_name(db, project_name)
        if not project:
            # Return a readable error, don't raise. The model sees this
            # as an observation and can retry with a valid project name —
            # self-correction instead of a crash.
            existing = await models.list_projects(db)
            names = ", ".join(p.name for p in existing)
            return f"No project named '{project_name}'. Existing projects: {names}"

        task = Task(title=title.strip(), project_id=project.id)
        db.add(task)
        await db.commit()
        await db.refresh(task)
        return f"Created task #{task.id}: '{task.title}' in {project.name}"


@tool
async def complete_task(task_id: int) -> str:
    """Mark a task as done by its ID."""
    async with SessionLocal() as db:
        task = await models.get_task(db, task_id)
        if not task:
            return f"No task with id {task_id}. Use list_tasks to see valid ids."
        if task.done:
            return f"Task #{task_id} was already done."   # idempotent-ish
        task.done = True
        await db.commit()
        return f"Completed task #{task_id}: '{task.title}'"


@tool
async def create_project(name: str, color: str = "grey") -> str:
    """Create a new project to group tasks under."""
    async with SessionLocal() as db:
        if await models.find_project_by_name(db, name):
            return f"Project '{name}' already exists."
        p = Project(name=name.strip(), color=color)
        db.add(p)
        await db.commit()
        return f"Created project '{p.name}'"


TOOLS = [list_tasks, create_task, complete_task, create_project]

_llm = None
_router_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)
    return _llm


# ============================================================
# NODE 1 — the ROUTER (this is the conditional workflow)
# ============================================================

class Intent(TypedDict):
    intent: Literal["task_action", "chitchat"]


async def classify(state: AgentState) -> dict:
    """
    A cheap first pass that decides which path the request takes.

    WHY bother instead of letting the main agent handle everything:
    "hello" doesn't need database tools, a tool loop, or the token cost
    of sending four tool schemas to the model. Routing cheap requests
    away from the expensive path is real cost control — and it's the
    same instinct as gating expensive work behind a filter anywhere else.

    with_structured_output() forces the reply into the Intent shape —
    the model literally cannot return anything else.
    """
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(Intent)

    last = state["messages"][-1].content
    result = await _router_llm.ainvoke([
        HumanMessage(content=(
            "Classify this message.\n"
            "'task_action' = anything about creating, listing, completing tasks or projects.\n"
            "'chitchat'    = greetings, thanks, general questions.\n\n"
            f"Message: {last}"
        ))
    ])
    return {"intent": result["intent"]}


def route_by_intent(state: AgentState) -> str:
    """The conditional EDGE. Reads state, returns the next node's name."""
    return "chat" if state["intent"] == "chitchat" else "agent"


# ============================================================
# NODE 2a — chitchat (the cheap path: no tools, one call, done)
# ============================================================

async def chat(state: AgentState) -> dict:
    plain = ChatOpenAI(model=MODEL, temperature=0.4)   # note: NO tools bound
    reply = await plain.ainvoke([
        HumanMessage(content=(
            "You are a friendly task-manager assistant. Reply in one short "
            f"sentence.\n\nUser: {state['messages'][-1].content}"
        ))
    ])
    return {"messages": [reply], "steps": state.get("steps", 0) + 1}


# ============================================================
# NODE 2b — the agent (the iterative path)
# ============================================================

async def agent(state: AgentState) -> dict:
    """Same role as P1's `reason` node — decide, or answer."""
    system = HumanMessage(content=(
        "You are a task manager. Use the tools to inspect and modify the "
        "user's tasks. If a tool returns an error, read it and try again "
        "with corrected arguments. Be concise."
    ))
    response = await get_llm().ainvoke([system] + state["messages"])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


act = ToolNode(TOOLS)


def should_continue(state: AgentState) -> str:
    """
    The ITERATIVE workflow's exit condition.

    Two independent ways out — both necessary:
      1. The model stopped requesting tools  → it's satisfied, we're done.
      2. The step budget ran out             → safety net against a model
                                                that would loop forever.
    """
    if state.get("steps", 0) >= MAX_STEPS:
        return END
    last = state["messages"][-1]
    return "act" if getattr(last, "tool_calls", None) else END


# ============================================================
# THE GRAPH
# ============================================================
#
#   START → classify ─┬─ chitchat ──→ chat ────────────────→ END
#                     │
#                     └─ task_action → agent ⇄ act → ... → END
#                                        ↑_______|
#                                      (iterative loop)
#
# P1 was one straight path. This graph BRANCHES (conditional) and one of
# its branches LOOPS (iterative). Both are edges — which is why the graph
# can still draw itself.

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify)
    g.add_node("chat", chat)
    g.add_node("agent", agent)
    g.add_node("act", act)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_by_intent,
                            {"chat": "chat", "agent": "agent"})
    g.add_edge("chat", END)                                    # cheap path ends
    g.add_conditional_edges("agent", should_continue,
                            {"act": "act", END: END})
    g.add_edge("act", "agent")                                 # the loop
    return g.compile()


graph = build_graph()


async def run_agent(message: str) -> dict:
    """Returns {answer, intent, steps, trace} for the UI."""
    trace, answer, steps, intent = [], None, 0, "?"

    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=message)], "intent": "", "steps": 0},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if "intent" in update and update["intent"]:
                intent = update["intent"]
                trace.append({"type": "route", "node": node, "intent": intent})
            steps = update.get("steps", steps)
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            trace.append({"type": "tool_call", "node": node,
                                          "tool": tc["name"], "args": tc["args"]})
                    elif msg.content:
                        answer = msg.content
                        trace.append({"type": "answer", "node": node, "content": msg.content})
                elif isinstance(msg, ToolMessage):
                    trace.append({"type": "observation", "node": node,
                                  "tool": msg.name, "result": str(msg.content)[:300]})

    return {"answer": answer or "(no answer)", "intent": intent,
            "steps": steps, "trace": trace}


def graph_ascii() -> str:
    try:
        return graph.get_graph().draw_ascii()
    except Exception:
        return "START -> classify -> (chat | agent <-> act) -> END"