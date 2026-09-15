"""
planner.py — PLAN → EXECUTE → REPLAN.

A different shape of agent from everything you've built so far.

    REACT (projects 1-8)              PLAN-AND-EXECUTE (this one)
    ────────────────────────          ──────────────────────────────
    decide ONE step at a time         decide ALL steps up front
    "what next?" every turn           "here's the whole plan"
    no global view of the task        the plan IS the global view
    an LLM call per step              one planning call + N executions
    can wander indefinitely           bounded by the plan's length
    adapts instantly                  adapts only when it REPLANS

WHY PLANNING WINS FOR MULTI-STEP WORK:
  • You can SEE the plan before anything runs (and gate it — Project 6's
    HITL slots in right here)
  • Steps can declare dependencies, so independent ones run in parallel
  • A failure is localised: you know WHICH step broke and can fix just
    that part
  • Cost is predictable — you know the step count before you start

WHY IT ISN'T ALWAYS BETTER:
  Plans are made with incomplete information. Step 4 might be impossible
  in a way you only discover at step 3. That's why REPLANNING exists —
  and why a planner without it is worse than a ReAct loop.
"""

import asyncio
import os
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
MAX_REPLANS = 2
MAX_STEPS = 6


# ============================================================
# TOOLS — with one that fails on purpose
# ============================================================

async def search_web(query: str) -> str:
    await asyncio.sleep(0.2)
    return f"[web] Results about '{query}': widely discussed, mixed opinions."


async def get_metrics(metric: str) -> str:
    await asyncio.sleep(0.2)
    data = {"revenue": "$4.2M, up 18% YoY", "churn": "3.1% monthly",
            "headcount": "142 employees", "runway": "19 months"}
    key = next((k for k in data if k in metric.lower()), None)
    if not key:
        # A READABLE failure. This is what the replanner reads to decide
        # what to do differently — an opaque error teaches it nothing.
        return (f"ERROR: no metric named '{metric}'. "
                f"Available: {', '.join(data)}")
    return f"[metrics] {key}: {data[key]}"


async def query_db(table: str) -> str:
    await asyncio.sleep(0.2)
    if table.lower() not in ("customers", "orders"):
        return f"ERROR: no table '{table}'. Available: customers, orders"
    return f"[db] {table}: 1,240 rows, last updated today"


async def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return f"ERROR: '{expression}' is not plain arithmetic"
    try:
        return f"[calc] {expression} = {eval(expression)}"
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = {"search_web": search_web, "get_metrics": get_metrics,
         "query_db": query_db, "calculate": calculate}

TOOL_DOCS = """- search_web(query): general web research
- get_metrics(metric): internal metrics. ONLY: revenue, churn, headcount, runway
- query_db(table): database tables. ONLY: customers, orders
- calculate(expression): plain arithmetic only, e.g. "4.2 * 1.18\""""


# ============================================================
# STATE
# ============================================================

class Step(TypedDict):
    id: int
    description: str
    tool: str
    arg: str
    depends_on: list[int]     # step ids that must finish first
    status: str               # pending | done | failed
    result: str


class PlanState(TypedDict):
    goal: str
    plan: list[Step]
    completed: Annotated[list[dict], lambda a, b: a + b]
    replans: int
    answer: str
    log: Annotated[list[str], lambda a, b: a + b]


def llm(temp: float = 0):
    return ChatOpenAI(model=MODEL, temperature=temp)


# ============================================================
# NODE 1 — PLAN
# ============================================================

class PlanStep(TypedDict):
    description: str
    tool: Literal["search_web", "get_metrics", "query_db", "calculate"]
    arg: str
    depends_on: list[int]


class Plan(TypedDict):
    steps: list[PlanStep]


async def plan_node(state: PlanState) -> dict:
    """
    Decompose the goal into concrete steps ONCE, up front.

    `depends_on` is what makes this more than a list: it turns the plan
    into a DAG, so the executor can run independent steps concurrently
    (Project 4's fan-out lesson, now driven by the plan rather than
    hardcoded).
    """
    planner = llm().with_structured_output(Plan)
    out = await planner.ainvoke([HumanMessage(content=(
        f"Break this goal into at most {MAX_STEPS} concrete steps.\n\n"
        f"GOAL: {state['goal']}\n\nAVAILABLE TOOLS:\n{TOOL_DOCS}\n\n"
        "Rules:\n"
        "- one tool call per step\n"
        "- `depends_on` lists the 0-based indices of steps whose RESULT "
        "this step needs. Use [] when a step is independent — independent "
        "steps run in parallel, so do not invent dependencies.\n"
        "- only use tools and arguments the docs above actually support"
    ))])

    steps = [
        {"id": i, "description": s["description"], "tool": s["tool"],
         "arg": s["arg"], "depends_on": s.get("depends_on", []),
         "status": "pending", "result": ""}
        for i, s in enumerate(out["steps"][:MAX_STEPS])
    ]
    return {"plan": steps,
            "log": [f"planned {len(steps)} steps"]}


# ============================================================
# NODE 2 — EXECUTE
# ============================================================

async def execute_node(state: PlanState) -> dict:
    """
    Run the plan, respecting dependencies and parallelising where
    possible.

    THE ALGORITHM: repeatedly find every pending step whose dependencies
    are all done, run that whole batch concurrently, repeat. It's a
    topological execution — the same idea as a build system.

    WHY NOT just run steps in order: steps 1, 2 and 3 might be entirely
    independent. Running them serially wastes 3x the wall time for no
    reason. The plan already told us which ones can overlap.
    """
    plan = [dict(s) for s in state["plan"]]
    completed, log = [], []

    while True:
        ready = [
            s for s in plan
            if s["status"] == "pending"
            and all(plan[d]["status"] == "done"
                    for d in s["depends_on"] if d < len(plan))
        ]
        if not ready:
            break

        log.append(f"batch of {len(ready)}: {[s['id'] for s in ready]}"
                   + (" (parallel)" if len(ready) > 1 else ""))

        async def run_one(step):
            tool = TOOLS.get(step["tool"])
            if not tool:
                return step, f"ERROR: unknown tool '{step['tool']}'"
            try:
                return step, await tool(step["arg"])
            except Exception as e:
                return step, f"ERROR: {e}"

        for step, result in await asyncio.gather(*[run_one(s) for s in ready]):
            step["result"] = result
            # A tool returning "ERROR:" is a FAILED step, not an exception.
            # Failures are data the replanner can reason about.
            step["status"] = "failed" if result.startswith("ERROR") else "done"
            completed.append({"id": step["id"], "description": step["description"],
                              "tool": step["tool"], "arg": step["arg"],
                              "status": step["status"], "result": result})
            log.append(f"  step {step['id']} [{step['status']}] {step['tool']}({step['arg']})")

        if any(s["status"] == "failed" for s in ready):
            break      # stop the batch loop; let the router decide

    return {"plan": plan, "completed": completed, "log": log}


# ============================================================
# NODE 3 — REPLAN
# ============================================================

async def replan_node(state: PlanState) -> dict:
    """
    A step failed. Build a NEW plan for what's left, informed by what we
    learned.

    THIS IS THE NODE THAT MAKES PLANNING VIABLE. A planner without
    replanning is strictly worse than a ReAct loop: it commits to a plan
    made with incomplete information and has no way to adapt when
    reality disagrees.

    The failed step's error message is the key input — that's why the
    tools return readable errors like "no metric named X. Available: ..."
    rather than raising. The replanner can read that and pick a valid
    argument.
    """
    failed = [s for s in state["plan"] if s["status"] == "failed"]
    done = [s for s in state["plan"] if s["status"] == "done"]

    context = "\n".join(f"✓ {s['description']} → {s['result'][:120]}" for s in done)
    failures = "\n".join(f"✗ {s['description']} → {s['result'][:200]}" for s in failed)

    planner = llm().with_structured_output(Plan)
    out = await planner.ainvoke([HumanMessage(content=(
        f"GOAL: {state['goal']}\n\n"
        f"ALREADY DONE:\n{context or '(nothing)'}\n\n"
        f"FAILED:\n{failures}\n\n"
        f"AVAILABLE TOOLS:\n{TOOL_DOCS}\n\n"
        "Write a NEW plan for the REMAINING work only. Read the failure "
        "messages carefully — they usually say exactly what valid inputs "
        "exist. Do not repeat work already done. If a goal is genuinely "
        "unreachable with these tools, return an empty step list."
    ))])

    steps = [
        {"id": i, "description": s["description"], "tool": s["tool"],
         "arg": s["arg"], "depends_on": s.get("depends_on", []),
         "status": "pending", "result": ""}
        for i, s in enumerate(out["steps"][:MAX_STEPS])
    ]
    return {"plan": steps, "replans": state["replans"] + 1,
            "log": [f"↻ REPLAN #{state['replans'] + 1}: {len(steps)} new steps"]}


# ============================================================
# NODE 4 — SYNTHESIZE
# ============================================================

async def synthesize_node(state: PlanState) -> dict:
    results = "\n".join(
        f"[{c['status']}] {c['description']} → {c['result']}"
        for c in state["completed"]
    )
    r = await llm(0.3).ainvoke([HumanMessage(content=(
        f"GOAL: {state['goal']}\n\nWhat the agent gathered:\n{results}\n\n"
        "Write a direct answer to the goal using these results. If some "
        "steps failed and the goal is only partly answered, say so "
        "explicitly rather than papering over it."
    ))])
    return {"answer": r.content, "log": ["synthesized final answer"]}


# ============================================================
# ROUTING
# ============================================================

def after_execute(state: PlanState) -> str:
    """
    Bounded replanning. Without the cap, an impossible goal (asking for a
    metric that doesn't exist, say) would replan forever — paying for a
    planning call each time and never converging.
    """
    if any(s["status"] == "failed" for s in state["plan"]):
        if state["replans"] < MAX_REPLANS:
            return "replan"
        return "synthesize"      # give up honestly; synthesize says so
    return "synthesize"


def build_graph():
    g = StateGraph(PlanState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("replan", replan_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_conditional_edges("execute", after_execute,
                            {"replan": "replan", "synthesize": "synthesize"})
    g.add_edge("replan", "execute")          # ← the replanning loop
    g.add_edge("synthesize", END)
    return g.compile()


graph = build_graph()


async def run(goal: str) -> dict:
    t0 = time.perf_counter()
    trace = []

    async for chunk in graph.astream(
        {"goal": goal, "plan": [], "completed": [], "replans": 0,
         "answer": "", "log": []},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            entry = {"node": node}
            if "plan" in update:
                entry["plan"] = [
                    {"id": s["id"], "description": s["description"],
                     "tool": s["tool"], "arg": s["arg"],
                     "depends_on": s["depends_on"], "status": s["status"]}
                    for s in update["plan"]
                ]
            if update.get("completed"):
                entry["completed"] = update["completed"]
            if update.get("answer"):
                entry["answer"] = update["answer"]
            if update.get("log"):
                entry["log"] = update["log"]
            trace.append(entry)

    answer = next((t["answer"] for t in reversed(trace) if "answer" in t), "(none)")
    replans = sum(1 for t in trace if t["node"] == "replan")

    return {"goal": goal, "answer": answer, "replans": replans,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "trace": trace}


def graph_ascii() -> str:
    try:
        return graph.get_graph().draw_ascii()
    except Exception:
        return "START -> plan -> execute -> (replan?) -> synthesize -> END"