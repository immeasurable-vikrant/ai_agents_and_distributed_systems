"""
agent.py — a LangGraph agent with PARALLEL branches.

Projects 1-3 were all SEQUENTIAL: one node, then the next, then the next.
This project's graph does fan-out / fan-in:

                    ┌→ search_web  ─┐
    plan ───────────┼→ search_docs ─┼──→ synthesize → END
                    └→ search_news ─┘
                     (all at once)        (waits for all)

TWO WAYS TO FAN OUT, both shown here:

  STATIC  — you know the branches at build time. Just add multiple edges
            from one node. LangGraph runs them concurrently.

  DYNAMIC — the number of branches depends on runtime data (e.g. the
            planner decided on 5 sub-questions). That needs `Send`.

And the thing that makes fan-in work at all: a REDUCER. Three nodes
writing to the same State key would normally clobber each other. A
reducer says how to COMBINE their writes instead.
"""

import asyncio
import operator
import os
import time
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")

# Cap on how many parallel researchers we'll spawn. Unbounded fan-out on
# LLM calls means rate limits, cost spikes, and tail latency dominated by
# the slowest branch. Same lesson as exp4 in concurrency.py.
MAX_BRANCHES = 4


# ============================================================
# STATE — note the reducer
# ============================================================
class ResearchState(TypedDict):
    question: str
    subquestions: list[str]

    findings: Annotated[list[dict], operator.add]
    #   ⭐ THIS IS WHAT MAKES FAN-IN WORK.
    #
    #   Without `operator.add`, LangGraph REFUSES to run:
    #
    #     InvalidUpdateError: At key 'findings': Can receive only one
    #     value per step. Use an Annotated key to handle multiple values.
    #
    #   Three parallel branches each returning {"findings": [x]} is an
    #   ambiguous write — LangGraph won't guess whether you meant
    #   "replace" or "combine", so it errors instead of silently losing
    #   two results. (A loud failure, like async SQLAlchemy's
    #   MissingGreenlet in Project 2 — much better than a silent one.)
    #
    #   The reducer resolves the ambiguity: CONCATENATE the writes.
    #   Same idea as `add_messages`, just the generic version.
    #
    #   RULE: any State key written by parallel branches needs a reducer.

    answer: str
    timings: Annotated[list[dict], operator.add]


_llm = None


def get_llm(temp: float = 0):
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL, temperature=temp)
    return _llm


# ============================================================
# NODE 1 — PLAN (decides how many branches to spawn)
# ============================================================

class Plan(TypedDict):
    subquestions: list[str]


async def plan(state: ResearchState) -> dict:
    """Break the question into independent sub-questions."""
    planner = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(Plan)
    result = await planner.ainvoke([HumanMessage(content=(
        f"Break this research question into 2-{MAX_BRANCHES} INDEPENDENT "
        f"sub-questions that can be researched separately and in parallel. "
        f"Each must stand alone — no sub-question may depend on another's "
        f"answer.\n\nQuestion: {state['question']}"
    ))])
    return {"subquestions": result["subquestions"][:MAX_BRANCHES]}
    #   WHY "independent" is in the prompt: if sub-question 2 needs
    #   sub-question 1's answer, parallelism is WRONG for it — you'd need
    #   a sequential chain. Parallelism only works on genuinely
    #   independent work. That's a design constraint, not a prompt trick.


# ============================================================
# DYNAMIC FAN-OUT — Send()
# ============================================================

def fan_out(state: ResearchState):
    """
    Returns a LIST of Send objects — one per sub-question. LangGraph runs
    them all CONCURRENTLY.

    WHY Send() instead of just adding edges: the branch COUNT isn't known
    until the planner runs. Static edges are fixed at build time; Send
    creates branches at runtime, from data.

    Each Send carries its OWN payload — the researcher node receives only
    its one sub-question, not the whole state. Clean isolation.
    """
    return [
        Send("research", {"question": state["question"], "subquestion": sq})
        for sq in state["subquestions"]
    ]


# ============================================================
# NODE 2 — RESEARCH (runs N times, in parallel)
# ============================================================

async def research(payload: dict) -> dict:
    """
    One branch. Receives a Send payload, not the full state.

    Returns {"findings": [one_item]} — and the reducer concatenates all
    the branches' single-item lists into one combined list.
    """
    t0 = time.perf_counter()
    sq = payload["subquestion"]

    llm = ChatOpenAI(model=MODEL, temperature=0)
    result = await llm.ainvoke([HumanMessage(content=(
        f"Answer this specific sub-question in 2-3 sentences. Be concrete.\n\n{sq}"
    ))])

    elapsed = round((time.perf_counter() - t0) * 1000)
    return {
        "findings": [{"subquestion": sq, "finding": result.content}],
        "timings": [{"branch": sq[:40], "ms": elapsed}],
    }


# ============================================================
# NODE 3 — SYNTHESIZE (fan-in: runs ONCE, after ALL branches)
# ============================================================

async def synthesize(state: ResearchState) -> dict:
    """
    LangGraph waits for every parallel branch to finish before running
    this. You don't write that barrier — it's implied by the graph shape.

    ⚠️ The tail-latency catch: "waits for ALL" means the slowest branch
    determines total time. 3 branches at 0.5s and 1 at 4s = 4s total.
    That's why production fan-out needs per-branch timeouts — one wedged
    branch holds the whole response hostage.
    """
    findings = "\n\n".join(
        f"Q: {f['subquestion']}\nA: {f['finding']}" for f in state["findings"]
    )
    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    result = await llm.ainvoke([HumanMessage(content=(
        f"Original question: {state['question']}\n\n"
        f"Research findings:\n{findings}\n\n"
        "Write a single coherent answer that synthesizes these findings. "
        "3-4 sentences."
    ))])
    return {"answer": result.content}


# ============================================================
# THE GRAPH
# ============================================================
#
#   START → plan ═╤═→ research ─┐
#                 ├═→ research ─┼→ synthesize → END
#                 └═→ research ─┘
#                (dynamic, via Send)   (implicit barrier)

def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("synthesize", synthesize)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["research"])   # ← dynamic fan-out
    g.add_edge("research", "synthesize")                      # ← fan-in barrier
    g.add_edge("synthesize", END)
    return g.compile()


graph = build_graph()


async def run_research(question: str) -> dict:
    t0 = time.perf_counter()
    trace = []
    final = {}

    async for chunk in graph.astream(
        {"question": question, "subquestions": [], "findings": [],
         "answer": "", "timings": []},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node == "plan":
                trace.append({"type": "plan", "subquestions": update["subquestions"]})
            elif node == "research":
                for f in update.get("findings", []):
                    trace.append({"type": "branch", "subquestion": f["subquestion"],
                                  "finding": f["finding"]})
            elif node == "synthesize":
                final = update
                trace.append({"type": "synthesis", "answer": update["answer"]})

    total_ms = round((time.perf_counter() - t0) * 1000)
    return {"answer": final.get("answer", "(none)"), "trace": trace,
            "total_ms": total_ms}


# ============================================================
# THE COMPARISON — the same work, done sequentially
# ============================================================
# Exists purely so the UI can show both timings side by side. Same LLM
# calls, same prompts — only the scheduling differs.

async def run_research_sequential(question: str) -> dict:
    t0 = time.perf_counter()

    planner = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(Plan)
    p = await planner.ainvoke([HumanMessage(content=(
        f"Break this into 2-{MAX_BRANCHES} independent sub-questions.\n\n{question}"
    ))])
    subs = p["subquestions"][:MAX_BRANCHES]

    findings = []
    for sq in subs:                       # ❌ await in a loop — SEQUENTIAL
        r = await research({"question": question, "subquestion": sq})
        findings.extend(r["findings"])

    text = "\n\n".join(f"Q: {f['subquestion']}\nA: {f['finding']}" for f in findings)
    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    result = await llm.ainvoke([HumanMessage(content=(
        f"Question: {question}\n\nFindings:\n{text}\n\nSynthesize in 3-4 sentences."
    ))])

    return {"answer": result.content, "branches": len(subs),
            "total_ms": round((time.perf_counter() - t0) * 1000)}


def graph_ascii() -> str:
    try:
        return graph.get_graph().draw_ascii()
    except Exception:
        return "START -> plan =(fan-out)=> research -> synthesize -> END"