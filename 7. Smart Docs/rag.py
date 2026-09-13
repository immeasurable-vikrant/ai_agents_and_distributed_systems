"""
rag.py — THREE RAG graphs, built to be compared.

    NAIVE       retrieve → generate
                Trusts whatever came back. Answers confidently from
                irrelevant documents.

    CORRECTIVE  retrieve → GRADE each doc → (bad? rewrite query, retry)
                → generate
                Fixes RETRIEVAL failures. "Did I get the right documents?"

    SELF-RAG    retrieve → grade → generate → CHECK the answer is
                grounded → CHECK it answers the question → (bad? retry)
                Fixes GENERATION failures too. "Did I make that up?"

The progression is the lesson: each adds a specific reflection step that
catches a specific class of failure the previous one couldn't see.
"""

import os
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from store import assert_same_org, search

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
MAX_RETRIES = 2


def llm(temp: float = 0):
    return ChatOpenAI(model=MODEL, temperature=temp)


class RAGState(TypedDict):
    question: str
    org_id: int
    query: str              # may be REWRITTEN, unlike question
    docs: list[dict]
    graded: list[dict]
    answer: str
    retries: int
    notes: list[str]        # the reflection log, for the UI


# ============================================================
# SHARED NODES
# ============================================================

async def retrieve(state: RAGState) -> dict:
    docs = await search(state["org_id"], state["query"], k=4)

    # ⭐ LAYER 2 runs on EVERY retrieval, in every graph. If the query's
    # org filter ever breaks, this stops the leak before the docs reach
    # a prompt — let alone a user.
    assert_same_org(docs, state["org_id"])

    return {"docs": docs,
            "notes": state["notes"] + [f"retrieved {len(docs)} docs for '{state['query']}'"]}


async def generate(state: RAGState) -> dict:
    docs = state.get("graded") or state["docs"]
    if not docs:
        return {"answer": "I don't have any relevant documents to answer that.",
                "notes": state["notes"] + ["no docs → refused to guess"]}

    context = "\n\n".join(f"[{d['title']}] {d['content']}" for d in docs)
    r = await llm().ainvoke([HumanMessage(content=(
        "Answer using ONLY the context below. If the context doesn't "
        "contain the answer, say so plainly — do not guess.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}"
    ))])
    return {"answer": r.content, "notes": state["notes"] + ["generated answer"]}


# ============================================================
# GRAPH 1 — NAIVE RAG
# ============================================================
# The baseline. Whatever the vector search returns goes straight into
# the prompt. There is no check that any of it is actually relevant.

def build_naive():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


# ============================================================
# GRAPH 2 — CORRECTIVE RAG (CRAG)
# ============================================================

class Grade(TypedDict):
    relevant: bool
    reason: str


async def grade_docs(state: RAGState) -> dict:
    """
    Grade each document for relevance and KEEP ONLY the good ones.

    WHY per-document and not "is this set good": one irrelevant document
    in the context is a distractor that can pull the answer off course.
    Filtering individually means the prompt only contains signal.

    THE COST: one LLM call per document. That's the trade — CRAG is
    meaningfully more expensive than naive RAG. Worth it when a wrong
    answer is expensive; wasteful when it isn't.
    """
    grader = llm().with_structured_output(Grade)
    kept, notes = [], []

    for d in state["docs"]:
        g = await grader.ainvoke([HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Document: [{d['title']}] {d['content'][:600]}\n\n"
            "Does this document contain information useful for answering "
            "the question? Be strict — topically adjacent is NOT relevant."
        ))])
        if g["relevant"]:
            kept.append(d)
        notes.append(f"{'✓' if g['relevant'] else '✗'} {d['title']}: {g['reason'][:60]}")

    return {"graded": kept, "notes": state["notes"] + notes}


async def rewrite_query(state: RAGState) -> dict:
    """
    Nothing relevant came back, so the QUERY was probably the problem.
    Rewrite it and try again.

    WHY this helps: users ask questions in their own words; documents are
    written in the company's words. "Can I get my money back?" and
    "refund policy" are semantically close but not identical, and a
    rewrite can close that gap.
    """
    r = await llm(0.3).ainvoke([HumanMessage(content=(
        f"This search query returned nothing relevant: '{state['query']}'\n"
        f"The user actually asked: '{state['question']}'\n\n"
        "Write a BETTER search query — use the terminology a company "
        "document would use. Return only the query."
    ))])
    new_q = r.content.strip().strip('"')
    return {"query": new_q, "retries": state["retries"] + 1,
            "notes": state["notes"] + [f"↻ rewrote query → '{new_q}'"]}


def crag_decide(state: RAGState) -> str:
    """
    Bounded retry. Without the cap, a question genuinely unanswerable
    from this org's documents would rewrite-and-retry forever, paying for
    a grading pass each time.
    """
    if state["graded"]:
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        return "generate"       # generate() will honestly refuse
    return "rewrite"


def build_corrective():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade_docs)
    g.add_node("rewrite", rewrite_query)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", crag_decide,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")        # ← the correction loop
    g.add_edge("generate", END)
    return g.compile()


# ============================================================
# GRAPH 3 — SELF-RAG
# ============================================================
# CRAG checks the INPUT to generation. Self-RAG also checks the OUTPUT.

class Check(TypedDict):
    ok: bool
    reason: str


async def check_grounded(state: RAGState) -> dict:
    """
    HALLUCINATION CHECK: is every claim in the answer supported by the
    retrieved documents?

    This is the check that catches the scariest failure mode — a fluent,
    confident answer containing facts that appear nowhere in your data.
    Naive RAG cannot detect this at all, because it never looks at its
    own output.
    """
    docs = state.get("graded") or state["docs"]
    if not docs:
        return {"notes": state["notes"] + ["grounding check skipped (no docs)"]}

    context = "\n\n".join(f"[{d['title']}] {d['content']}" for d in docs)
    c = await llm().with_structured_output(Check).ainvoke([HumanMessage(content=(
        f"Documents:\n{context}\n\nAnswer:\n{state['answer']}\n\n"
        "Is EVERY factual claim in the answer directly supported by the "
        "documents? ok=false if anything was invented or inferred beyond "
        "what the documents state."
    ))])
    return {"notes": state["notes"] +
            [f"{'✓' if c['ok'] else '✗'} grounded: {c['reason'][:70]}"],
            "graded": docs if c["ok"] else []}
    #   Clearing `graded` on failure is what routes us back to retry —
    #   see selfrag_decide below.


async def check_useful(state: RAGState) -> dict:
    """
    USEFULNESS CHECK: a grounded answer can still be useless. "The
    documents mention refunds" is perfectly grounded and answers nothing.
    """
    c = await llm().with_structured_output(Check).ainvoke([HumanMessage(content=(
        f"Question: {state['question']}\n\nAnswer: {state['answer']}\n\n"
        "Does this answer actually address the question? ok=false if it "
        "is evasive, generic, or talks around the question."
    ))])
    return {"notes": state["notes"] +
            [f"{'✓' if c['ok'] else '✗'} useful: {c['reason'][:70]}"]}


def selfrag_decide(state: RAGState) -> str:
    """After the checks: accept, or rewrite and go around again."""
    last_two = " ".join(state["notes"][-2:])
    failed = "✗ grounded" in last_two or "✗ useful" in last_two
    if not failed or state["retries"] >= MAX_RETRIES:
        return END
    return "rewrite"


def build_selfrag():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade_docs)
    g.add_node("rewrite", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("check_grounded", check_grounded)
    g.add_node("check_useful", check_useful)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", crag_decide,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check_grounded")     # ← reflect on the OUTPUT
    g.add_edge("check_grounded", "check_useful")
    g.add_conditional_edges("check_useful", selfrag_decide,
                            {"rewrite": "rewrite", END: END})
    return g.compile()


GRAPHS = {
    "naive": build_naive(),
    "corrective": build_corrective(),
    "selfrag": build_selfrag(),
}


async def run_rag(mode: str, question: str, org_id: int) -> dict:
    graph = GRAPHS[mode]
    result = await graph.ainvoke({
        "question": question, "org_id": org_id, "query": question,
        "docs": [], "graded": [], "answer": "", "retries": 0, "notes": [],
    })
    return {
        "mode": mode,
        "answer": result["answer"],
        "notes": result["notes"],
        "retries": result["retries"],
        "docs_used": [d["title"] for d in (result.get("graded") or result["docs"])],
        "llm_calls": _estimate_calls(mode, result),
    }


def _estimate_calls(mode: str, result: dict) -> int:
    """Rough cost signal for the UI — the real point of comparing modes."""
    n_docs = len(result["docs"])
    if mode == "naive":
        return 1
    if mode == "corrective":
        return 1 + n_docs + result["retries"]
    return 3 + n_docs + result["retries"]      # + 2 reflection checks


def graph_ascii(mode: str) -> str:
    try:
        return GRAPHS[mode].get_graph().draw_ascii()
    except Exception:
        return mode