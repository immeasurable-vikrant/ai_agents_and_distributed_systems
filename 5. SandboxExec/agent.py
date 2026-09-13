"""
agent.py — a code-writing agent built from a SUBGRAPH.

THE NEW IDEA: SUBGRAPHS.

So far every graph has been flat. This one nests: a self-contained
"write → run → fix" loop is compiled as its OWN graph, then dropped into
the parent graph as a single node.

    PARENT:   START → understand → [ coder ] → explain → END
                                       │
    SUBGRAPH:              START → write → execute → (failed? fix) → END
                                              ↑__________|

WHY BOTHER instead of one flat graph:
  • REUSE — the coder subgraph can be dropped into any parent
  • TESTABLE — you can run and debug the subgraph alone
  • READABLE — the parent reads as 3 steps, not 8
  • SCOPED STATE — the subgraph has its own State shape; the parent
    doesn't need to know about `attempts` or `stderr`

That last one matters most. Without subgraphs, every internal detail of
the retry loop would have to live in one giant shared State.
"""

import os
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from sandbox import backend_name, run_code

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
MAX_FIX_ATTEMPTS = 2


def llm(temp: float = 0):
    return ChatOpenAI(model=MODEL, temperature=temp)


def _strip_fences(text: str) -> str:
    """Models wrap code in ```python fences despite being told not to."""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            block = parts[1]
            return block[len("python"):].strip() if block.startswith("python") else block.strip()
    return text.strip()


# ============================================================
# THE SUBGRAPH — write → execute → fix
# ============================================================

class CoderState(TypedDict):
    """
    The subgraph's OWN state. `attempts`, `stderr` and `code` are internal
    plumbing — the parent graph never sees them, and shouldn't have to.
    """
    task: str
    code: str
    stdout: str
    stderr: str
    success: bool
    attempts: int


async def write_code(state: CoderState) -> dict:
    """Generate Python for the task, or FIX it if a previous run failed."""
    if state.get("stderr"):
        # THE SELF-CORRECTION PROMPT. The error is fed back as context —
        # the same "errors become observations" idea as tool failures in
        # Projects 1-2, applied to code.
        prompt = (
            f"This Python failed.\n\nCODE:\n{state['code']}\n\n"
            f"ERROR:\n{state['stderr'][:800]}\n\n"
            "Return the CORRECTED code only. No markdown, no explanation. "
            "It must print its result."
        )
    else:
        prompt = (
            f"Write short Python to solve this task. Use ONLY the standard "
            f"library — the sandbox has no third-party packages and NO "
            f"network access. It must print the result.\n\n"
            f"TASK: {state['task']}\n\nReturn code only, no markdown."
        )

    result = await llm().ainvoke([HumanMessage(content=prompt)])
    return {"code": _strip_fences(result.content),
            "attempts": state.get("attempts", 0) + 1}


async def execute(state: CoderState) -> dict:
    """Run it in the sandbox. Never raises — failures come back as data."""
    r = await run_code(state["code"])
    return {"stdout": r["stdout"], "stderr": r["stderr"], "success": r["success"]}


def should_retry(state: CoderState) -> str:
    """
    Loop back to write_code on failure — but bounded.

    Without the attempt cap, a task the model genuinely cannot solve
    (or one the sandbox structurally forbids, like a network call) would
    loop forever, paying for an LLM call every time.
    """
    if state["success"]:
        return END
    if state.get("attempts", 0) >= MAX_FIX_ATTEMPTS:
        return END          # give up honestly rather than loop
    return "write_code"


def build_coder_subgraph():
    g = StateGraph(CoderState)
    g.add_node("write_code", write_code)
    g.add_node("execute", execute)
    g.add_edge(START, "write_code")
    g.add_edge("write_code", "execute")
    g.add_conditional_edges("execute", should_retry,
                            {"write_code": "write_code", END: END})
    return g.compile()


coder_subgraph = build_coder_subgraph()


# ============================================================
# THE PARENT GRAPH
# ============================================================

class MainState(TypedDict):
    request: str
    task: str
    code: str
    stdout: str
    stderr: str
    success: bool
    attempts: int
    explanation: str


async def understand(state: MainState) -> dict:
    """Turn a vague request into a precise, codeable task."""
    result = await llm().ainvoke([HumanMessage(content=(
        "Restate this as ONE precise, self-contained programming task "
        "solvable with Python's standard library in under 20 lines. "
        f"One sentence.\n\nRequest: {state['request']}"
    ))])
    return {"task": result.content.strip()}


async def run_coder(state: MainState) -> dict:
    """
    ⭐ THE SUBGRAPH AS A NODE.

    We invoke the compiled subgraph like any other callable, map our
    state into its shape, and map its result back out. That explicit
    mapping IS the boundary — it's what keeps `attempts` and `stderr`
    from leaking into the parent's concerns unless we choose to surface
    them.

    (LangGraph can also nest a subgraph directly via add_node(subgraph)
    when the state shapes align. Doing it manually here makes the
    boundary visible, which is the point of the lesson.)
    """
    result = await coder_subgraph.ainvoke({
        "task": state["task"], "code": "", "stdout": "",
        "stderr": "", "success": False, "attempts": 0,
    })
    return {
        "code": result["code"], "stdout": result["stdout"],
        "stderr": result["stderr"], "success": result["success"],
        "attempts": result["attempts"],
    }


async def explain(state: MainState) -> dict:
    """Turn the raw output into a human answer."""
    if state["success"]:
        prompt = (f"Task: {state['task']}\nCode output: {state['stdout']}\n\n"
                  "Explain the result in 1-2 plain sentences.")
    else:
        prompt = (f"Task: {state['task']}\nIt failed after "
                  f"{state['attempts']} attempts.\nError: {state['stderr'][:400]}\n\n"
                  "Explain in 1-2 sentences what went wrong. Be honest that it failed.")
    result = await llm(0.3).ainvoke([HumanMessage(content=prompt)])
    return {"explanation": result.content}


def build_graph():
    g = StateGraph(MainState)
    g.add_node("understand", understand)
    g.add_node("coder", run_coder)          # ← the subgraph, as one node
    g.add_node("explain", explain)
    g.add_edge(START, "understand")
    g.add_edge("understand", "coder")
    g.add_edge("coder", "explain")
    g.add_edge("explain", END)
    return g.compile()


graph = build_graph()


async def solve(request: str) -> dict:
    trace, out = [], {}
    async for chunk in graph.astream(
        {"request": request, "task": "", "code": "", "stdout": "",
         "stderr": "", "success": False, "attempts": 0, "explanation": ""},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            out.update(update)
            if node == "understand":
                trace.append({"type": "understand", "task": update["task"]})
            elif node == "coder":
                trace.append({"type": "coder", "code": update["code"],
                              "success": update["success"],
                              "attempts": update["attempts"],
                              "stdout": update["stdout"],
                              "stderr": update["stderr"][:400]})
            elif node == "explain":
                trace.append({"type": "explain", "content": update["explanation"]})

    return {
        "explanation": out.get("explanation", ""),
        "code": out.get("code", ""),
        "stdout": out.get("stdout", ""),
        "stderr": out.get("stderr", ""),
        "success": out.get("success", False),
        "attempts": out.get("attempts", 0),
        "backend": backend_name(),
        "trace": trace,
    }


def graph_ascii() -> tuple[str, str]:
    try:
        return graph.get_graph().draw_ascii(), coder_subgraph.get_graph().draw_ascii()
    except Exception:
        return ("START -> understand -> coder -> explain -> END",
                "START -> write_code -> execute -> (retry?) -> END")