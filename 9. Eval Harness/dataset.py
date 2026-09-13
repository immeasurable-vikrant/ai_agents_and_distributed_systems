"""
dataset.py — the test cases, and the agent being tested.

A DATASET IS JUST A LIST OF (input, expected_output) PAIRS. That's it.
The discipline is in curating it, not in the format.

WHERE REAL DATASETS COME FROM:
  • production failures you fixed  ← the highest-value source by far
  • edge cases you know are hard
  • a handful of happy paths, so a regression there is obvious
  • adversarial inputs (prompt injection, out-of-scope questions)

WHAT MAKES ONE BAD: only happy paths. If every case passes on day one,
the dataset can only ever tell you that you broke something — never that
you fixed something.
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


# ============================================================
# THE DATASET
# ============================================================
# `expected_tools` is what makes trajectory evaluation possible — the
# answer alone can't tell you whether the agent took a sane route.

DATASET = [
    {
        "id": "weather-simple",
        "question": "What's the weather in Delhi?",
        "expected": "34",
        "expected_tools": ["get_weather"],
        "note": "happy path — one tool, one fact",
    },
    {
        "id": "math-simple",
        "question": "What is 17 times 23?",
        "expected": "391",
        "expected_tools": ["calculate"],
        "note": "happy path — should NOT do mental arithmetic",
    },
    {
        "id": "two-tools",
        "question": "What's the weather in Delhi, and what is 12 times 12?",
        "expected": "144",
        "expected_tools": ["get_weather", "calculate"],
        "note": "must use BOTH tools — a classic partial-failure case",
        # ⚠️ WHY `expected` is "144" and not "34 and 144":
        # contains_answer() does a SUBSTRING check. The agent answers
        # "Delhi is 34C and 12 times 12 is 144" — which contains "34"
        # and contains "144", but NOT the literal string "34 and 144".
        # The eval would fail a perfectly correct answer.
        #
        # THE LESSON: heuristic evaluators need expectations shaped to
        # fit them. For genuinely multi-part answers you want either
        # several single-fact checks or an LLM judge. A brittle eval that
        # fails on correct output is worse than no eval — you stop
        # trusting the harness instead of the agent.
        # The `trajectory` eval is what really guards this case anyway:
        # it checks that BOTH tools were called.
    },
    {
        "id": "no-tool-needed",
        "question": "Hello, who are you?",
        "expected": "an assistant",
        "expected_tools": [],
        "note": "must NOT call tools — catches over-eager tool use",
    },
    {
        "id": "unknown-city",
        "question": "What's the weather in Atlantis?",
        "expected": "20",
        "expected_tools": ["get_weather"],
        "note": "tool returns a default — does the agent handle it gracefully?",
    },
    {
        "id": "chained",
        "question": "If it is 34 degrees in Delhi, what is that doubled?",
        "expected": "68",
        "expected_tools": ["calculate"],
        "note": "reasoning + one tool; agents often over-call here",
    },
]


# ============================================================
# THE AGENT UNDER TEST
# ============================================================
# A plain ReAct agent — Project 1's, essentially. The POINT of this
# project isn't the agent, it's being able to MEASURE it.

@tool
def get_weather(city: str) -> str:
    """Get the current temperature for a city in Celsius."""
    fake = {"delhi": 34, "mumbai": 31, "bangalore": 26, "london": 12}
    return f"{city}: {fake.get(city.lower(), 20)}C"


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '3 * (4 + 1)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "error: only basic arithmetic allowed"
    return str(eval(expression))


TOOLS = [get_weather, calculate]


class State(TypedDict):
    messages: Annotated[list, add_messages]
    system: str


# TWO PROMPT VERSIONS — this is what you A/B in the regression demo.
PROMPTS = {
    "v1": ("You are a helpful assistant. Use the tools available when "
           "they help answer the question."),

    "v2": ("You are a helpful assistant. ALWAYS use a tool for any "
           "question involving numbers or weather — never compute in "
           "your head. Answer in one short sentence."),
    #   v2 looks strictly better. Exercise 2 has you check whether it
    #   actually is — "ALWAYS use a tool" can push it to call tools on
    #   "hello", which the `no-tool-needed` case catches.
}


async def run_agent(question: str, prompt_version: str = "v1") -> dict:
    """Run the agent and return BOTH the answer and the tool trajectory."""
    llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)

    async def agent(state: State) -> dict:
        msgs = [HumanMessage(content=state["system"])] + state["messages"]
        return {"messages": [await llm.ainvoke(msgs)]}

    def cont(state: State) -> str:
        last = state["messages"][-1]
        return "act" if getattr(last, "tool_calls", None) else END

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("act", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", cont, {"act": "act", END: END})
    g.add_edge("act", "agent")
    graph = g.compile()

    result = await graph.ainvoke({
        "messages": [HumanMessage(content=question)],
        "system": PROMPTS[prompt_version],
    })

    # Extract the trajectory — which tools, in what order.
    tools_used = [
        tc["name"]
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    ]
    answer = next((m.content for m in reversed(result["messages"])
                   if isinstance(m, AIMessage) and m.content), "")

    return {"answer": answer, "tools_used": tools_used}