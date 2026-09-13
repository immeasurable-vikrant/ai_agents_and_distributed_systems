"""
agent_manual.py — the ReAct loop, written BY HAND.

Read this file first. Then read agent_graph.py, which does the exact same
thing using LangGraph. The comparison is the entire point of Project 1:
you can't appreciate what a framework does for you until you've done it
yourself once.

THE REACT PATTERN (Reason + Act):
    THOUGHT      "I need the weather to answer this"
    ACTION       get_weather(city="Delhi")
    OBSERVATION  {"temp_c": 34, "condition": "haze"}
    THOUGHT      "Now I can answer"
    ANSWER       "It's 34°C and hazy in Delhi."

It's a LOOP: call the LLM → it either asks for a tool or gives an answer
→ if a tool, run it and feed the result back → repeat.
"""

import json
import os

from openai import AsyncOpenAI

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")

# Built lazily, on first use. If we constructed the client at import time,
# the whole app would fail to start without an API key — meaning /health
# and the UI would be unreachable too. Fail at the point of USE, not at
# import, for anything optional.
_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _client


# Hard cap on loop iterations. Without this, a confused model can call
# tools forever — real money, real latency, no natural stopping point.
MAX_STEPS = 5


# ============================================================
# TOOLS — plain Python functions the model may ask us to run
# ============================================================
# Fake data keeps the focus on the LOOP, not on API integration.

def get_weather(city: str) -> dict:
    fake = {"delhi": 34, "mumbai": 31, "bangalore": 26, "london": 12}
    return {"city": city, "temp_c": fake.get(city.lower(), 20)}


def calculate(expression: str) -> dict:
    # Deliberately restricted: eval() on model output is a genuine RCE risk.
    # Project 5 builds a real Docker sandbox for running untrusted code.
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return {"error": "only basic arithmetic is allowed"}
    return {"expression": expression, "result": eval(expression)}


TOOLS = {"get_weather": get_weather, "calculate": calculate}

# The SCHEMA is what the model actually sees. It never sees your Python.
# It reads these descriptions, picks a name, and fills in arguments —
# which means a vague description directly causes wrong tool choices.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression, e.g. '3 * (4 + 1)'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


# ============================================================
# THE LOOP
# ============================================================

async def run_manual_agent(question: str) -> dict:
    """Returns {"answer": str, "trace": list, "steps": int}."""

    # The conversation. The model is stateless — this list IS its memory.
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": question},
    ]
    trace = []

    for step in range(1, MAX_STEPS + 1):
        # --- REASON: ask the model what to do next ---
        response = await get_client().chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS,
        )
        msg = response.choices[0].message

        # --- No tool requested → this is the final answer. Exit the loop. ---
        if not msg.tool_calls:
            trace.append({"step": step, "type": "answer", "content": msg.content})
            return {"answer": msg.content, "trace": trace, "steps": step}

        # The model's tool request must go into history before its results,
        # or the next API call errors out.
        messages.append(msg)

        # --- ACT: run each requested tool ---
        for call in msg.tool_calls:
            name = call.function.name

            # Three things can go wrong here, and none should crash the loop.
            # Each failure becomes an OBSERVATION the model can react to —
            # it can retry with different arguments or explain the problem.
            args = {}
            try:
                args = json.loads(call.function.arguments)   # 1. bad JSON
            except json.JSONDecodeError:
                result = {"error": "could not parse arguments"}
            else:
                if name not in TOOLS:                        # 2. unknown tool
                    result = {"error": f"no such tool: {name}"}
                else:
                    try:
                        result = TOOLS[name](**args)         # 3. tool raised
                    except Exception as e:
                        result = {"error": str(e)[:200]}

            trace.append({"step": step, "type": "tool", "tool": name,
                          "args": args, "result": result})

            # --- OBSERVE: feed the result back so the model can reason on it ---
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": json.dumps(result),
            })
        # loop continues → model sees the observations and decides again

    # --- Budget exhausted. Say so honestly rather than faking an answer. ---
    return {
        "answer": "I couldn't finish within my step budget.",
        "trace": trace, "steps": MAX_STEPS, "budget_exceeded": True,
    }