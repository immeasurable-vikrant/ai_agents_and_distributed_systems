# Project 1 — ReAct: By Hand vs LangGraph

> Build the agent loop yourself, then build the same agent in LangGraph,
> then run both side by side and watch them produce identical traces.
>
> **Why this order:** LangGraph's whole value is that it manages the loop
> for you. If you've never written that loop, "the graph handles iteration"
> is a black box. Write it once — ~60 lines — and every LangGraph concept
> in projects 2-11 becomes legible.

---

## Files (5)

```
agent_manual.py   the ReAct loop, hand-written      ← read this first
agent_graph.py    the same agent in LangGraph       ← then this
main.py           FastAPI + DTOs + config + health  ← the backend half
index.html        side-by-side comparison UI
README.md
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn openai langgraph langchain-openai grandalf

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

Open **http://localhost:8000** — one question, two engines, two traces.
API docs are auto-generated at **/docs**.

---

## What you learn

### LangGraph half

| Concept | Where |
|---|---|
| The ReAct loop (reason → act → observe) | `agent_manual.py` |
| **State** + reducers (`add_messages`) | `agent_graph.py` |
| **Nodes** — functions that take State, return partial State | `agent_graph.py` |
| **Conditional edges** — routing logic | `should_continue()` |
| **Tools** — `@tool` decorator, `ToolNode` | both files |
| Sequential workflow + loop-back edge | `build_graph()` |
| Streaming updates (`stream_mode="updates"`) | `run_graph_agent()` |

### Backend half

| Concept | Where |
|---|---|
| FastAPI structure, routing, `async def` | `main.py` |
| **DTOs** — request vs response models | `AskRequest` / `AskResponse` |
| **Validation** — `min_length`, `max_length`, required fields | `AskRequest` |
| HTTP status codes as contract (422, 500, 503) | `_run()` |
| Config from env vars (12-factor) | top of `main.py` |
| Health check endpoint + why LBs need it | `/api/health` |
| Auto-generated OpenAPI docs | `/docs` |

---

## The comparison, line by line

| | Manual | LangGraph |
|---|---|---|
| The loop | `for step in range(MAX_STEPS)` | an edge: `act → reason` |
| Routing | `if not msg.tool_calls: return` | `add_conditional_edges(...)` |
| Message history | `messages.append(...)` by hand | `Annotated[list, add_messages]` |
| Tool schemas | ~30 lines of hand-written JSON | the `@tool` decorator |
| Tool errors | try/except around every call | `ToolNode` handles it |
| Building the trace | append to a list manually | `stream_mode="updates"` |
| **Can it draw itself?** | no | **yes** — click "View Graph Structure" |

That last row is the real difference. In the manual version the loop is
*control flow*. In LangGraph it's *data* — a described structure. That's
why LangGraph can checkpoint it (P3), branch it in parallel (P4), stream it
and pause it mid-run for a human (P6). None of those are possible with a
`for` statement.

---

## Try these

**1. Watch tools chain.** Ask *"Weather in Delhi and Bangalore — which is
hotter?"* Both engines call `get_weather` twice, then reason over both
results. Two loop iterations, visible in the trace.

**2. Watch the no-tool path.** Ask *"Hello, who are you?"* No tool calls;
the conditional edge routes straight to END. One step.

**3. Break the step budget.** Set `MAX_STEPS = 1` in both files, then ask a
two-tool question. Both hit the ceiling. Now ask: what would happen with no
budget at all, on a question the model can't resolve? (Answer: an infinite
loop of real API calls. That's why the budget exists.)

**4. Break a tool description.** In `agent_manual.py`, change
`get_weather`'s description to just `"Gets data."` Ask a weather question.
The model now picks tools badly or asks for clarification — because the
description is the *only* thing it sees. Tool descriptions are prompt
engineering.

**5. Force a bad tool name.** In `agent_manual.py`, rename the key in the
`TOOLS` dict to `"get_weather_v2"` but leave `TOOL_SCHEMAS` alone. The model
requests `get_weather`, your registry doesn't have it → the "no such tool"
error becomes an observation and the model recovers. Then delete that
`if name not in TOOLS` check and watch it crash with a `KeyError` instead.
That's the difference between a handled failure and an outage.

**6. Send invalid input.** POST `{"question": "hi"}` to `/api/ask/manual`.
You get a **422** — Pydantic rejected it before any of your code ran. That's
the DTO doing its job.

---

## Check yourself

1. What are the three parts of a ReAct step?
2. Why does `add_messages` exist? What breaks without it?
3. Where is the "loop" in the LangGraph version — there's no `for`?
4. Why must the step budget exist?
5. Why is `AskRequest` a different class from `AskResponse`?
6. Why is `/api/health` deliberately cheap (no DB, no LLM call)?

<details><summary>Answers</summary>

1. **Reason** (LLM decides), **Act** (run the tool), **Observe** (feed the
   result back so it can reason again).
2. It's a reducer that **appends** to the message list. Without it, each
   node's `{"messages": [...]}` would *replace* the list and the agent
   would lose all memory every step.
3. It's the edge `g.add_edge("act", "reason")` — after running tools,
   control flows back to the LLM node. The loop is a line in the graph.
4. Without it, a model that keeps requesting tools loops forever — real
   API cost, real latency, no natural stopping point.
5. Different directions carry different data. The client sends a question;
   it receives an answer, trace, timing, and engine. In P2, the response
   DTO is also what stops `password_hash` from reaching the client.
6. A load balancer pings it constantly to decide whether to route traffic
   here. If it were expensive, health checking would itself become load —
   and a slow dependency would make healthy instances look dead.
</details>

---

**Next:** Project 2 — TaskFlow. Postgres, async SQLAlchemy, database
normalization, real DTOs over ORM models, and LangGraph tools that read and
write your database.