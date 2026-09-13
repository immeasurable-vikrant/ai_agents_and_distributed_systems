# Project 6 — Live Debate

> Two agents argue a topic, streamed **token by token** over a WebSocket.
> High-risk verdicts **pause the graph** and wait for you — and the pause
> survives killing the server.

---

## Files (4)

```
agent.py      LangGraph: streaming + interrupt() for HITL
main.py       FastAPI + WebSocket
index.html    live typing UI + approval box
README.md
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn langgraph langchain-openai \
            langgraph-checkpoint-sqlite grandalf

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

---

## What you learn

### Backend half

| Concept | Where |
|---|---|
| WebSocket lifecycle (accept, receive, disconnect) | `debate_ws` |
| Why SSE would have been enough here | `main.py` header |
| Surviving client disconnect without losing work | `WebSocketDisconnect` |
| Lifespan-managed resources | `lifespan` |
| **Event-loop binding of async DB connections** | `close_graph` docstring |
| State-machine guards (`409 Conflict`) | `decide` |
| Reading state to recover a reloaded page | `read_debate` |

### LangGraph half

| Concept | Where |
|---|---|
| **`stream_mode="messages"`** — token-level streaming | `stream_debate` |
| Combining `["messages", "updates"]` in one stream | same |
| **`interrupt()`** — pause the graph mid-run | `review` node |
| **`Command(resume=...)`** — continue from the pause | `resume_debate` |
| Why HITL needs a checkpointer to be durable | `review` docstring |
| Risk-gated HITL (don't ask a human every time) | `judge` / `needs_human` |
| Inspecting `snap.next` to detect a paused graph | `get_state` |

---

## The durable HITL proof

**Process 1** — run a high-risk debate to the interrupt, then exit:

```
events   : ['turn_end','turn_end','turn_end','turn_end','verdict','interrupt']
paused_at: ['review']        ← frozen here
turns    : 4 | verdict: 'Spend it.'
-- process 1 exiting, graph still PAUSED --
```

**Process 2** — a brand new interpreter, nothing in memory:

```
FRESH PROCESS sees paused_at: ['review']
             verdict        : 'Spend it.'

after resume -> approved: True
paused_at now: []            ← graph COMPLETED
```

**The naive version looks identical until it doesn't:**

```python
event = asyncio.Event()
PENDING[thread_id] = event
await event.wait()            # ❌
```

That dict and that Event live in **one process's RAM**. A restart
destroys them and nothing is left listening — the run is stranded
forever, with no error to tell you. `interrupt()` writes to a checkpoint
database instead, which is the entire difference.

---

## Two streaming modes, and why you need both

```python
stream_mode=["messages", "updates"]
```

| Mode | Yields | Used for |
|---|---|---|
| `"messages"` | individual **tokens** as generated | live typing in the UI |
| `"updates"` | whole-**node** results | which agent spoke, verdict, interrupt |

Tokens alone can't tell you *who* is speaking or that a pause happened.
Node updates alone leave the user staring at a spinner. The UI uses
`meta["langgraph_node"]` from the token stream to know which bubble to
append to.

**⚠️ Honesty note:** I verified the plumbing (both modes flow through one
stream, node updates arrive correctly, the WebSocket delivers them). I
could **not** verify token-level output without a real API key — my mock
LLM doesn't implement LangChain's streaming callback machinery, so it
produced 0 token events. With a real `ChatOpenAI(streaming=True)` the
tokens do flow. Treat that one piece as "correct by construction, verify
on your machine."

---

## Three bugs I hit building this

**1. `return` inside an `astream` loop corrupted the checkpoint.**
My first version did `yield {...}; return` on interrupt. Closing the
async generator cancelled the underlying stream *mid-write*. The state
came back with **1 turn instead of 4** and an empty verdict. Fix:
`continue` instead of `return` and let the stream end naturally.

**2. The checkpointer connection is bound to its event loop.**
Building the graph lazily meant the aiosqlite connection belonged to
whichever loop called first. Using it from another loop gave
`ValueError: no active connection`. In production uvicorn has one loop so
it "works" — but it's a trap. Fix: initialise the graph in the FastAPI
**lifespan**, on the app's loop.

**3. The WebSocket needs the API key check before the graph runs.**
Without `OPENAI_API_KEY` set, my test got a single `error` event and the
state read blew up with `KeyError: 'paused_at'` — because no debate was
ever created. The handler now fails cleanly over the socket.

---

## Risk-gated HITL

The judge rates each verdict `low` or `high` risk, and only high risk
pauses:

```
low-risk debate  → paused_at: [] | approved: True  | turns: 4
                   (the human was never asked)
approving it     → 409 "This debate is not waiting for approval"
```

**Why gate it at all:** a human rubber-stamping 100% of decisions isn't
oversight, it's friction. HITL is only worth its latency if it fires on
things that genuinely deserve a second look.

---

## Try these

**1. The restart test — do this one.** Start a high-risk debate
(*"Should we lay off 30% of engineering?"*). Wait for the amber approval
box. `Ctrl-C` the server. Restart it. Click **Reload from DB**. The
debate is still there, still paused. Approve it → it finishes.

**2. Watch the low-risk path skip you.** Try *"tabs or spaces for
indentation?"* The judge rates it low risk and the graph goes straight to
END. No approval box appears.

**3. Break the durability.** In `review()`, replace `interrupt()` with an
`asyncio.Event().wait()` and a module-level dict. It works — until you
restart mid-pause, at which point the debate is unrecoverable and
nothing tells you why.

**4. Reintroduce bug #1.** Change `continue` back to `return` in the
`__interrupt__` branch. Run a high-risk debate, then `GET
/api/debates/{id}` — watch `turns` come back wrong.

**5. Disconnect mid-debate.** Close the browser tab while agents are
arguing. Reopen, click **Reload from DB**. The turns that completed are
there — the checkpointer saved them independently of your connection.

**6. Remove the risk gate.** Make `needs_human` always return `"review"`.
Every debate now pauses, including trivial ones. Feel why gating matters.

**7. Approve twice.** Approve a debate, then click Approve again →
**409**. The guard catches a client bug instead of silently doing nothing.

---

## Check yourself

1. What does `interrupt()` actually do to the graph?
2. Why is `interrupt()` durable when `asyncio.Event` isn't?
3. Why combine `"messages"` and `"updates"` stream modes?
4. How does the UI know which agent a token belongs to?
5. Why does the resume endpoint not need to find the original process?
6. What does a non-empty `snap.next` tell you?

<details><summary>Answers</summary>

1. Saves state to the checkpointer, stops execution at that node, and
   surfaces its payload to the caller. The graph is frozen in a database.
2. `asyncio.Event` and the dict holding it live in one process's memory.
   A restart destroys them and the run is stranded with no error.
   `interrupt()` persists, so any process can resume it later.
3. `"messages"` gives tokens (live typing) but no structure;
   `"updates"` gives node results (who spoke, verdict, interrupt) but no
   progressive text. You need both for a good UI.
4. `meta["langgraph_node"]` on each token event — the node name is the
   speaker.
5. Because it loads the checkpoint from the database and injects the
   decision. It doesn't matter which process paused it, or whether that
   process still exists.
6. The graph is mid-run — i.e. **paused**. Empty means it completed.
</details>

---

**Next:** Project 7 — Smart Docs. JWT auth, multi-tenancy, pgvector, and
**RAG → Corrective RAG → Self-RAG** built as three graphs you can compare.