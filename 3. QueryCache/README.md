# Project 3 — QueryCache

> A chat agent that **remembers** — and whose memory survives killing the
> server. Plus Redis caching, rate limiting, and real database migrations
> replacing `create_all()`.

---

## Files (5)

```
store.py        DB models + Redis cache + rate limiter
migrations.py   versioned schema changes (runnable)
agent.py        LangGraph with a CHECKPOINTER ← the big new idea
main.py         FastAPI
index.html      multi-thread chat UI
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn "sqlalchemy[asyncio]" aiosqlite \
            redis alembic langgraph langchain-openai langgraph-checkpoint-sqlite

docker run -d -p 6379:6379 redis:7-alpine     # optional — see below
export OPENAI_API_KEY=sk-...

python migrations.py upgrade      # look at the output
uvicorn main:app --reload
```

**Redis is optional.** Without it the app still runs — caching misses,
rate limiting fails open. That's deliberate (see below).

---

## What you learn

### Backend half

| Concept | Where |
|---|---|
| **Migrations** vs `create_all()` | `migrations.py` |
| **Expand → Migrate → Contract** | migrations 0002, 0003 |
| Reversible schema changes (`downgrade`) | `downgrade()` |
| **Cache-aside** + TTL | `cache_get` / `cache_set` |
| **Rate limiting** + the race condition | `RATE_LIMIT_LUA` |
| Lua for atomic read-modify-write | same |
| **Fail-open vs fail-closed** as a decision | `check_rate_limit` |
| Graceful degradation in health checks | `/api/health` |
| `429` + `Retry-After` | `chat()` in `main.py` |

### LangGraph half

| Concept | Where |
|---|---|
| **Checkpointers** — persistence in one argument | `compile(checkpointer=...)` |
| **`thread_id`** — the memory key | `chat()` |
| **Short-term memory (STM)** | the whole agent |
| Reading state without running (`aget_state`) | `get_history()` |
| **Summarization** to bound context growth | `summarize` node |
| `RemoveMessage` — deleting from a reducer-guarded list | same |
| Caching inside a tool | `get_weather` |

---

## The persistence lesson (verified, not asserted)

**Process 1** — three turns across two threads:
```
turn 1 (thread-A): model saw 2 msgs | state: 2
turn 2 (thread-A): model saw 4 msgs | state: 4    ← it loaded turn 1 back
turn 1 (thread-B): model saw 2 msgs | state: 2    ← isolated, fresh
```

**Process 2** — a brand new interpreter, nothing in memory:
```
thread-A history:
       user: My name is Singh
  assistant: [saw 2 msgs]
       user: What is my name?
  assistant: [saw 4 msgs]
thread-B history:
       user: What is my name?
  assistant: [saw 2 msgs]
```

The conversation was never in the browser and never in the first
process's memory. It's in `checkpoints.db`.

**What changed vs Projects 1-2:** one argument.

```python
graph = g.compile(checkpointer=checkpointer)
```

Everything else — loading prior state, saving after each node, keying by
`thread_id` — is handled for you. *This is the payoff for the graph being
a described structure instead of a `for` loop.*

---

## The memory-growth lesson

Every message is re-sent on **every** call. A 200-turn conversation costs
200 messages of tokens per reply, until the context window breaks.

So past `KEEP_LAST_N`, the graph summarizes and deletes:

```
turn 1: msgs= 2  summarized=False
turn 2: msgs= 4  summarized=False
turn 3: msgs= 6  summarized=False
turn 4: msgs= 6  summarized=True    ← would have been 8; trimmed back
turn 5: msgs= 6  summarized=True    ← stays bounded
```

`RemoveMessage` is how you delete from a list guarded by `add_messages`.
Returning a shorter list does nothing — the reducer *appends*. You return
an explicit "remove this id" instruction instead.

**The trade-off is real:** you trade fidelity for cost. Summaries are lossy.

---

## The migration lesson

```
$ python migrations.py status
  [ ] 0001_create_conversations
  [ ] 0002_add_summary_nullable
  [ ] 0003_backfill_and_require_summary

$ python migrations.py upgrade
  ▲ 0001_create_conversations APPLIED
  ▲ 0002_add_summary_nullable APPLIED
  ▲ 0003_backfill_and_require_summary APPLIED

$ python migrations.py downgrade
  ▼ 0003_backfill_and_require_summary REVERTED

$ python migrations.py upgrade
  ✓ 0001_create_conversations (already applied)
  ✓ 0002_add_summary_nullable (already applied)
  ▲ 0003_backfill_and_require_summary APPLIED
```

**Why three migrations for one column:** the goal is a *required*
`summary` column on a table that already has rows. Adding `NOT NULL` in
one step fails immediately — existing rows have no value.

```
0002 EXPAND    add it NULLABLE          always safe, instant
0003 MIGRATE   backfill existing rows   a data operation
     CONTRACT  enforce NOT NULL         safe now, no NULLs remain
```

In production these ship as **separate releases**, so old and new code can
both run against the intermediate schema. That's what zero-downtime
deploys are made of.

---

## The Redis lesson

**The rate limiter's race condition:**

```python
count = await redis.get(key)     # READ
if int(count) < limit:           # CHECK
    await redis.incr(key)        # WRITE   ← gap between READ and WRITE
```

Two requests both read `count=9` against a limit of 10. Both proceed.
Eleven get through.

The Lua script closes it — Redis runs the whole script atomically, so no
other command can interleave. Same idea as `SELECT FOR UPDATE`,
compare-and-swap, and `SET NX`: **make read-modify-write inseparable.**

**Fail-open, verified with Redis actually down:**
```
health              -> {'status': 'ok', 'redis': 'down (degraded)'}
rate limit          -> True   (fail-open: app stays usable)
cache get           -> None   (treated as a miss, no crash)
```

That's a *decision*, not a default. For a chat app, "briefly unprotected
from abuse" beats "completely unavailable." A payments endpoint would
choose the opposite. Notice the health check reports `degraded` rather
than lying about being fully healthy.

---

## Try these

**1. Memory.** Say *"My name is Singh."* Then *"What is my name?"* It
remembers.

**2. Isolation.** Click **+ New chat**, ask your name again. No idea.
Different `thread_id` = different memory.

**3. Durability.** `Ctrl-C` the server. Restart it. Reload the page, open
the old thread. Still there.

**4. Watch the summarizer.** Send 5-6 messages. Watch `msgs in memory`
climb, then stop growing, and the `✂️ summarized` badge appear. Read the
summary — notice what got lost.

**5. Break the cache TTL.** Ask for Delhi's weather twice — second is
`(cached)`. Now change `fake` in `get_weather` to a different temperature
and ask again within 120s. **You still get the old value.** TTL *bounds*
staleness; it doesn't prevent it.

**6. Trip the rate limiter.** Change the limit to 3 in `main.py`, send 4
messages fast → `429` with `Retry-After`.

**7. Kill Redis.** `docker stop <redis>`. The app keeps working —
health says `degraded`, caching misses, rate limiting allows. Now flip
`check_rate_limit`'s except to `return False` and try again: every
request is rejected because a *cache* is down. Feel the difference.

**8. Run a migration backwards.** `python migrations.py downgrade`, then
`status`, then `upgrade`. Note it skips what's already applied.

---

## Check yourself

1. What does `compile(checkpointer=...)` actually change?
2. Why can't you shorten the message list by just returning a shorter list?
3. Why does adding a required column take three migrations?
4. Where is the race in the naive rate limiter, and why does Lua fix it?
5. Why fail-open here but fail-closed for payments?
6. Why does the health check say `degraded` instead of `ok` or `down`?

<details><summary>Answers</summary>

1. LangGraph saves State after every node, keyed by `thread_id`, and
   loads it back before the next run. Persistence + memory in one argument.
2. `add_messages` is a reducer that **appends**. A shorter list would just
   be appended too. You need `RemoveMessage(id=...)` to delete explicitly.
3. Adding `NOT NULL` to a table with existing rows fails — they have no
   value. So: add nullable (safe) → backfill (data op) → enforce NOT NULL
   (safe now). Three steps so each can ship independently.
4. Between READ and WRITE, another request can read the same value. Lua
   runs the whole script as one atomic operation — the gap doesn't exist.
5. Here, Redis is a *cache and a throttle*. Losing it briefly means
   unprotected, which beats unavailable. For payments, being briefly
   unprotected from fraud is worse than being briefly unavailable.
6. Because both are true: the app works, but a dependency is down. A
   health check that says `ok` is lying; one that says `down` would get
   the instance pulled from the load balancer unnecessarily.
</details>

---

**Next:** Project 4 — Research Fanout. **Parallel workflows** (fan-out /
fan-in), and the concurrency lesson underneath: the GIL, threads vs
processes, and why `await` in a `for` loop is secretly sequential.