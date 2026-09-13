# Project 2 — TaskFlow

> A task manager with **two front doors**: a REST API, and an agent that
> speaks English. Both hit the same normalized database through the same
> query layer.

---

## Files (4)

```
models.py     ORM models + DTOs + queries    ← normalization lives here
agent.py      LangGraph: router → chat | agent ⇄ tools
main.py       FastAPI: REST + agent endpoint
index.html    both front doors, side by side
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn "sqlalchemy[asyncio]" aiosqlite \
            langgraph langchain-openai grandalf

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

SQLite, so there's nothing to install or start. Open **http://localhost:8000**.

---

## What you learn

### Backend half

| Concept | Where |
|---|---|
| **Normalization to 3NF** — why two tables, not one | `models.py` comment block |
| One-to-many + foreign keys | `Task.project_id` |
| **DTOs over ORM models** — 5 reasons | `TaskCreate` / `TaskOut` |
| Denormalized reads from normalized writes | `TaskOut.build()` |
| **N+1 and eager loading** | `selectinload` in `list_tasks()` |
| `exclude_unset` for real PATCH semantics | `update_task()` |
| Dependency injection + session lifecycle | `get_db()` |
| Async SQLAlchemy (and why the async driver matters) | top of `models.py` |

### LangGraph half

| Concept | Where |
|---|---|
| **Tools with side effects** (they write to your DB) | `agent.py` tools |
| **Conditional workflow** — a router node + branching edges | `classify` / `route_by_intent` |
| **Iterative workflow** — loop until the goal is met | `should_continue` |
| `with_structured_output()` for forced-shape replies | `classify()` |
| State beyond messages (`intent`, `steps`) | `AgentState` |
| Tool errors as recoverable observations | `create_task()` |

---

## The normalization lesson, in one experiment

The DB stores project data **once**:

```
projects            tasks
┌────┬──────┬──────┐  ┌────┬──────────────┬────────────┐
│ 1  │ Work │ blue │  │ 1  │ Fix login    │ project_id=1│
└────┴──────┴──────┘  │ 2  │ Write tests  │ project_id=1│
                      └────┴──────────────┴────────────┘
```

Rename the project — **one UPDATE** — and every task reflects it:

```
before: [(2,'Write tests','Work'), (1,'Fix login bug','Work')]
after : [(2,'Office'),            (1,'Office')]
```

In the unnormalized version (`project_name` as a column on `tasks`), that
rename means editing every row. Miss one and your data contradicts itself.
That's the **update anomaly**, and avoiding it is what 3NF buys you.

---

## The N+1 lesson (this one actually crashes)

`TaskOut.build()` reaches through `t.project` to flatten the join. If that
relationship isn't already loaded:

```
WITHOUT selectinload -> MissingGreenlet
    greenlet_spawn has not been called; can't call await_only() here...
WITH selectinload    -> OK: Work
```

In **sync** SQLAlchemy this would silently fire one extra query per task —
100 tasks, 101 queries, slow but working. In **async** SQLAlchemy it
refuses implicit lazy IO and raises instead.

That's a feature: a silent performance bug becomes a loud error.

**The rule: your eager loading must match your DTO's nesting.**

---

## The graph

```
      +-----------+
      | __start__ |
      +-----------+
      +----------+
      | classify |          ← cheap router: task_action or chitchat?
      +----------+
        ..      ..
+-------+       +------+
| agent |       | chat |    ← two branches
+-------+...    +------+
    *       ..      *
 +-----+       +---------+
 | act |       | __end__ |
 +-----+       +---------+
```

**Conditional:** `classify` picks a branch. "Hey, thanks!" never touches the
database, never loads tool schemas, never enters a loop.

**Iterative:** `agent ⇄ act` loops until the model stops requesting tools or
the step budget runs out.

Why route at all? Sending four tool schemas and running a tool loop to
answer "hello" is wasted latency and wasted tokens. Routing cheap requests
away from the expensive path is real cost control.

---

## Try these

**1. Watch the router split.** Send "Hey, thanks!" then "What tasks do I
have?" Check the `route` step in the trace — different branches, and the
chitchat one has zero tool calls.

**2. Watch error recovery.** Say *"Add a task to my Gardening project."*
There is no Gardening project. The tool returns:

```
No project named 'Gardening'. Existing projects: Work, Personal
```

The model reads that observation and retries with a valid project — or asks
you. **Now change that return to `raise ValueError("bad project")`** and try
again: the graph crashes instead of recovering. Errors-as-observations is
what makes the iterative loop useful.

**3. Watch multi-tool chaining.** *"Create a Fitness project then add a task
to go running in it."* Two tools, two loop iterations, in order.

**4. Break the eager loading.** In `models.py`, delete
`.options(selectinload(Task.project))` from `list_tasks`. Load the page →
`MissingGreenlet`. Put it back.

**5. Break PATCH.** In `update_task`, change `exclude_unset=True` to
`exclude_unset=False`. Now tick a checkbox in the UI: `title` gets
overwritten with `None` because the client didn't send it. That one flag is
the whole difference between PATCH and PUT.

**6. Denormalize on purpose.** Add `project_name` as a column on `Task`,
populate it, then rename a project. Two sources of truth, instantly out of
sync. Now you've *felt* the update anomaly rather than read about it.

**7. Send bad input.** POST `{"title": "", "project_id": 1}` → **422**,
before any of your code runs.

---

## Check yourself

1. Name the three anomalies an unnormalized schema causes.
2. Why is `project_color` on `Project` and not on `Task`?
3. Why does `TaskOut` flatten `project_name` when the DB deliberately
   doesn't?
4. Why does `TaskUpdate` NOT inherit from `TaskCreate`?
5. What does the `classify` node save you, concretely?
6. Why do the agent's tools open their own DB session instead of taking
   one from `Depends`?

<details><summary>Answers</summary>

1. **Update** (rename touches N rows, miss one → contradiction),
   **Insert** (can't create a project with no tasks), **Delete** (removing
   the last task erases the project).
2. It describes the *project*, not the task. On `Task` it would be a
   transitive dependency — a non-key column determined by another non-key
   column (`project_id`). That's the 3NF violation.
3. Storage optimizes for writes and correctness (no duplication). Transfer
   optimizes for the consumer (a UI shouldn't make a second request or do a
   join to render a task row). Normalized writes, denormalized reads.
4. Inherited fields stay **required**. A PATCH sending only `done` would
   fail validation on `title`.
5. Tokens and latency. "Hello" skips four tool schemas, the DB, and the
   iterative loop entirely — one cheap call instead of a full agent run.
6. Because they run inside the graph, not inside an HTTP request — there's
   no `Depends` available. It also means the same tools work unchanged from
   a background worker later (Project 5).
</details>

---

**Next:** Project 3 — QueryCache. Redis caching and rate limiting, **Alembic
migrations** replacing `create_all()`, and LangGraph **persistence**:
checkpointers, `thread_id`, and short-term memory that survives a restart.