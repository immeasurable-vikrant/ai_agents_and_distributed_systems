# Project 4 — Research Fanout

> A research agent that splits your question into sub-questions and
> researches them **all at once** — with a concurrency lab underneath
> that explains why that works, and when it doesn't.

---

## Files (5)

```
concurrency.py   5 runnable experiments — the backend half
agent.py         LangGraph parallel fan-out / fan-in
main.py          FastAPI
index.html       research comparison + lab buttons
README.md
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn langgraph langchain-openai grandalf

python concurrency.py       # the lab, standalone — no API key needed

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

---

## What you learn

### Backend half

| Concept | Where |
|---|---|
| `await` in a loop is **sequential** | exp 1 |
| **The GIL** — threads vs processes on CPU work | exp 2 |
| Threads **do** help I/O (GIL is released) | exp 3 |
| **Bounded concurrency** / backpressure | exp 4 |
| **Blocking the event loop** | exp 5 |
| CPU-bound vs I/O-bound as the deciding question | throughout |
| Why pickling constrains `ProcessPoolExecutor` | `cpu_work` comment |

### LangGraph half

| Concept | Where |
|---|---|
| **Parallel branches** (fan-out) | `fan_out()` |
| **`Send()`** — dynamic, data-driven branching | same |
| **Fan-in** — the implicit barrier | `synthesize` |
| **Reducers** for parallel writes | `ResearchState.findings` |
| Tail latency in fan-out | `synthesize` docstring |
| Bounding branch count | `MAX_BRANCHES` |

---

## Real numbers from the lab

```
exp1  await-in-loop 2.40s  vs  gather 0.30s      → 8.0x
exp3  sequential    2.40s  vs  threads 0.30s     → threads DO help I/O
exp4  unbounded     0.30s  vs  bounded(5) 1.20s  → slower, but safe
exp5  blocking      1.52s  vs  non-blocking 0.32s → 5x
```

**exp2 (the GIL) needs multiple cores.** On the 1-CPU machine this was
written on it honestly reports:

```
cpu_count: 1, sequential 0.70s, threads 0.66s, processes 0.69s
verdict: "1 CPU available — processes can't help here either"
```

Run it on your multi-core laptop and you'll see threads ≈ sequential
(the GIL) while processes get a real speedup. The code checks
`os.cpu_count()` and tells you the truth rather than claiming a result
it can't demonstrate.

---

## The parallel graph

```
                    ┌→ research ─┐
  START → plan ═════┼→ research ─┼→ synthesize → END
                    └→ research ─┘
             (Send, dynamic)   (implicit barrier)
```

**Verified concurrent**, with a fake LLM sleeping 0.5s per call:

```
4 branches × 0.5s + plan 0.2s + synth 0.5s
  sequential would be:  2.70s
  measured (parallel):  1.21s   ✅
```

---

## The reducer lesson (LangGraph fails loudly here — good)

Parallel branches all write `{"findings": [...]}` to the same key. Without
a reducer:

```
InvalidUpdateError: At key 'findings': Can receive only one value per
step. Use an Annotated key to handle multiple values.
```

With `Annotated[list, operator.add]`:

```
[0, 1, 2, 3]  — 4/4 kept
```

LangGraph **refuses to guess** whether you meant "replace" or "combine".
It errors instead of silently dropping three of four results.

That's the same design instinct as async SQLAlchemy's `MissingGreenlet`
in Project 2: turn a silent data-loss bug into a loud startup error.

**RULE: any State key written by parallel branches needs a reducer.**

---

## Static vs dynamic fan-out

```python
# STATIC — branch count known at build time. Just add edges.
g.add_edge("plan", "search_web")
g.add_edge("plan", "search_docs")     # these run concurrently

# DYNAMIC — branch count comes from runtime data. Needs Send().
def fan_out(state):
    return [Send("research", {"subquestion": sq})
            for sq in state["subquestions"]]     # N decided by the planner
```

`Send` also carries a **per-branch payload** — each researcher gets only
its own sub-question, not the whole state. Clean isolation, and smaller
prompts.

---

## The two catches nobody mentions

**1. Parallelism only works on INDEPENDENT work.** The planner prompt
explicitly demands sub-questions that don't depend on each other. If
sub-question 2 needs sub-question 1's answer, fan-out is simply the wrong
shape — you need a chain. That's a design constraint, not a prompt trick.

**2. Fan-in waits for the SLOWEST branch.** Three branches at 0.5s and one
at 4s means 4s total. One wedged branch holds the entire response hostage.
Production fan-out needs per-branch timeouts:

```python
async with asyncio.timeout(5):
    result = await llm.ainvoke(...)
```

---

## Try these

**1. Run the lab first.** `python concurrency.py`. Read the numbers before
reading any more explanation.

**2. Compare research modes.** Click "Run Both & Compare". Parallel should
be roughly *N times* faster, where N is the branch count.

**3. Break the reducer.** In `agent.py`, change
`Annotated[list[dict], operator.add]` to plain `list[dict]`. Run a
research query → `InvalidUpdateError`. Put it back.

**4. Make fan-out pointless.** Change the planner prompt to ask for
*dependent* sub-questions ("each should build on the previous"). Run it.
The branches still execute in parallel — but branch 2 can't use branch 1's
answer, so the synthesis gets worse. **Parallelism isn't free; it requires
independence.**

**5. Simulate a slow branch.** In `research()`, add
`if "X" in sq: await asyncio.sleep(5)`. Watch total time jump to ~5s even
though the other branches finished instantly. That's tail latency.

**6. Find the blocking call.** In `research()`, swap `await
llm.ainvoke(...)` for a `time.sleep(1)` before it. Run two research
requests at once from two browser tabs — the second waits for the first.
That's exp5, in your own code.

**7. Raise MAX_BRANCHES.** Set it to 10 and run. More branches = more
concurrent LLM calls = closer to a rate limit. Then add a semaphore
around `research()` (exp4's pattern) and watch it bound.

---

## Check yourself

1. Why is `await` inside a `for` loop sequential?
2. Why don't threads speed up CPU-bound Python, but do speed up HTTP calls?
3. Why does a State key written by parallel branches need a reducer?
4. When do you need `Send()` instead of just adding edges?
5. What determines the total time of a fan-out?
6. Why does `cpu_work` have to be a module-level function?

<details><summary>Answers</summary>

1. Each `await` completes before the next iteration begins. Nothing
   overlaps. Use `gather` to start them all before awaiting any.
2. The GIL lets only one thread run Python bytecode at a time — so CPU
   work just takes turns. But the GIL is **released** during I/O, so
   threads waiting on the network genuinely overlap.
3. Multiple branches writing the same key in one step is ambiguous.
   LangGraph raises `InvalidUpdateError` rather than guess. The reducer
   says how to combine them.
4. When the branch **count** depends on runtime data. Static edges are
   fixed at build time; `Send` creates branches from data.
5. The **slowest branch** — fan-in waits for all of them. Hence
   per-branch timeouts.
6. `ProcessPoolExecutor` must **pickle** the function to ship it to
   another process. Lambdas and closures can't be pickled.
</details>

---

**Next:** Project 5 — SandboxExec. Running LLM-generated code safely in
Docker containers, background workers, and LangGraph **subgraphs**.