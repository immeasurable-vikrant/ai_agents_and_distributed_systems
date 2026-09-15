# Project 10 — Planner Agent

> An agent that writes a **plan** before acting, runs independent steps
> in **parallel**, and **replans** when a step fails — with a ReAct
> baseline running the same goal so the difference is visible, not
> theoretical.

---

## Files (4)

```
planner.py          plan → execute (parallel) → replan → synthesize
react_baseline.py   the SAME tools, driven by a ReAct loop
main.py             FastAPI — runs either architecture
index.html          side-by-side comparison
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn langgraph langchain-openai grandalf

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

---

## The two architectures

```
ReAct (Projects 1-9)              PLAN-AND-EXECUTE (this one)
──────────────────────────        ────────────────────────────────
one LLM call PER STEP             one planning call, then execute
no plan exists anywhere           the plan is inspectable state
strictly sequential               independent steps run in PARALLEL
adapts implicitly each turn       adapts via explicit REPLANNING
can't show intent in advance      can show intent BEFORE acting
good for short/exploratory        good for long/dependent work
```

**Neither wins outright.** ReAct is simpler and adapts without ceremony.
Planning costs a planning call up front and buys you visibility,
parallelism, and a plan a human could approve *before* anything runs.

---

## The graph

```
        +-----------+
        | __start__ |
        +-----------+
           +------+
           | plan |
           +------+
          +---------+
          | execute |
          +---------+
          *         .
+--------+        +------------+
| replan |        | synthesize |
+--------+        +------------+
                    +---------+
                    | __end__ |
                    +---------+
```

---

## A real run (verified)

Goal: *"What is our ARR and churn?"* — there is **no `ARR` metric**, so
step 1 fails.

```
[plan]
    0. get_metrics('ARR')    [pending]
    1. get_metrics('churn')  [pending]
  · planned 2 steps

[execute]
  · batch of 2: [0, 1] (parallel)      ← independent steps, run together
  ·   step 0 [failed] get_metrics(ARR)
  ·   step 1 [done]   get_metrics(churn)

[replan]
    0. get_metrics('revenue') [pending]
  · ↻ REPLAN #1: 1 new steps           ← works around the failure

[execute]
  ·   step 0 [done] get_metrics(revenue)
  · synthesized final answer
```

**Two things to notice.**

**Parallel batching.** Steps 0 and 1 have no `depends_on` relationship,
so they ran concurrently. The planner knows the dependency graph because
it *wrote* it — ReAct can't do this, because it doesn't know step 2
exists until step 1 finishes.

**Replanning kept the work that succeeded.** Churn had already been
fetched; the replan only rewrote the *remaining* work. Replanning from
scratch would discard successful steps — slow at best, and dangerous when
steps have side effects (imagine re-running a payment).

**The same goal through ReAct:**

```
llm_calls: 3 | replans: 0
  · call metrics(ARR)
  ·   failed → ERROR: no metric named 'ARR'. Available: revenue, churn...
  · call metrics(revenue)
  ·   done → [metrics] revenue: $4.2M, up 18% YoY
```

Same recovery, different mechanism: ReAct just *re-decided* on its next
turn. No plan, no replan event, no parallelism — but also no planning
overhead.

---

## Why the tools return readable errors

```python
return f"ERROR: no metric named '{metric}'. Available: {', '.join(data)}"
```

Not `ERROR: invalid input`. The replanner **reads this string** to decide
what to do differently — an opaque error teaches it nothing, and it will
either give up or retry the same thing.

**Error messages written for an LLM to read are a real design concern.**
Same instinct as Project 2's tools listing valid project names on failure.

---

## Why replanning is bounded — and allowed to give up

Two guards, both necessary:

```python
MAX_REPLANS = 2     # stop re-planning forever
MAX_STEPS   = 6     # stop planning absurdly long plans
```

And the replan prompt explicitly permits an **empty plan**:

> *"If the failure is structural and no tool can work around it, return
> an EMPTY list of steps — giving up honestly is better than looping."*

Some failures aren't solvable by trying differently. An agent that always
finds "another approach" will loop forever pretending to make progress.
Being allowed to stop is what makes bounded replanning honest — the same
lesson as Project 5's sandbox retry cap.

---

## Try these

**1. Watch a replan.** Ask *"What is our ARR and how does churn compare?"*
There's no ARR metric. Watch the planner fail, replan, and keep the churn
result.

**2. Watch parallel batching.** Ask *"Look up our revenue, churn, and
runway"* — three independent steps. The log shows
`batch of 3: [0, 1, 2] (parallel)`. Now ask *"Get revenue and headcount,
then compute revenue per employee"* — the calculation `depends_on` both,
so it runs in a second batch.

**3. Make replanning impossible.** Set `MAX_REPLANS = 0`. Ask the ARR
question. The agent reports honestly that it couldn't get ARR, instead of
silently substituting revenue. **A plan without replanning is a script.**

**4. Break the error messages.** In `get_metrics`, change the error to
just `"ERROR"`. Run the ARR goal again. The replanner now has nothing to
work with and either gives up or retries the same bad argument.

**5. Compare LLM call counts.** Run both architectures on a 3-step goal.
The planner spends 1 call planning + 1 synthesizing; ReAct spends one per
step plus one to answer. On long tasks, planning is *cheaper*. On a
one-step task, it's pure overhead.

**6. Remove `depends_on`.** Make the planner always return `[]` for it.
Every step becomes "independent" and runs in one parallel batch —
including steps that needed earlier results. Watch the calculation fail
because its inputs weren't ready.

**7. Add a human approval gate.** Insert Project 6's `interrupt()`
between `plan` and `execute`. Now the user sees and approves the plan
before anything runs. **This is the single biggest practical advantage of
planning over ReAct** — you cannot approve a ReAct agent's plan, because
it doesn't have one.

---

## Check yourself

1. What can a planner do that a ReAct loop structurally cannot?
2. Why does replanning keep completed steps instead of starting over?
3. Why must tool errors be human-readable?
4. Why is the replanner allowed to return an empty plan?
5. When is ReAct the better choice?
6. What does `depends_on` enable?

<details><summary>Answers</summary>

1. Show its intentions **before acting** (so a human can approve them),
   and run independent steps in **parallel** — both require knowing all
   the steps up front.
2. Discarding successful work is slow, and unsafe when steps have side
   effects. Only the remaining work needs rewriting.
3. The replanner reads the error text to decide what to do differently.
   `ERROR: invalid input` gives it nothing to reason about.
4. Some failures are structural. An agent that always finds "another
   approach" loops forever pretending to progress.
5. Short, exploratory tasks where the next step genuinely depends on what
   you just learned — and where planning overhead would dominate.
6. The dependency graph, which lets independent steps be batched and run
   concurrently instead of strictly in sequence.
</details>

---

**Next:** Project 11 — Agent Platform. Microservices, an API gateway,
worker processes, and MCP client/server: the systems capstone.