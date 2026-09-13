# Project 5 — SandboxExec

> An agent that writes Python, runs it in a sandbox, reads the error,
> and fixes itself — with the retry loop packaged as a **subgraph**.

---

## Files (5)

```
sandbox.py    isolated execution — Docker + subprocess fallback
agent.py      LangGraph SUBGRAPH nested in a parent graph
main.py       FastAPI
index.html    agent panel + "attack your own sandbox" panel
README.md
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn langgraph langchain-openai grandalf

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

**Docker is optional but strongly recommended.** With it, you get real
isolation. Without it, the app falls back to a weaker subprocess sandbox
and says so loudly in the UI banner and in `/api/health`.

---

## What you learn

### Backend half

| Concept | Where |
|---|---|
| Why `exec()` on LLM output is an RCE hole | `sandbox.py` header |
| Container isolation flags, each justified | `_run_docker` |
| `--network none` as the single most important flag | same |
| OS resource limits (`RLIMIT_*`) | `_LIMITS_PREAMBLE` |
| **Scrubbing the environment** so secrets can't leak | `env={"PATH": ...}` |
| Timeouts that actually kill the process | both backends |
| Truncating output so a print-loop can't fill your DB | `_result` |
| Reporting *which* security posture is active | `/api/health` |

### LangGraph half

| Concept | Where |
|---|---|
| **Subgraphs** — a compiled graph used as a node | `run_coder` |
| **Scoped state** — the parent never sees `attempts` | `CoderState` vs `MainState` |
| Self-correction: errors fed back as context | `write_code` |
| Bounded retry loops | `should_retry` |
| Nested graph visualisation | `/api/graph` |

---

## The subgraph

```
PARENT:   START → understand → [ coder ] → explain → END
                                   │
SUBGRAPH:          START → write_code → execute → (failed? loop back) → END
                               ↑____________________|
```

**Verified with a fake LLM that fails once, then fixes:**

```
attempts: 2 | success: True
stdout  : fixed output
trace   : ['understand', 'coder', 'explain']
```

Note the trace: **three steps**, even though the coder internally wrote
bad code, ran it, read the traceback, rewrote it, and ran again. The
retry loop stayed inside the subgraph.

**Why that matters:** without subgraphs, `attempts` and `stderr` would
have to live in one shared State that every node sees. Four reasons to
nest:

- **Reuse** — drop the coder into any parent graph
- **Testable** — run and debug the subgraph alone
- **Readable** — the parent is 3 steps, not 8
- **Scoped state** — the parent doesn't carry the child's plumbing

---

## The sandbox — two backends, honestly labelled

| | Docker | Subprocess fallback |
|---|---|---|
| Network | **none** — no interface exists | ⚠️ shares yours |
| Filesystem | read-only + tmpfs | ⚠️ shares your view |
| Memory | capped by kernel cgroups | ⚠️ **cannot cap** (see below) |
| CPU | `--cpus 0.5`, structurally bounded | `nice(19)` + `RLIMIT_CPU` |
| Fork bombs | `--pids-limit 50` | `RLIMIT_NPROC` |
| File writes | read-only fs | `RLIMIT_FSIZE 0` |
| Secrets | fresh container env | scrubbed `env=` |

---

## Three things I found by actually breaking this

These are in the code comments because the failures *are* the lesson.

**1. A CPU spin starved the parent.** Running `while True: pass` in the
subprocess backend on a 1-core machine starved the parent process so
badly it couldn't run its own timeout to kill the child. The whole
environment locked up. Fix: `os.nice(19)` on the child, so the parent
always gets scheduled. Docker's `--cpus 0.5` prevents this
*structurally* — that difference is exactly why Docker is the real
backend.

**2. `RLIMIT_AS = 128MB` killed Python itself.** The obvious way to cap
memory in a subprocess crashed on `print(1+1)`. Python 3.12 maps far more
*virtual* address space than that just to start, and `RLIMIT_AS` limits
address space, not resident memory. Real memory limiting needs cgroups,
i.e. Docker. So the fallback simply doesn't cap memory, and says so.

**3. `RLIMIT_FSIZE(0,0)` is subtler than it looks.** The file still gets
**created** — it just stays 0 bytes. And because Python buffers writes,
a small write *looks* like it succeeded; the failure only surfaces at
flush or close:

```
write without flush  →  "wrote"          (buffered, never hit the limit)
write with flush     →  BLOCKED          (traceback)
on disk              →  0 bytes
```

So it blocks exfiltration-to-disk, but it is not "no filesystem access".
Docker's `--read-only` is the stronger guarantee.

**Verified blocked in the fallback:** secrets (`KEY=None` — scrubbed
env), file *content* writes, and syntax errors handled as data rather
than crashes.

---

## Try these

**1. Read your own secrets.** Paste into the attack panel:
```python
import os
print(os.environ.get("OPENAI_API_KEY"))
```
→ `None`. The child got a deliberately empty environment. Now delete the
`env={...}` line in `_run_subprocess` and try again — **your API key
prints**. That one line is the whole defense.

**2. Phone home.** Try the `urllib.request` example. Under Docker it
fails with no network at all. Under the fallback it *succeeds* — which
is the honest demonstration of why the fallback isn't a real boundary.

**3. Write a file.** Try it with and without `f.flush()`. Notice the
buffering nuance from finding #3 above.

**4. Ask for something impossible.** *"Fetch the current bitcoin price
from an API."* The sandbox has no network, so the code fails, the agent
reads the error and rewrites — and fails again, because the problem is
structural, not a bug. After `MAX_FIX_ATTEMPTS` it stops and says so.
**A bounded retry loop that admits defeat beats one that loops forever.**

**5. Remove the retry cap.** Set `MAX_FIX_ATTEMPTS = 99` and rerun #4.
Watch it burn LLM calls on a problem it cannot solve. Put it back.

**6. Collapse the subgraph.** Inline `write_code` / `execute` / the retry
edge directly into the parent graph. It works — but now `MainState` needs
`attempts` and `stderr`, and the parent trace shows every internal retry.
That's what the subgraph was buying you.

**7. Compare backends.** If you have Docker, run once normally, then
`SANDBOX_BACKEND=subprocess uvicorn main:app`. Same attacks, different
outcomes. The `/api/health` warning changes too.

---

## Check yourself

1. Why is `exec()` on LLM-generated code a remote code execution hole?
2. Which single Docker flag matters most, and why?
3. Why can't the subprocess backend cap memory with `RLIMIT_AS`?
4. Why does the child get `os.nice(19)`?
5. What four things does packaging the retry loop as a subgraph buy you?
6. Why is a *bounded* retry loop better than an unbounded one?

<details><summary>Answers</summary>

1. It runs with your process's permissions — your files, your network,
   your environment variables (API keys, DB passwords). An infinite loop
   also freezes your event loop and every concurrent request with it.
2. `--network none`. The container gets no network interface at all, so
   the code physically cannot exfiltrate data or fetch a payload — it's
   not a rule being enforced, it's a capability that doesn't exist.
3. `RLIMIT_AS` caps *virtual address space*, and Python 3.12 maps far
   more of that than its actual memory use — so a realistic limit kills
   the interpreter at startup. Real memory capping needs cgroups.
4. So a CPU-spinning child can't starve the parent out of running its own
   timeout. Found by locking up an environment exactly that way.
5. Reuse, independent testability, a readable parent trace, and scoped
   state (the parent never carries `attempts`/`stderr`).
6. Some failures are structural, not bugs — no amount of rewriting fixes
   "there is no network." An unbounded loop burns money forever on those.
</details>

---

**Next:** Project 6 — Live Debate. WebSockets, Redis pub/sub, token
**streaming**, and **human-in-the-loop** with durable interrupt/resume.