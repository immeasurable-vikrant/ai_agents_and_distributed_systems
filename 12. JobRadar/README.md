# Project 12 — JobRadar

> Jobs come to you. Multi-agent pipeline over a Kafka event bus.
> Target **₹40L+ base**, floor ₹35L, with base-vs-total comp never conflated.

---

## Files (6)

```
bus.py         Kafka event bus (+ in-memory fallback)
sources.py     6 job board adapters, fetched concurrently
agents.py      4 LangGraph agents, split by cost
workers.py     ⭐ the multi-agent pipeline over the bus
store.py       models, DTOs, salary banding, resume layers
main.py        FastAPI dashboard API
index.html     the dashboard
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn "sqlalchemy[asyncio]" aiosqlite \
            httpx langgraph langchain-openai aiokafka

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload        # → http://localhost:8000

# with real Kafka:
docker run -d -p 9092:9092 apache/kafka:latest
KAFKA_URL=localhost:9092 uvicorn main:app
```

**Kafka is optional.** Without `KAFKA_URL` the bus falls back to
`asyncio.Queue` with the same API — same topology, no durability. That
contrast is itself a lesson.

---

## ⭐ The multi-agent architecture

```
ingest ──→ [jobs.discovered] ─┬→ classifier  (group: classify)
                              └→ metrics     (group: metrics)
classifier ─→ [jobs.classified] ──→ salary   (group: salary)
salary ─────→ [jobs.enriched]   ──→ dashboard
```

**Two patterns in one picture:**

**PUB/SUB FAN-OUT** — classifier and metrics read the *same* topic with
*different* group ids, so both see every job. Adding a third observer
requires touching nothing that already exists.

**QUEUE HANDOFF** — classifier publishes to `jobs.classified`; the salary
worker consumes it as a separate process. The salary agent can be slow,
crash, or restart without ingestion noticing. Run two with the same group
id and they split the load.

**Verified running:**
```
[classifier] started (group=classify)
[metrics] started (group=metrics)
[salary] started (group=salary)
[metrics] saw job 1        ← fan-out: metrics AND classifier both got it
[metrics] saw job 2

Senior Backend Engineer  status=enriched fit=85  band=GREEN base=46L total=72L
Marketing Manager        status=enriched fit=40  band=UNKNOWN
                         note: Skipped comp research (fit 40 < 60)   ← cost gate
```

---

## The salary problem — and why base/total are never merged

Most public comp data reports **total**. A "₹58L" figure is often
₹34L base + ₹18L ESOPs + ₹6L bonus — which **fails your ₹40L base bar
while looking like it clears it easily**.

So the agent returns them separately, with a confidence on base alone:

| Band | Condition |
|---|---|
| 🟢 `GREEN` | base confirmed **≥ ₹40L** |
| 🟡 `YELLOW_HIGH` | base confirmed **₹35–40L** |
| 🟡 `YELLOW_UNCONFIRMED` | total ≥ ₹40L but **base unknown** ⚠️ |
| 🔴 `RED` | base or total < ₹35L |
| ⚪ `UNKNOWN` | no data — ask in the first call |

**Verified:**
```
base 45L confirmed     → GREEN
base 37L confirmed     → YELLOW_HIGH   "above your ₹35L floor, below your ₹40L target"
only total 58L known   → YELLOW_UNCONFIRMED  "⚠️ Base NOT confirmed... ask for the split"
base 30L confirmed     → RED           "below your ₹35L floor"
nothing found          → UNKNOWN       "ask about base in the first call"
```

**Nothing is filtered out.** Red is dimmed and sorted last, because a
company below your number can still be worth a conversation — that's your
call, not the tool's.

---

## Resume tailoring — three layers

| Layer | What | When |
|---|---|---|
| **1. Master** | your real experience, full detail | written once, never modified |
| **2. Variants** | `backend` / `ai_agentic` / `platform` | generated once, **reused** |
| **3. Per-JD tailoring** | reorder, re-word, re-emphasize | **only on approval** |

The classifier tags each job's `archetype`, so most jobs just get the
matching variant at **zero LLM cost**. Full tailoring runs ~5×/week
instead of 200×.

**Why not rewrite per discovered job:** 200 strong-model rewrites is real
money for applications you'll never send — and 200 slightly different
versions of your career is a consistency risk if two reach the same
company.

**The hard rule: reframe, never invent.** Plus a mandatory
`changes_summary` so you audit every edit in ten seconds. If the JD
demands Kubernetes and you don't have it, the agent surfaces your closest
real experience — it does not claim Kubernetes.

---

## What this deliberately does NOT do

**No LinkedIn/Naukri scraping.** Their partner APIs are enterprise-only
(incorporated companies, months of approval, thousands/month), and a
personal job-search tool *competes with their product* — an automatic
rejection. Scraping violates ToS and risks your own profile, which is
itself a job-hunting asset.

**That costs less than it sounds.** Most LinkedIn postings redirect to
Greenhouse, Lever, Ashby or Workday. Pulling those is getting the job **at
its source**, often before LinkedIn syndicates it. For manual finds,
the paste box runs the identical pipeline.

**No auto-submission.** Violates ATS terms, employers flag bulk
applicants, and a bad submission can't be un-sent. On approval you get a
tailored resume, cover letter and the apply link. Your effort per job
drops ~20 min → ~30 sec; the final click stays yours.

---

## Two bugs I hit building this

**1. `"Remote - US"` scored 9h timezone overlap.** The generic key
`"remote"` matched before `"us"` in the longest-match lookup, so a
US-Pacific role looked same-timezone as Gurgaon. `"remote"` is now
deliberately **not** a key — a bare "remote" tells you nothing about a
timezone. Correct output now: `Remote - US → 0h (night shift)`.

**2. The dedupe key needed company-suffix stripping.** `"Stripe"` and
`"Stripe Inc."` hashed differently, so the same job appeared twice.
Verified fixed:
```
'Stripe' vs 'Stripe Inc.'    → same key ✅
'(Remote)' title suffix      → same key ✅
```

---

## Try these

**1. Watch the fan-out.** Run it and watch the logs: `[metrics] saw job N`
appears for every job the classifier also processes. Different group ids,
same topic. Now comment out `metrics_worker` from `start_workers` — the
classifier is unaffected. Independent consumers.

**2. Watch the cost gate.** Paste a job clearly outside your field (a
marketing role). Fit scores low, and the salary worker skips research
entirely — `"Skipped comp research (fit 40 < 60)"`. Set `FIT_GATE = 0` in
`agents.py` and watch every junk job trigger a strong-model call.

**3. Trap the visa classifier.** Paste two near-identical jobs differing
only in one line: *"We sponsor work visas"* vs *"Candidates must already
hold work authorization. We do not sponsor."* Both contain "sponsor" and
"visa". Check `visa_evidence` — it should quote the exact sentence it
used. Getting the second one wrong is the most expensive failure in this
system.

**4. Feel the durability difference.** Run with the memory bus, trigger an
ingest, and `Ctrl-C` mid-enrichment. In-flight events are gone. Now run
with `KAFKA_URL=localhost:9092` and do the same — on restart, the
consumer resumes from its last committed offset.

**5. Scale a worker.** Start a second `salary_worker` with the same group
id. With Kafka they split the partitions. Change one to a *different*
group id and both process every job — twice the cost, no extra throughput.

**6. Break the resume rule.** Delete rule 1 (*"NEVER invent"*) from
`TAILOR_SYS`, approve a job demanding a skill you lack, and read the
output. That resume would go out with your name on it.

**7. Test the base/total trap.** Paste a job and have the salary agent
return only total comp. Confirm it bands `YELLOW_UNCONFIRMED` with the
warning — not green. That single distinction is the difference between
applying to the right jobs and wasting weeks.

---

## Check yourself

1. What decides pub/sub vs queue semantics on one topic?
2. Why does ingestion do no LLM work?
3. Why are base and total comp stored separately?
4. Why is red sorted-last rather than filtered out?
5. Why does resume tailoring run on approval, not discovery?
6. Why is not scraping LinkedIn a small loss?

<details><summary>Answers</summary>

1. The `group_id`. Same group = consumers split the work. Different
   groups = each group receives every message.
2. So a slow or failing agent can never block new jobs arriving.
   Ingestion is fast and free; enrichment is slow and costly. The log
   between them buys independence.
3. Public sources report total. A ₹58L total can be ₹34L base — which
   fails a ₹40L base bar while appearing to clear it. Conflating them
   produces confidently wrong recommendations.
4. Because filtering is a decision, and it's yours. A company below your
   number can still be worth a conversation.
5. ~5 tailoring calls a week instead of 200 — and fewer divergent
   versions of your career reaching the same company.
6. Most LinkedIn postings redirect to Greenhouse/Lever/Ashby/Workday
   anyway. Pulling from the source gets the same job, often earlier, with
   no ban risk.
</details>

---

## What's in your master resume

The dummy resume in `store.py` reflects: Birdeye (1.7y), Bangalore
product startup (8mo), Gurgaon services company (3.8y), Bangalore startup
(3mo) — **6.4 years total** — plus the systems you built across projects
1-11 as an independent-projects section. Replace `MASTER_RESUME` with
your real one when ready; nothing else changes.