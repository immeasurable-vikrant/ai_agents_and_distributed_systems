# Project 8 — Memory Agent

> An agent with **two kinds of memory at once**: the conversation you're
> in (short-term) and the person you are (long-term). Start a brand new
> chat and it still knows you're vegetarian.

---

## Files (4)

```
memory.py     LTM with TWO swappable backends: LangGraph Store + mem0
agent.py      recall → respond → remember, with STM and LTM together
main.py       FastAPI
index.html    chat + a live view of the memory store
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn langgraph langchain-openai \
            langgraph-checkpoint-sqlite grandalf

export OPENAI_API_KEY=sk-...
uvicorn main:app --reload

# to use mem0 instead:
pip install mem0ai
MEMORY_BACKEND=mem0 uvicorn main:app --reload
```

---

## STM vs LTM — the distinction this project is built around

```
SHORT-TERM (STM)                    LONG-TERM (LTM)
──────────────────────────          ─────────────────────────────
the checkpointer (Project 3)        the Store (this project)
keyed by thread_id                  keyed by user_id
"what we said in this chat"         "what I know about this person"
full message history                distilled FACTS
grows every turn, gets trimmed      grows slowly, curated
gone when the chat ends             survives forever
```

**Proven by running it** — same user, two different threads:

```
T1 turn1: saved ['is vegetarian']
T1 turn2: stm: 4  | recalled: ['is vegetarian']
T2 turn1: stm: 2  | recalled: ['is vegetarian']   ← NEW THREAD
          ^^^^^^                ^^^^^^^^^^^^^^^
          fresh conversation    but it still knows you

alice LTM: ['is vegetarian']
bob   LTM: []                                     ← namespace isolation
```

That's the entire point: **STM reset, LTM survived.**

---

## The graph

```
START → recall ──→ respond ──→ remember → END
        (read LTM)  (STM+LTM)   (write LTM)
```

`recall` runs **before** responding so relevant facts land in the prompt.
`remember` runs **after**, so the user isn't waiting on extraction and
the extractor sees the full exchange.

---

## Two backends, one interface

| | LangGraph Store | mem0 |
|---|---|---|
| Extra deps | none | `mem0ai` |
| Extraction | **you write it** | built in |
| Deduplication | you (crudely) | embedding similarity |
| Conflict resolution | you | built in ("I love pizza" → "I don't eat pizza") |
| Decay / forgetting | you | built in |
| You can *see* how it works | **yes** | no |

Build on the Store to understand the machinery. Switch to mem0 when you'd
rather someone else maintained it. `MEMORY_BACKEND=mem0` and the agent
code doesn't change at all — that's what the shared interface buys you.

---

## An honest limitation, verified

`InMemoryStore.asearch(query=...)` with **no embedding index configured
ignores the query completely**:

```
query='vegetarian'    -> ['vegetarian','lives in Gurgaon','loves Kafka']
query='Kafka'         -> ['vegetarian','lives in Gurgaon','loves Kafka']
query='zzzz-nonsense' -> ['vegetarian','lives in Gurgaon','loves Kafka']
```

Not fuzzy matching — **no filtering at all.** I'd initially written
"substring matching" in the comments; testing showed that was wrong, and
the code now says what actually happens.

So `relevant_memories()` fetches everything and uses an **LLM to pick**
which memories matter for this question. That works at tens of memories
and breaks at thousands. Configure the store with an embedding index and
`asearch` becomes real vector search — at which point you can drop the
LLM filtering step entirely.

---

## Why filter at all instead of injecting everything?

At 10 memories you could just dump them all into the prompt. At 500 you'd
blow the context window and pay for irrelevant tokens on **every single
turn**. Selection is what lets long-term memory grow without cost growing
with it.

The same logic drives extraction being **strict**. Most turns should save
nothing. A store full of *"user asked about the weather"* is worse than an
empty one, because noise crowds out the facts that matter at recall time.

---

## Try these

**1. The core demo.** As `alice`: *"I am vegetarian and I live in
Gurgaon."* → watch `+2 new memory`. Then *"What should I cook tonight?"*
→ watch `LTM: recalled`. Now click **+ New conversation** and ask about
cooking again. Empty chat log, same knowledge.

**2. User isolation.** Switch to `bob`, ask the same thing. He knows
nothing about you. Different namespace, structurally.

**3. Watch it save nothing.** Ask *"What is 2 + 2?"* — no memory badge.
Correct behaviour. Then make extraction lenient by deleting *"Most
messages are worth nothing — be strict"* from `EXTRACTION_PROMPT` and try
again. Watch the store fill with junk, then watch recall get worse
because the junk crowds out real facts.

**4. Dedup, and its absence.** Say *"I'm vegetarian"* three times in three
separate conversations. Only the first should save — `extract_facts`
receives the known facts and is told not to repeat them. Now remove
`existing` from that prompt and repeat: three identical memories.

**5. The right to be forgotten.** Click × on a memory, then ask a
question that depended on it. The agent no longer knows. Note how easy
that was *because* LTM is discrete records — deleting a fact from STM
would mean surgically editing a transcript.

**6. Swap the backend.** `pip install mem0ai` and run with
`MEMORY_BACKEND=mem0`. Same UI, same agent code. Tell it *"I love pizza"*,
then later *"Actually I don't eat pizza any more"* — mem0 resolves the
conflict; the Store backend happily keeps both contradictory facts.

**7. Make STM tiny.** Set `KEEP_LAST = 2` in `agent.py`. The agent forgets
what you said two turns ago (STM trimmed) but still knows your long-term
facts. The two mechanisms are genuinely independent.

---

## Check yourself

1. What are the two keys, and what does each scope?
2. Why does `recall` run before `respond` and `remember` after?
3. Why filter memories instead of injecting all of them?
4. Why should extraction be strict?
5. What does mem0 give you that the raw Store doesn't?
6. Why is deleting an LTM fact easy but deleting from STM messy?

<details><summary>Answers</summary>

1. `thread_id` scopes STM (this conversation, via the checkpointer).
   `user_id` scopes LTM (this person, via the Store namespace).
2. `recall` must run first so relevant facts are in the prompt.
   `remember` runs last so the user isn't waiting on extraction, and so
   the extractor sees the complete exchange.
3. Cost and context. At 500 memories, injecting everything blows the
   window and charges you for irrelevant tokens every turn.
4. Noise crowds out signal. A store full of trivia makes recall *worse*,
   because the useful facts are harder to select from.
5. Automatic extraction, embedding-based deduplication, conflict
   resolution, and decay — all things you'd otherwise hand-roll.
6. LTM facts are discrete, addressable records. STM is a message
   transcript, so removing a fact means editing history without breaking
   the conversation's coherence.
</details>

---

**Next:** Project 9 — Eval Harness. LangSmith, datasets, rubric and
trajectory evals, and catching regressions when you change a prompt.