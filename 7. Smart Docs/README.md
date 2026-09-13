# Project 7 — Smart Docs

> Multi-tenant RAG with JWT auth, two layers of tenant isolation, and
> **three RAG strategies** — naive, Corrective, and Self-RAG — built as
> separate graphs so you can run all three on one question and compare.

---

## Files (4)

```
store.py      multi-tenant models, auth, vector search, isolation layers
rag.py        THREE graphs: naive / corrective / self-RAG
main.py       FastAPI with JWT-gated endpoints
index.html    tenant switcher + 3-way RAG comparison
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn "sqlalchemy[asyncio]" aiosqlite \
            bcrypt "python-jose[cryptography]" langgraph langchain-openai \
            openai grandalf

export OPENAI_API_KEY=sk-...      # needed for the RAG graphs
uvicorn main:app --reload
```

Two seeded tenants, password `password123`:
`alice@acme.com` (Acme) and `bob@globex.com` (Globex).

Without an API key, auth and isolation still work (embeddings fall back
to a deterministic hash), but `/api/ask` returns 503 — the graphs need a
real LLM.

---

## What you learn

### Backend half

| Concept | Where |
|---|---|
| JWT auth, stateless (no session store) | `make_token` / `read_token` |
| bcrypt password hashing | `hash_pw` |
| Carrying `org_id` **in the token** | `make_token` |
| **Multi-tenant isolation, layer 1** (query filter) | `search()` |
| **Defense in depth, layer 2** (independent check) | `assert_same_org` |
| Fail-closed on a suspected leak | `ask` endpoint |
| Never trusting client-supplied `org_id` | `DocReq` has no org field |
| Not leaking which of email/password was wrong | `login()` |

### LangGraph half

| Concept | Where |
|---|---|
| **Naive RAG** — retrieve → generate | `build_naive` |
| **Corrective RAG** — grade docs, rewrite query, retry | `build_corrective` |
| **Self-RAG** — also reflect on the OUTPUT | `build_selfrag` |
| Document grading with structured output | `grade_docs` |
| Query rewriting as a correction loop | `rewrite_query` |
| Hallucination checking | `check_grounded` |
| Usefulness checking | `check_useful` |
| Bounded retry on reflection failure | `selfrag_decide` |

---

## The three graphs

```
NAIVE       retrieve → generate
            Trusts whatever came back.

CORRECTIVE  retrieve → grade → (nothing good? rewrite → retry) → generate
            Fixes RETRIEVAL failures: "did I get the right documents?"

SELF-RAG    retrieve → grade → generate → grounded? → useful? → (retry)
            Also fixes GENERATION failures: "did I make that up?"
```

**Measured on the same question** (`"What is the refund policy?"`, Acme):

```
=== NAIVE ===        llm calls: 1
  docs used: ['Acme Refund Policy', 'Acme Onboarding', 'Acme Q4 Revenue']

=== CORRECTIVE ===   llm calls: 4
  docs used: ['Acme Refund Policy']
   ✓ Acme Refund Policy
   ✗ Acme Onboarding
   ✗ Acme Q4 Revenue

=== SELFRAG ===      llm calls: 6
  docs used: ['Acme Refund Policy']
   ✓ Acme Refund Policy   ✗ Onboarding   ✗ Q4 Revenue
   ✓ grounded: all claims in docs
   ✓ useful: answers directly
```

Look at what naive did: it stuffed **Acme Q4 Revenue** — a confidential
document with nothing to do with refunds — into the prompt. It wasn't a
leak (same org), but it's noise that can pull the answer off course.

**The trade is explicit:** 1 call vs 4 vs 6. Neither is "correct" —
Self-RAG is the right answer when a wrong answer is expensive, and
overkill when it isn't.

---

## Two layers of tenant isolation

**Layer 1** — the `WHERE` clause in `search()`:

```
alice (Acme) searches 'refund policy' → org1 Acme Refund Policy, org1 Acme Onboarding...
bob (Globex) searches 'refund policy' → org2 Globex Refund Policy, org2 Globex Hiring Plan
```

**Layer 2** — `assert_same_org()`, an independent check on every
retrieval result.

**Why bother with layer 2 if layer 1 works?** Because layer 1 is one
line of code, and code gets refactored. Here's the buggy version (someone
"simplifies" the query and drops the filter) actually running:

```
BUGGY retrieval returned:
   org2  Globex Hiring Plan     ← 🚨 WRONG ORG
   org1  Acme Onboarding
   org2  Globex Refund Policy   ← 🚨 WRONG ORG
   org1  Acme Refund Policy

Now Layer 2 inspects the same result set:
   🛡️ BLOCKED: Doc 5 belongs to org 2, caller is org 1. Refusing to answer.
   → the leak never reaches the LLM prompt or the user
```

**That's the whole argument.** The broken query genuinely pulled Globex
documents into Acme's result set. Layer 2 stopped them before the prompt.

And it **raises** rather than filtering-and-continuing — because if it
ever fires, cross-tenant data was about to leave the system. Refusing to
answer is recoverable; leaking is not.

---

## Verified at the API level

```
no token      -> 401
bad token     -> 401
alice docs    -> ['Acme Refund Policy', 'Acme Q4 Revenue', 'Acme Onboarding']
bob   docs    -> ['Globex Refund Policy', 'Globex Hiring Plan']
cross-org leak? set()        ← empty = isolated
create doc    -> 201, org 1  ← org came from the TOKEN, not the request body
```

Note `DocReq` has **no `org_id` field**. There is no code path where a
client names its own tenant.

---

## A dependency bug worth knowing

The first version used `passlib` for hashing and every call blew up with:

```
ValueError: password cannot be longer than 72 bytes
```

...on a 11-character password. `passlib` 1.7.x trips an internal
version-detection path against `bcrypt` 4.x. The fix was to drop passlib
and use `bcrypt` directly — one fewer dependency and no broken
compatibility shim. Documented in `store.py` because you'll hit it in
any tutorial written before ~2024.

---

## Try these

**1. Try to read the other tenant's data.** Log in as Alice, ask *"How
many engineers are we hiring next quarter?"* That fact lives in Globex's
document. She gets nothing — retrieval is filtered by her token's org.

**2. Break layer 1, watch layer 2 catch it.** In `store.py`, uncomment
`search_BUGGY` and call it from `rag.retrieve` instead of `search`. Ask
Alice a question. You'll get a 500, and the server log shows
`🚨 TENANT LEAK BLOCKED`.

**3. Now break layer 2 as well.** Comment out the `assert_same_org` call
in `rag.retrieve`. Ask again. **Alice now gets an answer built from
Globex's confidential documents.** That's the leak, with both defenses
removed. Put them both back.

**4. Compare the three modes on an unanswerable question.** Ask *"What is
the CEO's mobile number?"* Naive will often produce something
confident-sounding from whatever ranked highest. Self-RAG's grounding
check catches it and refuses.

**5. Force a query rewrite.** Ask *"Can I get my money back after two
months?"* — different words than "refund policy". Watch corrective mode's
notes: if nothing grades relevant, it rewrites the query and retries.

**6. Forge a token.** Take Alice's JWT, decode it at jwt.io, change
`org_id` to 2, re-encode with a guessed secret, and use it. It fails —
the signature won't verify. Now set `JWT_SECRET` to something obvious
like `"secret"` and try again with that. This is why the secret is an
env var and not a constant.

**7. Make the grader lenient.** In `grade_docs`, remove *"Be strict —
topically adjacent is NOT relevant"* from the prompt. Corrective mode
starts keeping documents naive would have kept, and the quality gap
narrows. The grader's prompt IS the filter.

---

## Check yourself

1. Why keep layer 2 if layer 1 is correct?
2. Why does `assert_same_org` raise instead of just dropping bad docs?
3. Why does `DocReq` have no `org_id` field?
4. What class of failure does Corrective RAG fix that naive can't?
5. What does Self-RAG add on top of that?
6. Why does the token carry `org_id` instead of looking it up per request?

<details><summary>Answers</summary>

1. Layer 1 is a single line that future refactors can break. Layer 2 is
   an independent tripwire that stops the leak even when layer 1 fails —
   demonstrated above with a real buggy query.
2. If it fires, cross-tenant data was about to leave the system.
   Refusing to answer is recoverable; leaking is not. Fail loud, fail closed.
3. Because the org must come from the verified token. Accepting it from
   the client would let anyone write into — or read from — any tenant.
4. **Retrieval** failures: the vector search returned documents that
   aren't actually relevant. CRAG grades them and rewrites the query.
5. **Generation** failures: hallucination (claims not in the documents)
   and uselessness (grounded but evasive). Naive never looks at its own
   output, so it can't detect either.
6. It makes auth stateless — no session store, no DB round-trip per
   request. The cost is staleness: an org change won't take effect until
   the token expires.
</details>

---

**Next:** Project 8 — Memory Agent. **Long-term memory** across
conversations with LangGraph's Store, plus **mem0** as the
production-grade alternative.