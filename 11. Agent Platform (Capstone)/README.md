# Project 11 — Agent Platform (Capstone)

> Three services, two databases, one gateway — plus an MCP server so
> Claude Desktop (or anything) can call your agents.

---

## Files (6)

```
gateway.py         :8000  the only public port — routing, rate limit, tracing
auth_service.py    :8001  own DB, RS256 signing, /public-key + /verify
agent_service.py   :8002  own DB, supervisor + specialists, auth enforced itself
mcp_server.py             standalone MCP server: tools + resources + prompts
mcp_client.py             MCP client — consumes ours AND someone else's
index.html                live topology + supervisor UI
run.sh                    starts all three
```

---

## Run

```bash
pip install "fastapi[standard]" uvicorn "sqlalchemy[asyncio]" aiosqlite bcrypt \
            "python-jose[cryptography]" httpx langgraph langchain-openai mcp

export OPENAI_API_KEY=sk-...
./run.sh                       # → http://localhost:8000

python mcp_client.py           # MCP round-trip (no API key needed)
```

---

## Architecture

```
  browser → :8000 gateway ─┬→ :8001 auth service   (auth.db)
                           └→ :8002 agent service  (agent.db)

  Claude Desktop → mcp_server.py → :8000 gateway → ...
```

---

## What you learn

### Backend

| Concept | Where |
|---|---|
| **Database-per-service** + its real cost | `Job.org_id` comment |
| **RS256** vs HS256, and why services force the change | `auth_service.py` header |
| `/public-key` as an unauthenticated endpoint | `public_key()` |
| **Local vs remote** token verification, both built | `verify_local` / `verify_remote` |
| Key **persistence** across restarts | `load_or_create_keys` |
| **API gateway**: routing, rate limit, aggregated health | `gateway.py` |
| **Trace-ID propagation** across services | `add_trace_id` middleware |
| Services enforcing auth **themselves** | `current_user` |
| 502 vs 500 — whose fault is it? | `proxy()` |

### Agentic

| Concept | Where |
|---|---|
| **Supervisor pattern** — delegate, don't do | `supervisor()` |
| Specialist subagents, dynamic fan-out | `delegate()` / `Send` |
| **MCP server** (standalone, stdio) | `mcp_server.py` |
| MCP **tools vs resources vs prompts** | same |
| **MCP client** + runtime capability discovery | `mcp_client.py` |
| Consuming a server you didn't write | `consume_external_server` |

---

## The result that matters most

**Local vs remote verification, with the auth service killed:**

```
public key cached: True

💀 auth service STOPPED

LOCAL  verify -> 200  ✅ still works (cached key)
REMOTE verify -> 503 Auth service unreachable  ❌ dead
```

That's the whole microservices trade-off in four lines.

**Local:** fetch the public key once, cache it, verify forever. The agent
service survives auth being down. The cost: revoking a user doesn't take
effect until their token expires.

**Remote:** ask auth on every request. Revocation is instant. The cost:
every request in your system now depends on one service being up — you've
recreated the single point of failure microservices were supposed to
remove.

Neither is "right." Pick per the revocation latency you can tolerate.

---

## Other verified results

**RS256 — the public key really is enough to verify:**
```
🔑 decoded with PUBLIC key only: {'sub':'1','org_id':1,'role':'admin','iss':'auth-service'}
```
Only the auth service holds the private key. Five services verifying
tokens means five services that *cannot* forge them — which is exactly
what HS256's shared secret fails to give you.

**Keys persist across restart:**
```
key before restart: 916a1875faaa2d41
key after  restart: 916a1875faaa2d41
old token still valid after restart: True
```
Regenerating keys on boot logs out every user on every deploy.

**Full stack through the gateway, with tracing:**
```
[gateway] tr-59d3812fc5 POST /api/login → 200 (394ms)
[gateway] tr-a9094b9c4b GET  /api/jobs  → 200 (74ms)
gateway health -> {'gateway':'ok', 'auth':'ok', 'agent':'ok'}
```

**The gateway is NOT the security boundary:**
```
direct :8002 no token -> 401    ← service enforces auth itself
direct :8002 w/ token -> 200
```

**MCP, full protocol round-trip:**
```
✅ connected to our agent-platform server
TOOLS (2):      ask_agent, list_recent_jobs
RESOURCES (2):  platform://health, platform://architecture
PROMPTS (2):    /deep_dive, /review_activity
READING platform://architecture: { ... }
```

---

## Two bugs I hit building this

**1. `mcp` 2.x renamed `FastMCP` → `MCPServer`.** Every tutorial written
before the 2.0 release uses `from mcp.server.fastmcp import FastMCP`,
which now raises a `ModuleNotFoundError` with a migration hint. This
project uses the 2.x API; pin `mcp<2` if you're following older material.

**2. `Mapped` without a type parameter.** I wrote
`created_at: Mapped = mapped_column(...)` and SQLAlchemy 2.x rejected the
whole class with `MappedAnnotationError`. The annotation must be
`Mapped[datetime]` — the generic parameter isn't decoration, it's how the
ORM infers the column type.

---

## Connect Claude Desktop to your agents

```json
{
  "mcpServers": {
    "agent-platform": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/absolute/path/to/project-11",
      "env": { "MCP_TOKEN": "<paste the JWT from the UI>" }
    }
  }
}
```

Restart Claude Desktop and ask it to use `ask_agent`. **That moment — an
app you didn't write, calling a multi-agent platform you did, over a
protocol neither of you negotiated — is what MCP is for.**

---

## Try these

**1. Kill auth, compare strategies.** Start with `./run.sh`, log in, then
`Ctrl-C` just the auth service. Ask a question → still works. Now restart
with `VERIFY_MODE=remote ./run.sh`, kill auth again → 503. The most
important exercise here.

**2. Bypass the gateway.** `curl localhost:8002/jobs` → 401. Now delete
the `Depends(current_user)` from `/jobs` and try again — the service is
wide open to anything that can reach port 8002. The gateway never
protected it.

**3. Prove database-per-service.** `sqlite3 auth.db ".tables"` shows
`orgs, users`. `sqlite3 agent.db ".tables"` shows `jobs`. There is no
query that joins them — that's the cost you accepted.

**4. Break key persistence.** Delete `jwt_key.pem` and restart auth. Every
existing token now fails. This is uncoordinated key rotation, and it's why
production uses JWKS with a `kid` so old and new keys coexist.

**5. Follow a trace.** Ask a question, note the `trace_id` in the UI, then
grep it across all three terminal windows. One id, the whole journey.

**6. Trip the rate limiter.** Set `RATE_LIMIT = 3` in `gateway.py` and
send four requests → 429 with `Retry-After`.

**7. The `--workers` trap.** Run the gateway with `--workers 4`. The rate
limit becomes 4× looser, because `_hits` is per-process memory. This is
Project 4's lesson in production form — shared state belongs in Redis.

**8. Watch the supervisor economize.** Ask *"What is the capital of
France?"* → probably one specialist. Ask *"Should we migrate our monolith
to microservices?"* → likely all three. Now delete *"Assigning all three
when one would do wastes time and money"* from the supervisor prompt and
watch it over-assign.

**9. Add a third service.** Copy `agent_service.py` into
`search_service.py` on :8003 with its own DB, and route `/api/search` to
it in the gateway. Notice what you had to touch: the gateway and nothing
else.

---

## Check yourself

1. Why RS256 here when HS256 was fine in Project 7?
2. What does local verification buy, and what does it cost?
3. Why does `Job` have no foreign key to `organizations`?
4. Why does the agent service authenticate when the gateway already did?
5. What's the difference between an MCP tool, resource, and prompt?
6. Why must `/public-key` be unauthenticated?

<details><summary>Answers</summary>

1. HS256's shared secret means every service that can *verify* can also
   *forge*. With five services that's five places a forging key could
   leak. RS256 keeps the private key in one service; everyone else gets
   a public key that can only verify.
2. Buys: independence — proven above, the agent service survived auth
   being killed. Costs: revocation isn't instant; a deleted user's token
   works until it expires.
3. The `organizations` table is in another service's database. Foreign
   keys can't span databases. You trade database-enforced referential
   integrity for independent deployability.
4. Because anything that reaches port 8002 — another pod, a bad network
   policy, a developer on the host — bypasses the gateway entirely. The
   gateway is convenience; the service is the boundary.
5. **Tool** = the model decides to call it. **Resource** = the client
   loads it by URI as context. **Prompt** = the user picks it, like a
   slash command.
6. A public key is public by design — that's the premise of asymmetric
   crypto. Requiring auth to fetch it would create a chicken-and-egg
   problem: you'd need a verified token to get the key you need to
   verify tokens.
</details>

---

## Where the series stands

Projects 1-11 cover the backend and agentic foundations end to end. Two
multi-agent patterns are still unbuilt and worth knowing: **queue-based
handoff** (Agent A → Redis → Agent B, fully decoupled processes) and
**event-bus multi-agent** (Kafka consumer groups, fan-out to independent
agents). Both are natural extensions of this project's worker section.

**Next:** Project 12 — the LinkedIn job-application agent, rebuilt on
everything here.