"""
gateway.py — the API GATEWAY. The only port the outside world touches.

    browser → :8000 gateway ─┬→ :8001 auth service
                             └→ :8002 agent service

WHAT A GATEWAY IS FOR:
  • ONE public entry point (services bind to localhost/internal network)
  • cross-cutting concerns in one place: rate limiting, tracing, CORS
  • the client doesn't need to know your service topology

WHAT A GATEWAY IS NOT:
  • a security boundary. Each service still authenticates for itself —
    see agent_service.current_user. Exercise 5 bypasses the gateway
    entirely to prove the difference.
"""

import os
import time
import uuid
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse

AUTH_URL = os.getenv("AUTH_URL", "http://localhost:8001")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8002")

app = FastAPI(title="API Gateway")


# ============================================================
# RATE LIMITING — in-process, with an honest caveat
# ============================================================
# A simple fixed-window counter. Project 3 built the proper Lua/Redis
# version; this is deliberately the simple one so the project needs no
# Redis container.
#
# ⚠️ THE CAVEAT THAT MATTERS: this dict lives in ONE process's memory.
# Run the gateway with --workers 4 and you get four independent counters,
# so your "20/min" limit becomes 80/min. That's Project 4's lesson 6,
# reappearing in production form. Shared state must live in Redis.

_hits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT, WINDOW = 30, 60


def rate_limited(client: str) -> bool:
    now = time.time()
    _hits[client] = [t for t in _hits[client] if now - t < WINDOW]
    if len(_hits[client]) >= RATE_LIMIT:
        return True
    _hits[client].append(now)
    return False


# ============================================================
# TRACING
# ============================================================

@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """
    Mint a trace id and attach it to every request.

    WHY THIS EXISTS: in a monolith, one stack trace tells you everything.
    Across three services, a single user action produces three separate
    log streams with no connection between them. The trace id is the
    thread that stitches them together — grep one id, see the whole
    request's journey.

    This is distributed tracing in its crudest useful form. OpenTelemetry
    does this properly with spans, timings and parent-child relationships.
    """
    trace_id = request.headers.get("X-Trace-Id") or f"tr-{uuid.uuid4().hex[:10]}"
    request.state.trace_id = trace_id
    start = time.perf_counter()

    response = await call_next(request)

    ms = round((time.perf_counter() - start) * 1000)
    response.headers["X-Trace-Id"] = trace_id
    print(f"[gateway] {trace_id} {request.method} {request.url.path} "
          f"→ {response.status_code} ({ms}ms)")
    return response


async def proxy(request: Request, target: str, path: str):
    """Forward a request to a service, preserving auth and the trace id."""
    client_ip = request.client.host if request.client else "anon"
    if rate_limited(client_ip):
        return JSONResponse(
            {"detail": f"Rate limit exceeded ({RATE_LIMIT}/min)"},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": "60"},
        )

    headers = {"X-Trace-Id": request.state.trace_id}
    if auth := request.headers.get("Authorization"):
        headers["Authorization"] = auth      # ← pass the token through UNCHANGED
    if request.headers.get("content-type"):
        headers["Content-Type"] = request.headers["content-type"]

    body = await request.body()

    async with httpx.AsyncClient(timeout=60) as c:
        try:
            r = await c.request(request.method, f"{target}{path}",
                                content=body, headers=headers,
                                params=dict(request.query_params))
        except httpx.HTTPError as e:
            # A service being down is a 502/503 at the gateway, not a 500.
            # The gateway is fine; its upstream isn't. Say which.
            return JSONResponse(
                {"detail": f"Upstream service unavailable: {str(e)[:100]}"},
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

    try:
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception:
        return JSONResponse({"detail": r.text[:300]}, status_code=r.status_code)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def ui():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    """Aggregated health — the gateway reports on its whole topology."""
    out = {"gateway": "ok"}
    async with httpx.AsyncClient(timeout=3) as c:
        for name, url in (("auth", AUTH_URL), ("agent", AGENT_URL)):
            try:
                r = await c.get(f"{url}/health")
                out[name] = "ok" if r.status_code == 200 else f"unhealthy ({r.status_code})"
            except httpx.HTTPError:
                out[name] = "unreachable"
    return out


@app.post("/api/login")
async def login(request: Request):
    return await proxy(request, AUTH_URL, "/login")


@app.post("/api/ask")
async def ask(request: Request):
    return await proxy(request, AGENT_URL, "/ask")


@app.get("/api/jobs")
async def jobs(request: Request):
    return await proxy(request, AGENT_URL, "/jobs")