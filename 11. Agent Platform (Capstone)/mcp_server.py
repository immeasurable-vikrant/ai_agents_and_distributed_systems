"""
mcp_server.py — a STANDALONE MCP server (stdio transport).

WHY MCP AT ALL: your agent's tools are currently locked inside your
FastAPI process. MCP is a protocol that exposes them so ANY client —
Claude Desktop, Claude Code, Cursor, a colleague's agent — can use them
without importing your code or knowing your framework.

THREE PRIMITIVES, and the distinction matters more than the code:

    TOOL      the MODEL decides to call it      "go search for X"
    RESOURCE  the CLIENT loads it by URI        "here's context, take it"
    PROMPT    the USER picks it (slash command) "here's a good way to use me"

Most tutorials only cover tools. Resources and prompts are what make a
server feel like a real integration rather than a function wrapper.

RUN:
    python mcp_server.py        (waits on stdin for protocol messages)

CONNECT CLAUDE DESKTOP — add to claude_desktop_config.json:
    {
      "mcpServers": {
        "agent-platform": {
          "command": "python",
          "args": ["mcp_server.py"],
          "cwd": "/absolute/path/to/project-11"
        }
      }
    }
"""

import json
import os

import httpx
from mcp.server.mcpserver import MCPServer

# NOTE: mcp 2.x renamed FastMCP → MCPServer. If you're following an older
# tutorial that imports `from mcp.server.fastmcp import FastMCP`, either
# update the import or pin `mcp<2`. The SDK is young and moving.

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
MCP_TOKEN = os.getenv("MCP_TOKEN", "")

mcp = MCPServer("agent-platform")


# ============================================================
# TOOLS — actions the model can invoke
# ============================================================
# These call your RUNNING SERVICES over HTTP rather than reimplementing
# anything. The MCP server is a thin protocol adapter, not a second copy
# of your business logic — that's what keeps the two from drifting apart.

@mcp.tool()
async def ask_agent(question: str) -> str:
    """Ask the multi-agent platform a question. A supervisor routes it to specialists."""
    if not MCP_TOKEN:
        return "ERROR: MCP_TOKEN not set. Log in and export the JWT first."
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(f"{GATEWAY_URL}/api/ask",
                         json={"question": question},
                         headers={"Authorization": f"Bearer {MCP_TOKEN}"})
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:200]}"
    d = r.json()
    return f"{d['answer']}\n\n(specialists used: {', '.join(d['specialists'])})"


@mcp.tool()
async def list_recent_jobs() -> str:
    """List recent questions asked on the platform by your organization."""
    if not MCP_TOKEN:
        return "ERROR: MCP_TOKEN not set."
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{GATEWAY_URL}/api/jobs",
                        headers={"Authorization": f"Bearer {MCP_TOKEN}"})
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:200]}"
    return json.dumps(r.json(), indent=2)


# ============================================================
# RESOURCES — context the client reads, no model decision needed
# ============================================================
# WHY THIS IS A RESOURCE AND NOT A TOOL: service health is reference
# context, not an action worth spending a tool-call round trip on. A
# client can attach it up front and let the model reason over it.

@mcp.resource("platform://health")
async def platform_health() -> str:
    """Current health of every service in the platform."""
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            r = await c.get(f"{GATEWAY_URL}/api/health")
            return json.dumps(r.json(), indent=2)
        except httpx.HTTPError as e:
            return json.dumps({"error": f"gateway unreachable: {e}"})


@mcp.resource("platform://architecture")
async def architecture() -> str:
    """How this platform is put together."""
    return json.dumps({
        "gateway": {"port": 8000, "role": "only public port; routing, rate limit, tracing"},
        "auth_service": {"port": 8001, "db": "auth.db", "signs": "RS256"},
        "agent_service": {"port": 8002, "db": "agent.db", "pattern": "supervisor + specialists"},
        "note": "database-per-service: no cross-service foreign keys",
    }, indent=2)


# ============================================================
# PROMPTS — reusable templates the USER picks
# ============================================================
# In Claude Desktop these appear as slash commands. The server is saying
# "here are good ways to use me" instead of making every client author
# reinvent the phrasing. Packaged expertise, shipped with the tools.

@mcp.prompt()
def deep_dive(topic: str) -> str:
    """Prompt template: thorough multi-specialist analysis of a topic."""
    return (f"Use ask_agent on '{topic}'. Then read platform://health to "
            f"confirm all services were up. Summarize the findings and "
            f"note which specialists were consulted and why.")


@mcp.prompt()
def review_activity() -> str:
    """Prompt template: review what the org has been asking."""
    return ("Call list_recent_jobs. Group the questions by theme, note "
            "any that look repetitive, and suggest what could be "
            "documented so people stop re-asking it.")


if __name__ == "__main__":
    # stdio transport: stdout IS the protocol channel. NEVER print() to
    # stdout in this process — it corrupts the message stream. Use stderr
    # for logging. (This is the #1 MCP debugging trap.)
    mcp.run()