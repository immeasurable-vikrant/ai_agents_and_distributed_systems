"""
mcp_client.py — the OTHER half of MCP: being a CONSUMER.

mcp_server.py made you a provider. This makes you a client. Being able
to do both is what "understands MCP" actually means — most people only
ever write servers.

WHAT THIS DEMONSTRATES:
  1. Connecting to a server as a separate PROCESS over stdio
  2. DISCOVERING its capabilities at runtime — you don't hardcode what
     it offers, you ask it
  3. Calling a tool, reading a resource, listing prompts
  4. Connecting to a server you did NOT write

RUN:
    python mcp_client.py
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def inspect_our_server():
    """
    Connect to OUR server as a client.

    WHY BOTHER when we wrote it: importing the module only proves the
    Python works. This proves it speaks valid MCP over a real transport —
    handshake, capability negotiation, the lot. If this succeeds, Claude
    Desktop will work too.
    """
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # HANDSHAKE — both sides declare what they support.
            # Neither assumes; that's what makes the protocol portable.
            await session.initialize()
            print("✅ connected to our agent-platform server\n")

            tools = await session.list_tools()
            print(f"TOOLS ({len(tools.tools)}):")
            for t in tools.tools:
                print(f"   {t.name} — {(t.description or '')[:62]}")

            resources = await session.list_resources()
            print(f"\nRESOURCES ({len(resources.resources)}):")
            for r in resources.resources:
                print(f"   {r.uri}")

            prompts = await session.list_prompts()
            print(f"\nPROMPTS ({len(prompts.prompts)}):")
            for p in prompts.prompts:
                print(f"   /{p.name} — {(p.description or '')[:56]}")

            # Read a RESOURCE (not a tool call — no model decision involved)
            print("\nREADING platform://architecture:")
            content = await session.read_resource("platform://architecture")
            for c in content.contents:
                if hasattr(c, "text"):
                    print("   " + c.text[:300].replace("\n", "\n   "))


async def consume_external_server(path: str = "/tmp"):
    """
    Connect to a server we did NOT write — the official filesystem server.

    THE POINT: the discovery and calling code below is IDENTICAL to the
    code above. Different vendor, different language even (this one is
    Node), same protocol. That interoperability is the entire reason MCP
    exists — otherwise every tool integration is bespoke glue.

    Requires Node: npx -y @modelcontextprotocol/server-filesystem
    """
    params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", path],
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"✅ connected to the official filesystem server ({path})")
                print(f"   it offers {len(tools.tools)} tools: "
                      f"{', '.join(t.name for t in tools.tools[:6])}")

                result = await session.call_tool("list_directory", {"path": path})
                for block in result.content:
                    if hasattr(block, "text"):
                        print("   " + block.text[:200].replace("\n", "\n   "))
    except Exception as e:
        print(f"⚠️  skipped (needs Node/npx): {str(e)[:110]}")


if __name__ == "__main__":
    print("=" * 60)
    print("1. CONSUMING OUR OWN SERVER (protocol round-trip)")
    print("=" * 60)
    asyncio.run(inspect_our_server())

    print("\n" + "=" * 60)
    print("2. CONSUMING A SERVER WE DIDN'T WRITE")
    print("=" * 60)
    asyncio.run(consume_external_server())