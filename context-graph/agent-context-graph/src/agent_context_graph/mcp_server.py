"""``agent-context-graph mcp``: a stdio MCP server for every registered tool.

The harness starts it (the Claude Code and Codex plugins bundle it), so like
the hooks it reads configuration from the config file alone. Needs the
``mcp`` extra: ``agent-context-graph[mcp]``.

A result carries the tool's text only, never ``structuredContent``: Claude
Code and Codex both show the model the structured JSON *instead of* the text
when a result has both, and the text is the form recall was benchmarked in.
Programs get the JSON from the CLI (``agent-context-graph recall --json``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_context_graph.tools import Tool, ToolError, load_tools

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server

SERVER_NAME = "context-graph"


def build_server(tools: dict[str, Tool] | None = None) -> Server:
    """An MCP server exposing ``tools``, all registered tools by default."""
    import mcp.types as types
    from anyio import to_thread
    from mcp.server.lowlevel import Server

    from agent_context_graph.adapters._identity import load_config

    tools = load_tools() if tools is None else tools
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=tool.name, description=tool.description, inputSchema=tool.input_schema)
            for tool in tools.values()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        tool = tools.get(name)
        if tool is None:
            return _error(f"Unknown tool: {name}")
        try:
            # Tools query Memgraph synchronously; a thread keeps the server responsive.
            result = await to_thread.run_sync(tool.call, arguments, load_config())
        except ToolError as exc:
            return _error(str(exc))
        return types.CallToolResult(content=[types.TextContent(type="text", text=result.text)])

    return server


def serve() -> int:
    """Serve over stdio until the harness closes the stream."""
    import anyio
    from mcp.server.stdio import stdio_server

    server = build_server()

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(run)
    return 0


def _error(message: str) -> Any:
    import mcp.types as types

    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], isError=True)
