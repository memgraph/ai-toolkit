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

from agent_context_graph.tools import Tool, ToolError, is_available, load_tools

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server

SERVER_NAME = "context-graph"


def build_server(tools: dict[str, Tool] | None = None) -> Server:
    """An MCP server exposing ``tools``, all registered tools by default.

    Each listing and call re-checks the config file, so a tool whose
    ``available`` check fails (e.g. ``memory`` before the user opts in) is
    hidden and refused. Harnesses list tools once per session, so turning a
    tool on shows up in the next session.

    Supports mcp 1.x, which registers handlers through decorators, and 2.x,
    which takes them as constructor arguments; the protocol is the same.
    """
    import mcp.types as types
    from anyio import to_thread
    from mcp.server.lowlevel import Server

    from agent_context_graph.adapters._identity import reload_config

    tools = load_tools() if tools is None else tools

    def current_config() -> Any:
        # The server lives as long as the harness session; re-read so
        # `setup --memory-backend` or `config set` take effect mid-session.
        return reload_config()

    async def list_tools() -> list[types.Tool]:
        config = current_config()
        return [
            types.Tool(name=tool.name, description=tool.description, inputSchema=tool.input_schema)
            for tool in tools.values()
            if is_available(tool, config)
        ]

    async def call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        config = current_config()
        tool = tools.get(name)
        if tool is None or not is_available(tool, config):
            return _error(f"Unknown tool: {name}")
        try:
            # Tools query Memgraph synchronously; a thread keeps the server responsive.
            result = await to_thread.run_sync(tool.call, arguments or {}, config)
        except ToolError as exc:
            return _error(str(exc))
        return types.CallToolResult(content=[types.TextContent(type="text", text=result.text)])

    if hasattr(Server, "call_tool"):  # mcp 1.x
        server: Server = Server(SERVER_NAME)
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
        return server

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=await list_tools())

    async def on_call_tool(ctx: Any, params: Any) -> types.CallToolResult:
        return await call_tool(params.name, params.arguments)

    return Server(SERVER_NAME, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


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
