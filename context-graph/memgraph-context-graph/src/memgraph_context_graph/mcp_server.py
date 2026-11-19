"""``memgraph-context-graph mcp``: a stdio MCP server for every registered tool.

The harness starts it (the Claude Code and Codex plugins bundle it), so like
the hooks it reads configuration from the config file alone. Needs the
``mcp`` extra: ``memgraph-context-graph[mcp]``.

A result carries the tool's text only, never ``structuredContent``: Claude
Code and Codex both show the model the structured JSON *instead of* the text
when a result has both, and the text is the form recall was benchmarked in.
Programs get the JSON from the CLI (``memgraph-context-graph recall --json``).

With ``session_context``, the server's ``instructions`` carry the text a
SessionStart hook shows elsewhere: the memory index and where the user left
off. Only for an agent whose session-start hooks can't reach the model (Grok
Build); an agent that also reads the hook output would get it twice.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from memgraph_context_graph import graph_version
from memgraph_context_graph.graph_version import GraphTooNewError
from memgraph_context_graph.tools import Tool, ToolError, is_available, load_tools, writes_graph

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server

SERVER_NAME = "context-graph"


def build_server(tools: dict[str, Tool] | None = None, *, instructions: str | None = None) -> Server:
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

    from memgraph_context_graph.adapters._identity import reload_config

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
            result = await to_thread.run_sync(_call_checked, tool, arguments or {}, config)
        except (ToolError, GraphTooNewError) as exc:
            return _error(str(exc))
        return types.CallToolResult(content=[types.TextContent(type="text", text=result.text)])

    if hasattr(Server, "call_tool"):  # mcp 1.x
        server: Server = Server(SERVER_NAME, instructions=instructions)
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
        return server

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=await list_tools())

    async def on_call_tool(ctx: Any, params: Any) -> types.CallToolResult:
        return await call_tool(params.name, params.arguments)

    return Server(SERVER_NAME, instructions=instructions, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def _call_checked(tool: Tool, arguments: dict[str, Any], config: Any) -> Any:
    """Call *tool* unless a newer Context Graph reshaped the graph it reads, or writes (``writes_graph``).

    The server outlives package upgrades, so it checks on each call rather than once at start; the check is cached.
    """
    rules = graph_version.check_configured()
    if not rules.may_read(tool.connector):
        raise rules.refuse_read(tool.connector)
    if writes_graph(tool) and not rules.may_write(tool.connector):
        raise rules.refuse_write(tool.connector)
    return tool.call(arguments, config)


def session_instructions(tools: dict[str, Tool]) -> str | None:
    """The session-start text for the directory the harness started this server in."""
    from memgraph_context_graph.hooks.runner import session_start_text

    return session_start_text(sorted({tool.connector for tool in tools.values()}), {"cwd": os.getcwd()})


def serve(*, session_context: bool = False) -> int:
    """Serve over stdio until the harness closes the stream; with *session_context*, send the session text too."""
    import anyio
    from mcp.server.stdio import stdio_server

    tools = load_tools()
    server = build_server(tools, instructions=session_instructions(tools) if session_context else None)

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(run)
    return 0


def _error(message: str) -> Any:
    import mcp.types as types

    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], isError=True)
