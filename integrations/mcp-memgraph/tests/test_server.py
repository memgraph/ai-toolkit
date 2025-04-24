import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

from mcp_memgraph import run_query, get_schema
import pytest

pytestmark = pytest.mark.asyncio  # Mark all tests in this file as asyncio-compatible

load_dotenv()  # Load environment variables from .env


class MCPClient:
    """Client for connecting to an MCP server."""

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()


@pytest.mark.asyncio
async def test_mcp_client():
    """Test the MCP client connection to the server."""
    server_script_path = "integrations/mcp-memgraph/src/mcp_memgraph/server.py"
    client = MCPClient()
    try:
        await client.connect_to_server(server_script_path)
        response = await client.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        assert client.session is not None, "Session should be initialized"
        assert client.stdio is not None, "Stdio transport should be initialized"
        assert client.write is not None, "Write transport should be initialized"

    finally:
        await client.exit_stack.aclose()


@pytest.mark.asyncio
async def test_run_query():
    """Test the run_query tool."""
    query = "MATCH (n) RETURN n LIMIT 1;"
    response = run_query(query)
    assert isinstance(response, list), "Expected response to be a list"
    assert len(response) >= 0, "Expected response to have at least 0 results"
    # Add more assertions based on expected query results


@pytest.mark.asyncio
async def test_get_schema():
    """Test the get_schema tool."""
    response = get_schema()
    assert isinstance(response, list), "Expected response to be a list"
    assert len(response) >= 0, "Expected response to have at least 0 results"
    # Add more assertions based on expected schema information


@pytest.mark.asyncio
async def test_tools_and_resources():
    """Test that all tools and resources are present in the MCP server."""
    server_script_path = "integrations/mcp-memgraph/src/mcp_memgraph/server.py"
    client = MCPClient()
    try:
        await client.connect_to_server(server_script_path)

        # TODO(@antejavor): Add this dynamically.
        expected_tools = [
            "run_query",
            "get_configuration",
            "get_index",
            "get_constraint",
            "get_schema",
            "get_storage",
            "get_triggers",
            "get_betweenness_centrality",
            "get_page_rank",
        ]
        expected_resources = []

        response = await client.session.list_tools()
        available_tools = [tool.name for tool in response.tools]

        response = await client.session.list_resources()
        available_resources = [str(resource.uri) for resource in response.resources]

        assert len(available_tools) == len(
            expected_tools
        ), "Mismatch in number of tools"
        for tool in expected_tools:
            assert tool in available_tools, f"Tool '{tool}' is missing from the server"

        assert len(available_resources) == len(
            expected_resources
        ), "Mismatch in number of resources"
        for resource in expected_resources:
            assert (
                resource in available_resources
            ), f"Resource '{resource}' is missing from the server"

    finally:
        await client.exit_stack.aclose()
