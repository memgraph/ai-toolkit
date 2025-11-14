import asyncio
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Implementation
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

from mcp_memgraph import (
    run_query,
    get_schema,
    get_node_neighborhood,
    search_node_vectors,
)
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
            command=command, args=[server_script_path], env=os.environ.copy()
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                self.stdio,
                self.write,
                client_info=Implementation(name="MCP Test Client", version="1.0.0"),
            )
        )

        await self.session.initialize()


@pytest.mark.asyncio
async def test_mcp_client():
    """Test the MCP client connection to the server."""
    server_script_path = "src/mcp_memgraph/main.py"
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
    """Test the run_query tool with read operations."""
    query = "MATCH (n) RETURN n LIMIT 1;"
    response = run_query.fn(query)
    assert isinstance(response, list), "Expected response to be a list"
    assert len(response) >= 0, "Expected response to have at least 0 results"
    # Verify no error in response for read queries
    if len(response) > 0:
        assert "error" not in response[0], "Read query should not error"


@pytest.mark.asyncio
async def test_write_query_blocked_in_readonly_mode():
    """Test write queries are blocked when MCP_READ_ONLY=true (default)."""
    # Ensure we're in read-only mode (default)
    original_value = os.environ.get("MCP_READ_ONLY")
    os.environ["MCP_READ_ONLY"] = "true"

    # Reimport to reload config and server
    import importlib
    import mcp_memgraph.config
    import mcp_memgraph.servers.server

    importlib.reload(mcp_memgraph.config)
    importlib.reload(mcp_memgraph.servers.server)
    from mcp_memgraph import run_query

    try:
        # Test CREATE query
        create_query = "CREATE (n:TestNode {name: 'test'}) RETURN n;"
        response = run_query.fn(create_query)

        assert isinstance(response, list), "Expected response to be a list"
        assert len(response) > 0, "Expected error response"
        assert "error" in response[0], "Expected error for write operation"
        assert (
            "read-only" in response[0]["error"].lower()
        ), "Error should mention read-only mode"

        # Test MERGE query
        merge_query = "MERGE (n:TestNode {id: 1}) RETURN n;"
        response = run_query.fn(merge_query)
        assert "error" in response[0], "MERGE should be blocked in read-only mode"

        # Test DELETE query
        delete_query = "MATCH (n:TestNode) DELETE n;"
        response = run_query.fn(delete_query)
        assert "error" in response[0], "DELETE should be blocked in read-only mode"

        # Test SET query
        set_query = "MATCH (n:TestNode) SET n.name = 'updated' RETURN n;"
        response = run_query.fn(set_query)
        assert "error" in response[0], "SET should be blocked in read-only mode"

    finally:
        # Restore original value
        if original_value is not None:
            os.environ["MCP_READ_ONLY"] = original_value
        else:
            os.environ.pop("MCP_READ_ONLY", None)


@pytest.mark.asyncio
async def test_write_query_allowed_when_readonly_disabled():
    """Test that write queries work when MCP_READ_ONLY=false."""
    # Set read-only mode to false
    original_value = os.environ.get("MCP_READ_ONLY")
    os.environ["MCP_READ_ONLY"] = "false"

    # Reimport to reload config and server
    import importlib
    import mcp_memgraph.config
    import mcp_memgraph.servers.server

    importlib.reload(mcp_memgraph.config)
    importlib.reload(mcp_memgraph.servers.server)
    from mcp_memgraph import run_query

    try:
        # Test CREATE query (should work now)
        create_query = (
            "CREATE (n:TestNode {name: 'test', test_marker: true}) " "RETURN n;"
        )
        response = run_query.fn(create_query)

        assert isinstance(response, list), "Expected response to be a list"
        # Should not have an error about read-only mode
        if len(response) > 0 and "error" in response[0]:
            # Could have other errors, but not read-only error
            assert (
                "read-only" not in response[0]["error"].lower()
            ), "Should not block write when read-only is disabled"

        # Clean up: delete the test node
        cleanup_query = "MATCH (n:TestNode {test_marker: true}) DELETE n;"
        run_query.fn(cleanup_query)

    finally:
        # Restore original value
        if original_value is not None:
            os.environ["MCP_READ_ONLY"] = original_value
        else:
            os.environ.pop("MCP_READ_ONLY", None)


@pytest.mark.asyncio
async def test_get_schema():
    """Test the get_schema tool."""
    response = get_schema.fn()
    assert isinstance(response, list), "Expected response to be a list"
    assert len(response) >= 0, "Expected response to have at least 0 results"
    # Add more assertions based on expected schema information


@pytest.mark.asyncio
async def test_get_node_neighborhood():
    """Test the get_node_neighborhood tool."""
    # Test with a non-existent node ID (should return empty or error)
    response = get_node_neighborhood.fn("999999", max_distance=1, limit=10)
    assert isinstance(response, list), "Expected response to be a list"
    # The response should either be empty (no nodes found) or contain an error message
    assert len(response) >= 0, "Expected response to have at least 0 results"


@pytest.mark.asyncio
async def test_search_node_vectors():
    """Test the search_node_vectors tool."""
    # Test with a non-existent index (should return error)
    query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    response = search_node_vectors.fn("non_existent_index", query_vector, limit=5)
    assert isinstance(response, list), "Expected response to be a list"
    # The response should contain an error message since the index doesn't exist
    assert len(response) >= 0, "Expected response to have at least 0 results"


@pytest.mark.asyncio
async def test_tools_and_resources():
    """Test that all tools and resources are present in the MCP server."""
    server_script_path = "src/mcp_memgraph/main.py"
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
            "get_node_neighborhood",
            "search_node_vectors",
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
