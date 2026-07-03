"""Tests for MCP tool readOnlyHint annotations."""

import asyncio
import os

import pytest


READ_ONLY_TOOLS = {
    "list_databases",
    "get_configuration",
    "get_index",
    "get_constraint",
    "get_schema",
    "get_storage",
    "get_triggers",
    "get_procedures",
    "get_betweenness_centrality",
    "get_page_rank",
    "get_node_neighborhood",
    "search_node_vectors",
}

# Tools that must NOT be advertised as read-only
NOT_READ_ONLY_TOOLS = {"use_database"}


def _get_tool_annotations_map():
    """Import the MCP server and return {tool_name: ToolAnnotations} synchronously."""
    import importlib

    import mcp_memgraph.config
    import mcp_memgraph.servers.server

    importlib.reload(mcp_memgraph.config)
    importlib.reload(mcp_memgraph.servers.server)

    from mcp_memgraph.servers.server import mcp

    tools = asyncio.run(mcp.list_tools())
    return {t.name: t.annotations for t in tools}


@pytest.fixture(scope="module")
def tool_annotations_readonly_mode():
    """Return tool annotations when MCP_READ_ONLY=true."""
    original = os.environ.get("MCP_READ_ONLY")
    os.environ["MCP_READ_ONLY"] = "true"
    try:
        return _get_tool_annotations_map()
    finally:
        if original is None:
            os.environ.pop("MCP_READ_ONLY", None)
        else:
            os.environ["MCP_READ_ONLY"] = original


@pytest.fixture(scope="module")
def tool_annotations_readwrite_mode():
    """Return tool annotations when MCP_READ_ONLY=false."""
    original = os.environ.get("MCP_READ_ONLY")
    os.environ["MCP_READ_ONLY"] = "false"
    try:
        return _get_tool_annotations_map()
    finally:
        if original is None:
            os.environ.pop("MCP_READ_ONLY", None)
        else:
            os.environ["MCP_READ_ONLY"] = original


@pytest.mark.parametrize("tool_name", sorted(READ_ONLY_TOOLS))
def test_read_only_tools_have_readonly_hint(tool_annotations_readonly_mode, tool_name):
    """Definitively read-only tools must advertise readOnlyHint=True."""
    annotations = tool_annotations_readonly_mode.get(tool_name)
    assert annotations is not None, f"Tool '{tool_name}' has no annotations"
    assert annotations.readOnlyHint is True, (
        f"Tool '{tool_name}' should have readOnlyHint=True, got {annotations.readOnlyHint}"
    )


@pytest.mark.parametrize("tool_name", sorted(NOT_READ_ONLY_TOOLS))
def test_non_readonly_tools_do_not_have_readonly_hint(tool_annotations_readonly_mode, tool_name):
    """Tools that mutate session state must not be advertised as read-only."""
    annotations = tool_annotations_readonly_mode.get(tool_name)
    read_only_hint = annotations.readOnlyHint if annotations is not None else None
    assert read_only_hint is not True, (
        f"Tool '{tool_name}' must not have readOnlyHint=True (it mutates session state)"
    )


def test_run_query_is_readonly_when_mode_is_readonly(tool_annotations_readonly_mode):
    """run_query must advertise readOnlyHint=True when MCP_READ_ONLY=true."""
    annotations = tool_annotations_readonly_mode.get("run_query")
    assert annotations is not None, "run_query has no annotations in read-only mode"
    assert annotations.readOnlyHint is True, (
        f"run_query should have readOnlyHint=True in read-only mode, got {annotations.readOnlyHint}"
    )


def test_run_query_is_not_readonly_when_writes_allowed(tool_annotations_readwrite_mode):
    """run_query must not advertise readOnlyHint=True when MCP_READ_ONLY=false."""
    annotations = tool_annotations_readwrite_mode.get("run_query")
    read_only_hint = annotations.readOnlyHint if annotations is not None else None
    assert read_only_hint is not True, (
        f"run_query should not have readOnlyHint=True when writes are allowed, got {read_only_hint}"
    )
