"""Shared spaces through the MCP server a harness starts: ``memgraph-context-graph mcp`` over stdio, one per member.

Each member joins with a config file of their own, and their server is a real subprocess
reading only that file, the way Claude Code or Codex runs it. So this covers what the
in-process tool tests can't: the installed entry point and its MCP dependency, the tool
list a harness sees, and that two members' servers share the space and nothing else.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from context_graph_spaces.provision import apply
from context_graph_spaces.spec import parse_spec
from memgraph_context_graph import cli, spaces
from memgraph_context_graph.adapters import _identity

SHARED = "/memories/spaces/payments/deploys.md"
PRIVATE = "/memories/alice-only.md"


@pytest.fixture
def team(admin):
    """alice and bob share payments; carol is a member with no shared space."""
    spec = parse_spec({"members": ["alice", "bob", "carol"], "spaces": {"payments": {"members": ["alice", "bob"]}}})
    return apply(spec, admin).passwords


@pytest.fixture
def member(enterprise_url, team, tmp_path, monkeypatch):
    """``member(login, graph_memory=True)``: join as *login* into a config file of its own, return the file."""

    def make(login: str, *, graph_memory: bool = True) -> Path:
        path = tmp_path / f"{login}.toml"
        monkeypatch.setenv(_identity.CONFIG_PATH_ENV, str(path))
        _identity._reset_cache()
        spaces._reset_cache()
        assert cli.main(["join", enterprise_url, login, f"--password={team[login]}"]) == 0
        if graph_memory:
            _identity.write_config(memory_backend=_identity.MEMORY_BACKEND_GRAPH)
        return path

    return make


def harness(config: Path, *calls: tuple[str, dict[str, Any]]) -> tuple[list[str], list[tuple[bool, str]]]:
    """Start a member's MCP server over stdio, list its tools, then make *calls*: ``(is_error, text)`` for each."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(
        command=str(Path(sys.executable).with_name("memgraph-context-graph")),
        args=["mcp"],
        env={**os.environ, _identity.CONFIG_PATH_ENV: str(config)},
    )

    async def run() -> tuple[list[str], list[tuple[bool, str]]]:
        async with stdio_client(server) as streams, ClientSession(*streams) as client:
            await client.initialize()
            # By alias: mcp 2.x renamed the Python fields, not the JSON a harness reads.
            listed = (await client.list_tools()).model_dump(by_alias=True)
            results = []
            for name, arguments in calls:
                result = (await client.call_tool(name, arguments)).model_dump(by_alias=True)
                results.append((bool(result.get("isError")), "".join(c.get("text", "") for c in result["content"])))
            return sorted(tool["name"] for tool in listed["tools"]), results

    return asyncio.run(run())


def memory(**arguments: Any) -> tuple[str, dict[str, Any]]:
    return ("memory", arguments)


def test_the_memory_tool_is_served_only_once_context_graph_is_the_memory(member):
    native, _ = harness(member("alice", graph_memory=False))
    graph, _ = harness(member("alice"))

    # Other installed connectors may serve tools of their own (resources-graph's resource), so only these two are asserted.
    assert "recall" in native and "memory" not in native
    assert {"memory", "recall"} <= set(graph)


def test_members_share_a_space_through_their_own_servers_and_nothing_else(member):
    alice, bob, carol = member("alice"), member("bob"), member("carol")

    harness(
        alice,
        memory(command="create", path=SHARED, file_text="Deploys go out on Thursdays."),
        memory(command="create", path=PRIVATE, file_text="Only alice knows this."),
    )
    _, bob_sees = harness(
        bob,
        memory(command="view", path=SHARED),
        memory(command="str_replace", path=SHARED, old_str="Thursdays", new_str="Wednesdays"),
        memory(command="view", path=PRIVATE),
        memory(command="view", path="/memories"),
    )
    _, alice_sees = harness(alice, memory(command="view", path=SHARED))
    _, carol_sees = harness(carol, memory(command="view", path=SHARED))

    (_, shared), (edit_error, _), (_, private), (_, root) = bob_sees
    assert "Thursdays" in shared and not edit_error
    assert "does not exist" in private
    assert "/memories/spaces/payments/" in root
    assert "Wednesdays" in alice_sees[0][1]
    assert carol_sees[0][0] and "aren't a member of a shared space named 'payments'" in carol_sees[0][1]


def test_recall_over_mcp_finds_the_shared_file_but_not_another_members_own(member, warmed_embeddings):
    alice, bob = member("alice"), member("bob")
    harness(
        alice,
        memory(command="create", path=SHARED, file_text="Deploys go out on Thursdays after the release review."),
        memory(command="create", path=PRIVATE, file_text="Alice's deploy key rotates on Thursdays."),
    )

    _, [(failed, rows)] = harness(bob, ("recall", {"question": "when do deploys go out?"}))

    assert not failed
    assert "after the release review" in rows
    assert "deploy key" not in rows
