"""The launch ship gate (map #501): two users and one shared space, isolated by Memgraph, end to end.

One scenario, the way two people would actually use it: each joins with their
own login, works through the memory tool and recall, and an admin changes the
spec. Every check is against a real Enterprise Memgraph.
"""

from __future__ import annotations

import time

import neo4j
import pytest
from sessions_graph.tool import MemoryTool, RecallTool

from context_graph_spaces import resolver
from context_graph_spaces.provision import apply
from context_graph_spaces.spec import parse_spec
from memgraph_context_graph.adapters import _identity
from memgraph_context_graph.tools import ToolError

CANARY = "canary-ante-4417"
SHARED = "/memories/spaces/launch/decision.md"


def test_two_users_one_shared_space(enterprise_url, admin, join, connect, monkeypatch, warmed_embeddings):
    monkeypatch.setattr(resolver, "REVOCATION_DELAY_SECONDS", 1.0)
    spec = parse_spec({"members": ["alice", "bob"], "spaces": {"launch": {"members": ["alice", "bob"]}}})
    passwords = apply(spec, admin).passwords

    def as_login(login):
        join(enterprise_url, login, passwords[login])
        return _identity.load_config()

    alice = as_login("alice")
    MemoryTool().call(
        {"command": "create", "path": "/memories/private.md", "file_text": f"The launch password is {CANARY}."}, alice
    )
    MemoryTool().call(
        {"command": "create", "path": "/memories/decision.md", "file_text": "We launch on Tuesday."}, alice
    )

    # 1. B can't open A's personal database.
    with pytest.raises(neo4j.exceptions.ClientError, match="not authorized"):
        connect("bob", passwords["bob"], "MATCH (n) RETURN n", database="u_alice")

    # 2. A promotes a memory into the space; B's recall returns it, labelled by its shared path.
    MemoryTool().call({"command": "rename", "old_path": "/memories/decision.md", "new_path": SHARED}, alice)
    bob = as_login("bob")
    recalled = RecallTool().call({"question": "When do we launch?"}, bob).text
    assert f"MEMORY [{SHARED}" in recalled
    assert "Tuesday" in recalled

    # 3. B's recall never returns A's personal content.
    for question in ("What is the launch password?", "launch", "password"):
        assert CANARY not in RecallTool().call({"question": question}, bob).text

    # 4. Nobody can write collection-tier data into the shared database.
    with pytest.raises(neo4j.exceptions.ClientError, match="CREATE permission"):
        connect("bob", passwords["bob"], "CREATE (:Action {text: 'leak'})", database="space_launch")

    # 5. Removing B from the spec cuts B off from the space on the next connection.
    tool = MemoryTool()
    tool.call({"command": "view", "path": SHARED}, bob)
    apply(parse_spec({"members": ["alice", "bob"], "spaces": {"launch": {"members": ["alice"]}}}), admin)
    with pytest.raises(neo4j.exceptions.ClientError, match="not authorized"):
        connect("bob", passwords["bob"], "MATCH (n) RETURN n", database="space_launch")
    time.sleep(1.5)  # The MCP server's pooled connection ages out (REVOCATION_DELAY_SECONDS, patched to 1s).
    with pytest.raises(ToolError):
        tool.call({"command": "view", "path": SHARED}, bob)
