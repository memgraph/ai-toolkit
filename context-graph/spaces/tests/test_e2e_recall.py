"""Recall across a login's spaces, against a real Enterprise Memgraph: personal and shared, fused."""

from __future__ import annotations

import pytest
from sessions_graph.tool import MemoryTool, RecallTool

from agent_context_graph.adapters import _identity
from context_graph_spaces.provision import apply
from context_graph_spaces.spec import parse_spec


@pytest.fixture
def team(admin):
    spec = parse_spec({"members": ["alice", "bob"], "spaces": {"payments": {"members": ["alice", "bob"]}}})
    return apply(spec, admin).passwords


def _as(enterprise_url, join, team, login):
    join(enterprise_url, login, team[login])
    return _identity.load_config()


def test_recall_reads_personal_and_shared_memory_and_nobody_elses(enterprise_url, team, join):
    bob = _as(enterprise_url, join, team, "bob")
    MemoryTool().call(
        {
            "command": "create",
            "path": "/memories/spaces/payments/db.md",
            "file_text": "The payments database is Postgres 16.",
        },
        bob,
    )
    MemoryTool().call(
        {
            "command": "create",
            "path": "/memories/bob-only.md",
            "file_text": "The payments database password is canary-7731.",
        },
        bob,
    )
    alice = _as(enterprise_url, join, team, "alice")
    MemoryTool().call(
        {"command": "create", "path": "/memories/alice.md", "file_text": "I review payments database migrations."},
        alice,
    )

    result = RecallTool().call({"question": "payments database"}, alice)

    assert "MEMORY [/memories/spaces/payments/db.md" in result.text
    assert "MEMORY [/memories/alice.md" in result.text
    assert "canary-7731" not in result.text
    assert "missing" not in result.text
    assert "Vector search is off" not in result.text


def test_a_space_that_cant_be_searched_is_named_and_the_rest_still_answers(enterprise_url, team, join, admin):
    alice = _as(enterprise_url, join, team, "alice")
    MemoryTool().call(
        {"command": "create", "path": "/memories/alice.md", "file_text": "I review payments database migrations."},
        alice,
    )
    admin("CREATE (:SpaceMembership {space: 'ghost', database: 'space_ghost'})", "u_alice", None)
    _identity.reload_config()

    result = RecallTool().call({"question": "payments database"}, _identity.load_config())

    assert "MEMORY [/memories/alice.md" in result.text
    assert "Not searched, so rows from them are missing: shared space ghost" in result.text
