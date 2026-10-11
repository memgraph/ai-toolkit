"""The memory tool in shared spaces, against a real Enterprise Memgraph: /memories/spaces/<space>/ is shared."""

from __future__ import annotations

import pytest
from sessions_graph.tool import SHARED_OWNER, MemoryTool

from agent_context_graph.adapters import _identity
from agent_context_graph.tools import ToolError
from context_graph_spaces.provision import apply
from context_graph_spaces.spec import parse_spec

DOC = "/memories/spaces/payments/db.md"


@pytest.fixture
def team(admin):
    """alice and bob share payments; carol is a member with no shared space."""
    spec = parse_spec({"members": ["alice", "bob", "carol"], "spaces": {"payments": {"members": ["alice", "bob"]}}})
    return apply(spec, admin).passwords


@pytest.fixture
def as_login(enterprise_url, team, join):
    """``as_login(login)(arguments)``: the memory tool as that login, with its own tool instance."""
    tools = {}

    def bind(login):
        def call(arguments):
            join(enterprise_url, login, team[login])
            tool = tools.setdefault(login, MemoryTool())
            return tool.call(arguments, _identity.load_config()).text

        return call

    return bind


def _shared_files(admin):
    query = "MATCH (m:Memory) RETURN m.path AS path, m.user_id AS owner, m.author AS author, m.updated_by AS by"
    return admin(query, "space_payments", None)


def test_members_read_and_edit_one_shared_tree(as_login, admin):
    alice, bob = as_login("alice"), as_login("bob")

    alice({"command": "create", "path": DOC, "file_text": "Payments moves to Postgres 16."})
    assert "Postgres 16" in bob({"command": "view", "path": DOC})
    bob({"command": "str_replace", "path": DOC, "old_str": "16", "new_str": "17"})

    assert _shared_files(admin) == [{"path": DOC, "owner": SHARED_OWNER, "author": "alice", "by": "bob"}]
    assert "Postgres 17" in alice({"command": "view", "path": DOC})
    versions = admin("MATCH (v:MemoryVersion) RETURN v.content AS c, v.replaced_by AS by", "space_payments", None)
    assert versions == [{"c": "Payments moves to Postgres 16.", "by": "bob"}]


def test_a_non_member_is_told_so(as_login):
    with pytest.raises(ToolError, match="aren't a member of a shared space named 'payments'"):
        as_login("carol")({"command": "view", "path": DOC})


def test_viewing_the_spaces_folder_lists_mine(as_login):
    assert "/memories/spaces/payments/" in as_login("alice")({"command": "view", "path": "/memories/spaces"})
    assert "No shared spaces" in as_login("carol")({"command": "view", "path": "/memories/spaces"})


def test_rename_promotes_a_personal_file_into_a_space_and_back(as_login, admin):
    alice, bob = as_login("alice"), as_login("bob")
    alice({"command": "create", "path": "/memories/runbook.md", "file_text": "Restart with make deploy."})

    assert "now in shared space payments" in alice(
        {"command": "rename", "old_path": "/memories/runbook.md", "new_path": "/memories/spaces/payments/runbook.md"}
    )
    assert admin("MATCH (m:Memory) RETURN m.path AS p", "u_alice", None) == []
    assert "make deploy" in bob({"command": "view", "path": "/memories/spaces/payments/runbook.md"})

    bob({"command": "rename", "old_path": "/memories/spaces/payments/runbook.md", "new_path": "/memories/mine.md"})
    assert admin("MATCH (m:Memory) RETURN m.path AS p", "u_bob", None) == [{"p": "/memories/mine.md"}]
    assert _shared_files(admin) == []


def test_a_move_never_overwrites(as_login):
    alice = as_login("alice")
    alice({"command": "create", "path": DOC, "file_text": "shared"})
    alice({"command": "create", "path": "/memories/db.md", "file_text": "personal"})

    with pytest.raises(ToolError, match="already exists"):
        alice({"command": "rename", "old_path": "/memories/db.md", "new_path": DOC})
    assert "personal" in alice({"command": "view", "path": "/memories/db.md"})


def test_deleting_a_reconciled_shared_file_leaves_its_chunks_to_the_worker(as_login, admin):
    alice = as_login("alice")
    alice({"command": "create", "path": DOC, "file_text": "Payments moves to Postgres 16."})
    admin(
        "MATCH (m:Memory) CREATE (m)-[:HAS_CHUNK]->(:Chunk {hash: 'h1', text: 'Postgres 16'})", "space_payments", None
    )

    alice({"command": "delete", "path": DOC})

    assert _shared_files(admin) == []
    assert admin("MATCH (c:Chunk) RETURN c.hash AS h", "space_payments", None) == [{"h": "h1"}]


def test_the_session_index_lists_shared_files(enterprise_url, team, join, as_login):
    as_login("bob")({"command": "create", "path": DOC, "file_text": "---\ndescription: payments database\n---\nPG"})

    join(enterprise_url, "alice", team["alice"])
    index = MemoryTool().session_context(_identity.load_config(), {"cwd": "/tmp"})

    assert index is not None
    assert "Shared space payments (/memories/spaces/payments/" in index
    assert f"- {DOC} — payments database" in index


def test_leaving_a_space_refuses_writes_once_the_connection_ages_out(enterprise_url, team, join, admin, monkeypatch):
    import time

    from context_graph_spaces import resolver

    monkeypatch.setattr(resolver, "REVOCATION_DELAY_SECONDS", 1.0)
    join(enterprise_url, "bob", team["bob"])
    tool = MemoryTool()
    tool.call({"command": "create", "path": DOC, "file_text": "before"}, _identity.load_config())

    apply(parse_spec({"members": ["alice", "bob", "carol"], "spaces": {"payments": {"members": ["alice"]}}}), admin)
    time.sleep(1.5)

    with pytest.raises(ToolError, match="Memgraph refused"):
        tool.call({"command": "create", "path": DOC, "file_text": "after"}, _identity.load_config())
    with pytest.raises(ToolError, match="aren't a member"):
        tool.call({"command": "view", "path": DOC}, _identity.load_config())
