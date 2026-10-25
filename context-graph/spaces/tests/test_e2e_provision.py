"""Provisioning against a real Enterprise Memgraph: plan, apply, and what each login can then do."""

from __future__ import annotations

import neo4j
import pytest

from context_graph_spaces.provision import SERVICE_ACCOUNT, apply, live_databases, observed_state, plan
from context_graph_spaces.spec import parse_spec

DENIED = neo4j.exceptions.ClientError


def _spec(members=("alice", "bob"), spaces=None, **extra):
    data = {"members": list(members), **extra}
    if spaces is not None:
        data["spaces"] = spaces
    return parse_spec(data)


def _plan(admin, spec, **kwargs):
    return plan(spec, observed_state(admin, spec.members), **kwargs)


@pytest.fixture
def team(admin):
    """alice and bob, both in the payments space; returns (spec, passwords)."""
    spec = _spec(spaces={"payments": {"members": ["alice", "bob"]}})
    applied = apply(spec, admin)
    return spec, applied.passwords


def test_apply_is_idempotent(admin, team):
    spec, _ = team

    assert _plan(admin, spec).empty
    assert apply(spec, admin).steps == []


def test_each_member_lands_in_their_own_personal_space(team, connect):
    _, passwords = team
    alice, bob = ("alice", passwords["alice"]), ("bob", passwords["bob"])

    assert connect(*alice, "SHOW CURRENT DATABASE") == [{"Current": "u_alice"}]
    connect(*alice, "CREATE (:Action {text: 'alice-canary'})")

    assert connect(*alice, "MATCH (a:Action) RETURN a.text AS t") == [{"t": "alice-canary"}]
    assert connect(*bob, "MATCH (a:Action) RETURN a.text AS t") == []
    with pytest.raises(DENIED, match="not authorized"):
        connect(*bob, "MATCH (n) RETURN n", database="u_alice")


def test_members_share_curated_memory_but_nothing_else(team, connect):
    _, passwords = team
    alice, bob = ("alice", passwords["alice"]), ("bob", passwords["bob"])

    connect(*alice, "CREATE (:Memory {path: '/memories/spaces/payments/db.md'})", database="space_payments")
    assert connect(*bob, "MATCH (m:Memory) RETURN m.path AS p", database="space_payments") == [
        {"p": "/memories/spaces/payments/db.md"}
    ]
    connect(*bob, "MATCH (m:Memory) SET m.content = 'Postgres 16'", database="space_payments")
    with pytest.raises(DENIED, match="CREATE permission"):
        connect(*alice, "CREATE (:Action {text: 'leak'})", database="space_payments")


def test_a_member_knows_their_spaces_from_their_own_database(team, connect):
    _, passwords = team

    rows = connect("alice", passwords["alice"], "MATCH (m:SpaceMembership) RETURN m.space AS s, m.database AS d")

    assert rows == [{"s": "payments", "d": "space_payments"}]


def test_the_service_account_writes_what_members_cant(admin, team):
    roles = admin(f"SHOW ROLE FOR {SERVICE_ACCOUNT} ON DATABASE space_payments", None, None)

    assert [row["role"] for row in roles] == [
        name for name in observed_state(admin).roles if name.startswith("cg_worker")
    ]


def test_leaving_a_space_takes_effect_on_the_next_connection(admin, team, connect):
    _, passwords = team
    smaller = _spec(spaces={"payments": {"members": ["alice"]}})

    summaries = [step.summary for step in _plan(admin, smaller).steps]
    apply(smaller, admin)

    assert any(summary.startswith("take space_payments_member") and "from bob" in summary for summary in summaries)
    with pytest.raises(DENIED, match="not authorized"):
        connect("bob", passwords["bob"], "MATCH (n) RETURN n", database="space_payments")
    assert connect("bob", passwords["bob"], "MATCH (m:SpaceMembership) RETURN m") == []
    assert _plan(admin, smaller).empty


def test_widening_a_space_moves_members_to_a_new_role_without_a_gap(admin, team, connect):
    _, passwords = team
    before = {name for name in observed_state(admin).roles if name.startswith("space_payments_member")}
    wider = _spec(spaces={"payments": {"members": ["alice", "bob"], "labels": ["Memory", "Note"]}})

    apply(wider, admin)

    after = {name for name in observed_state(admin).roles if name.startswith("space_payments_member")}
    assert len(after) == 1 and after != before
    connect("bob", passwords["bob"], "CREATE (:Note {text: 'shared now'})", database="space_payments")
    assert _plan(admin, wider).empty


def test_removing_a_member_revokes_access_and_keeps_data_until_prune(admin, team, connect):
    _, passwords = team
    alice_only = _spec(members=("alice",))

    the_plan = _plan(admin, alice_only)
    apply(alice_only, admin)

    assert "user bob" in the_plan.kept and "database u_bob" in the_plan.kept
    with pytest.raises(neo4j.exceptions.Neo4jError):
        connect("bob", passwords["bob"], "MATCH (n) RETURN n")
    assert any(row["Name"] == "u_bob" for row in admin("SHOW DATABASES", None, None))

    apply(alice_only, admin, prune=True)

    # FORCE drops remain listed as DROPPING while existing connections drain.
    assert not {"u_bob", "space_payments"} & live_databases(admin)
    assert "bob" not in {row["user"] for row in admin("SHOW USERS", None, None)}
    assert _plan(admin, alice_only, prune=True).empty


def test_objects_the_admin_made_are_never_touched(admin):
    admin("CREATE ROLE IF NOT EXISTS analyst", None, None)
    admin("CREATE DATABASE reports", None, None)
    try:
        spec = _spec()
        apply(spec, admin)
        apply(_spec(members=()), admin, prune=True)

        assert "analyst" in {row["role"] for row in admin("SHOW ROLES", None, None)}
        assert "reports" in {row["Name"] for row in admin("SHOW DATABASES", None, None)}
    finally:
        admin("DROP ROLE analyst", None, None)
        admin("DROP DATABASE reports FORCE", None, None)


def test_memory_limits_follow_the_spec(admin):
    limited = _spec(personal_memory_limit_mb=256, spaces={"payments": {"members": ["alice"], "memory_limit_mb": 512}})

    apply(limited, admin)

    profiles = {row["profile"]: row for row in admin("SHOW TENANT PROFILES", None, None)}
    assert set(profiles) >= {"cg_256mb", "cg_512mb"}
    assert {name.strip() for name in profiles["cg_256mb"]["databases"].split(",")} == {"u_alice", "u_bob"}
    assert _plan(admin, limited).empty

    unlimited = _spec(spaces={"payments": {"members": ["alice"]}})
    apply(unlimited, admin)

    assert not any(row["profile"].startswith("cg_") for row in admin("SHOW TENANT PROFILES", None, None))
    assert _plan(admin, unlimited).empty
