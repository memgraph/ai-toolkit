"""Tests for the list_databases / use_database tools and SessionAuth lifecycle.

These tests do not require a running Memgraph or Keycloak: they exercise the
tools by setting a :class:`SessionAuth` on the request contextvar directly
(simulating what :class:`AuthMiddleware` would do for an authenticated
request).
"""

from __future__ import annotations

import importlib
import os

import pytest


def _reload_with_env(env: dict[str, str]):
    """Set env vars and reload the modules whose behavior depends on them.

    Returns the freshly-reloaded ``servers.server`` and ``auth`` modules so the
    test interacts with up-to-date singletons.
    """
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    import mcp_memgraph.auth
    import mcp_memgraph.config
    import mcp_memgraph.servers.server
    import mcp_memgraph.tenant_routing

    importlib.reload(mcp_memgraph.config)
    importlib.reload(mcp_memgraph.auth)
    # Drop the cached registry so it picks up the new catalog.
    mcp_memgraph.tenant_routing.reset_registry_for_tests()
    importlib.reload(mcp_memgraph.tenant_routing)
    importlib.reload(mcp_memgraph.servers.server)

    return mcp_memgraph.servers.server, mcp_memgraph.auth


@pytest.fixture
def alice_session(monkeypatch):
    """Auth-enabled environment + a SessionAuth representing alice on tenant-a."""
    monkeypatch.setenv("MCP_AUTH_ENABLED", "true")
    monkeypatch.setenv("MCP_AUTH_ISSUER", "https://example.test/realms/memgraph")
    monkeypatch.setenv("MCP_AUTH_AUDIENCE", "https://mcp.example.test")
    monkeypatch.setenv("MCP_AUTH_JWKS_URL", "https://example.test/jwks.json")
    monkeypatch.setenv("MCP_TENANT_CATALOG", "tenant-a,tenant-b,tenant-c")

    server_mod, auth_mod = _reload_with_env({})

    alice = auth_mod.SessionAuth(
        allowed_tenants=frozenset({"tenant-a", "tenant-c"}),
        current_tenant="tenant-a",
        subject="alice-uuid",
    )
    cv_token = auth_mod.set_current_session_auth(alice)
    try:
        yield server_mod, auth_mod, alice
    finally:
        auth_mod.reset_current_session_auth(cv_token)


# ---------------------------------------------------------------------------
# Auth-off behaviour (regression for stdio / pre-auth usage)
# ---------------------------------------------------------------------------


def test_list_databases_without_auth_returns_single_legacy_entry(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_ENABLED", "false")
    monkeypatch.delenv("MCP_TENANT_CATALOG", raising=False)
    server_mod, _ = _reload_with_env({})

    result = server_mod.list_databases()

    assert len(result) == 1
    assert result[0]["current"] is True


def test_use_database_without_auth_rejects(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_ENABLED", "false")
    monkeypatch.delenv("MCP_TENANT_CATALOG", raising=False)
    server_mod, _ = _reload_with_env({})

    result = server_mod.use_database("anything")

    assert "error" in result
    assert "auth disabled" in result["error"]


# ---------------------------------------------------------------------------
# Auth-on behaviour
# ---------------------------------------------------------------------------


def test_alice_list_databases_shows_only_her_tenants(alice_session):
    server_mod, _, _ = alice_session

    result = server_mod.list_databases()

    names = sorted(entry["name"] for entry in result)
    assert names == ["tenant-a", "tenant-c"]  # tenant-b is in catalog but not in JWT
    current = [entry["name"] for entry in result if entry["current"]]
    assert current == ["tenant-a"]


def test_alice_can_switch_to_tenant_c(alice_session):
    server_mod, _, alice = alice_session

    response = server_mod.use_database("tenant-c")

    assert response == {"status": "ok", "current": "tenant-c"}
    assert alice.current_tenant == "tenant-c"

    # list_databases should now report tenant-c as the current one.
    listing = server_mod.list_databases()
    current = [entry["name"] for entry in listing if entry["current"]]
    assert current == ["tenant-c"]


def test_alice_cannot_switch_to_tenant_b_in_catalog_but_not_in_token(alice_session):
    server_mod, _, alice = alice_session

    response = server_mod.use_database("tenant-b")

    assert "error" in response
    assert response["error"] == "database not available"
    assert response["requested"] == "tenant-b"
    assert response["allowed"] == ["tenant-a", "tenant-c"]
    # Current pointer is unchanged.
    assert alice.current_tenant == "tenant-a"


def test_use_database_nonexistent_returns_same_shape_no_enumeration(alice_session):
    """The "in catalog, not in token" and "not in catalog at all" cases must
    produce indistinguishable responses so callers can't enumerate the catalog.
    """
    server_mod, _, _ = alice_session

    in_catalog = server_mod.use_database("tenant-b")  # exists, not authorized
    not_in_catalog = server_mod.use_database("tenant-completely-fake")  # doesn't exist

    assert in_catalog.keys() == not_in_catalog.keys()
    assert in_catalog["error"] == not_in_catalog["error"]
    assert in_catalog["allowed"] == not_in_catalog["allowed"]


def test_parallel_sessions_have_independent_current_tenant(monkeypatch):
    """Two sessions for the same user must hold independent current_tenant pointers."""
    monkeypatch.setenv("MCP_AUTH_ENABLED", "true")
    monkeypatch.setenv("MCP_AUTH_ISSUER", "https://example.test/realms/memgraph")
    monkeypatch.setenv("MCP_AUTH_AUDIENCE", "https://mcp.example.test")
    monkeypatch.setenv("MCP_AUTH_JWKS_URL", "https://example.test/jwks.json")
    monkeypatch.setenv("MCP_TENANT_CATALOG", "tenant-a,tenant-c")
    server_mod, auth_mod = _reload_with_env({})

    sess_one = auth_mod.SessionAuth(
        allowed_tenants=frozenset({"tenant-a", "tenant-c"}),
        current_tenant="tenant-a",
    )
    sess_two = auth_mod.SessionAuth(
        allowed_tenants=frozenset({"tenant-a", "tenant-c"}),
        current_tenant="tenant-a",
    )

    # Session one switches to tenant-c.
    t = auth_mod.set_current_session_auth(sess_one)
    try:
        server_mod.use_database("tenant-c")
        assert sess_one.current_tenant == "tenant-c"
    finally:
        auth_mod.reset_current_session_auth(t)

    # Session two should still be on tenant-a — the contextvar reset, the
    # other SessionAuth object is independent.
    assert sess_two.current_tenant == "tenant-a"

    t = auth_mod.set_current_session_auth(sess_two)
    try:
        listing = server_mod.list_databases()
        current = [entry["name"] for entry in listing if entry["current"]]
        assert current == ["tenant-a"]
    finally:
        auth_mod.reset_current_session_auth(t)
