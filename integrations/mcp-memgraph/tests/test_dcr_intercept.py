"""Tests for the DCR intercept that hands back a pre-registered client_id."""

from __future__ import annotations

import json

import pytest

from mcp_memgraph.config import MCPAuthConfig
from mcp_memgraph.middleware import AuthMiddleware

# ---------------------------------------------------------------------------
# Minimal ASGI test harness
# ---------------------------------------------------------------------------


def _build_scope(*, method: str, path: str, host: str = "localhost:8000", headers=None) -> dict:
    base = [(b"host", host.encode())]
    for name, value in (headers or {}).items():
        base.append((name.encode(), value.encode()))
    return {
        "type": "http",
        "method": method,
        "path": path,
        "scheme": "http",
        "headers": base,
    }


class _Recorder:
    """Collects ASGI ``send`` calls; exposes status + body once finished."""

    def __init__(self) -> None:
        self.status: int | None = None
        self.headers: list[tuple[bytes, bytes]] = []
        self._chunks: list[bytes] = []

    async def __call__(self, message: dict) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = message["headers"]
        elif message["type"] == "http.response.body":
            self._chunks.append(message.get("body", b""))

    @property
    def body(self) -> bytes:
        return b"".join(self._chunks)

    def json(self):
        return json.loads(self.body)


async def _no_receive():
    # No request body expected for these paths.
    return {"type": "http.request", "body": b"", "more_body": False}


def _make_receive_for(body: bytes):
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _inner_app_not_called(*_a, **_k):  # pragma: no cover - defensive
    raise AssertionError("inner app should not be called for discovery / DCR paths")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _enable_auth(monkeypatch, *, static_client_id: str | None = None) -> MCPAuthConfig:
    monkeypatch.setenv("MCP_AUTH_ENABLED", "true")
    monkeypatch.setenv("MCP_AUTH_ISSUER", "https://example.test/realms/memgraph")
    monkeypatch.setenv("MCP_AUTH_AUDIENCE", "https://mcp.example.test")
    monkeypatch.setenv("MCP_AUTH_JWKS_URL", "https://example.test/jwks.json")
    monkeypatch.setenv("MCP_TENANT_CATALOG", "tenant-a,tenant-b")
    if static_client_id is None:
        monkeypatch.delenv("MCP_AUTH_STATIC_CLIENT_ID", raising=False)
    else:
        monkeypatch.setenv("MCP_AUTH_STATIC_CLIENT_ID", static_client_id)
    return MCPAuthConfig()


def _stub_oidc_fetch(mw: AuthMiddleware, doc: dict) -> None:
    """Pre-fill the middleware's OIDC cache so it doesn't try to reach Keycloak."""
    mw._oidc_doc = doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oidc_doc_rewrites_registration_endpoint_when_intercept_on(monkeypatch):
    cfg = _enable_auth(monkeypatch, static_client_id="mcp-memgraph")
    mw = AuthMiddleware(_inner_app_not_called, cfg)
    _stub_oidc_fetch(
        mw,
        {
            "issuer": cfg.issuer,
            "authorization_endpoint": cfg.issuer + "/protocol/openid-connect/auth",
            "token_endpoint": cfg.issuer + "/protocol/openid-connect/token",
            "registration_endpoint": cfg.issuer + "/clients-registrations/openid-connect",
            "mtls_endpoint_aliases": {
                "registration_endpoint": cfg.issuer + "/clients-registrations/openid-connect",
                "token_endpoint": cfg.issuer + "/protocol/openid-connect/token",
            },
        },
    )

    rec = _Recorder()
    await mw(
        _build_scope(method="GET", path="/.well-known/openid-configuration"),
        _no_receive,
        rec,
    )

    assert rec.status == 200
    doc = rec.json()
    assert doc["registration_endpoint"] == "http://localhost:8000/register"
    assert doc["mtls_endpoint_aliases"]["registration_endpoint"] == "http://localhost:8000/register"
    # Other endpoints are NOT rewritten — only registration.
    assert doc["token_endpoint"] == cfg.issuer + "/protocol/openid-connect/token"


@pytest.mark.asyncio
async def test_oidc_doc_passthrough_when_intercept_off(monkeypatch):
    cfg = _enable_auth(monkeypatch, static_client_id=None)
    mw = AuthMiddleware(_inner_app_not_called, cfg)
    upstream = {
        "issuer": cfg.issuer,
        "registration_endpoint": cfg.issuer + "/clients-registrations/openid-connect",
    }
    _stub_oidc_fetch(mw, upstream)

    rec = _Recorder()
    await mw(
        _build_scope(method="GET", path="/.well-known/openid-configuration"),
        _no_receive,
        rec,
    )

    assert rec.status == 200
    doc = rec.json()
    # Unmodified — registration still goes to Keycloak.
    assert doc["registration_endpoint"].endswith("/clients-registrations/openid-connect")


@pytest.mark.asyncio
async def test_dcr_intercept_returns_static_client_id(monkeypatch):
    cfg = _enable_auth(monkeypatch, static_client_id="mcp-memgraph")
    mw = AuthMiddleware(_inner_app_not_called, cfg)

    dcr_request = {
        "client_name": "Claude",
        "redirect_uris": ["http://localhost:12345/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": "mcp:tools offline_access",
    }
    body = json.dumps(dcr_request).encode("utf-8")
    rec = _Recorder()
    await mw(
        _build_scope(method="POST", path="/register"),
        _make_receive_for(body),
        rec,
    )

    assert rec.status == 201
    response = rec.json()
    assert response["client_id"] == "mcp-memgraph"
    assert "client_secret" not in response  # public client
    assert response["redirect_uris"] == ["http://localhost:12345/callback"]
    assert response["grant_types"] == ["authorization_code", "refresh_token"]
    assert response["client_name"] == "Claude"


@pytest.mark.asyncio
async def test_dcr_intercept_disabled_falls_through(monkeypatch):
    """Without MCP_AUTH_STATIC_CLIENT_ID, POST /register goes through the
    bearer-required path and gets 401 since the request has no token. That
    confirms we don't accidentally intercept when the env var is unset.
    """
    cfg = _enable_auth(monkeypatch, static_client_id=None)
    mw = AuthMiddleware(_inner_app_not_called, cfg)

    rec = _Recorder()
    await mw(
        _build_scope(method="POST", path="/register"),
        _make_receive_for(b"{}"),
        rec,
    )

    assert rec.status == 401


@pytest.mark.asyncio
async def test_prm_points_to_self_when_intercept_enabled(monkeypatch):
    """With intercept on, PRM must advertise the MCP server itself as the AS
    so clients come back to us for AS metadata + /register."""
    cfg = _enable_auth(monkeypatch, static_client_id="mcp-memgraph")
    mw = AuthMiddleware(_inner_app_not_called, cfg)

    rec = _Recorder()
    await mw(
        _build_scope(method="GET", path="/.well-known/oauth-protected-resource"),
        _no_receive,
        rec,
    )

    assert rec.status == 200
    doc = rec.json()
    assert doc["authorization_servers"] == ["http://localhost:8000"]
    assert doc["resource"] == "http://localhost:8000"


@pytest.mark.asyncio
async def test_prm_points_to_keycloak_when_intercept_disabled(monkeypatch):
    """With intercept off, PRM must advertise the real Keycloak issuer."""
    cfg = _enable_auth(monkeypatch, static_client_id=None)
    mw = AuthMiddleware(_inner_app_not_called, cfg)

    rec = _Recorder()
    await mw(
        _build_scope(method="GET", path="/.well-known/oauth-protected-resource"),
        _no_receive,
        rec,
    )

    assert rec.status == 200
    doc = rec.json()
    assert doc["authorization_servers"] == [cfg.issuer]


@pytest.mark.asyncio
async def test_dcr_intercept_handles_missing_optional_fields(monkeypatch):
    cfg = _enable_auth(monkeypatch, static_client_id="mcp-memgraph")
    mw = AuthMiddleware(_inner_app_not_called, cfg)

    # Empty body — client sent nothing but Content-Type.
    rec = _Recorder()
    await mw(
        _build_scope(method="POST", path="/register"),
        _make_receive_for(b""),
        rec,
    )

    assert rec.status == 201
    response = rec.json()
    assert response["client_id"] == "mcp-memgraph"
    # Sensible defaults kick in.
    assert response["redirect_uris"] == []
    assert "authorization_code" in response["grant_types"]
