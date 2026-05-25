"""Unit tests for the Keycloak JWT auth module.

These tests do not require a running Keycloak instance: they mint JWTs locally
with a freshly-generated RSA keypair and monkeypatch :class:`PyJWKClient` to
hand back the corresponding signing key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from mcp_memgraph import auth as auth_mod
from mcp_memgraph.config import MCPAuthConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeKey:
    key: object


@pytest.fixture(scope="module")
def rsa_keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return private_pem, public_pem, private.public_key()


@pytest.fixture
def fake_jwks_client(monkeypatch, rsa_keypair):
    _, _, public_key = rsa_keypair

    class _FakeJWKSClient:
        def __init__(self, *_a, **_k):
            pass

        def get_signing_key_from_jwt(self, _token):
            return _FakeKey(public_key)

    monkeypatch.setattr(auth_mod, "PyJWKClient", _FakeJWKSClient)
    # Clear the module-level cache so a fresh client is built per test.
    monkeypatch.setattr(auth_mod, "_JWKS_CLIENTS", {})


def _make_cfg(
    *,
    issuer: str = "https://example.test/realms/memgraph",
    audience: str = "https://mcp.example.test",
    tenant_catalog: str = "tenant-a,tenant-b,tenant-c",
    required_scope: str = "mcp:tools",
    monkeypatch,
) -> MCPAuthConfig:
    monkeypatch.setenv("MCP_AUTH_ENABLED", "true")
    monkeypatch.setenv("MCP_AUTH_ISSUER", issuer)
    monkeypatch.setenv("MCP_AUTH_AUDIENCE", audience)
    monkeypatch.setenv("MCP_AUTH_JWKS_URL", issuer + "/protocol/openid-connect/certs")
    monkeypatch.setenv("MCP_AUTH_REQUIRED_SCOPE", required_scope)
    monkeypatch.setenv("MCP_TENANT_CATALOG", tenant_catalog)
    return MCPAuthConfig()


def _mint(
    private_pem: bytes,
    *,
    iss: str,
    aud: str,
    tenants: list[str] | None,
    scope: str = "mcp:tools",
    default_tenant: str | None = None,
    sub: str = "user-uuid",
    expires_in: int = 300,
) -> str:
    now = int(time.time())
    payload: dict = {
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "iat": now,
        "exp": now + expires_in,
        "scope": scope,
    }
    if tenants is not None:
        payload["tenants"] = tenants
    if default_tenant is not None:
        payload["default_tenant"] = default_tenant
    return jwt.encode(payload, private_pem, algorithm="RS256")


# ---------------------------------------------------------------------------
# verify_bearer
# ---------------------------------------------------------------------------


def test_verify_bearer_accepts_valid_token(monkeypatch, rsa_keypair, fake_jwks_client):
    cfg = _make_cfg(monkeypatch=monkeypatch)
    private_pem, _, _ = rsa_keypair
    token = _mint(private_pem, iss=cfg.issuer, aud=cfg.audience, tenants=["tenant-a"])

    claims = auth_mod.verify_bearer(token, cfg)

    assert claims["aud"] == cfg.audience
    assert claims["tenants"] == ["tenant-a"]


def test_verify_bearer_rejects_wrong_audience(monkeypatch, rsa_keypair, fake_jwks_client):
    cfg = _make_cfg(monkeypatch=monkeypatch)
    private_pem, _, _ = rsa_keypair
    token = _mint(private_pem, iss=cfg.issuer, aud="https://someone-else", tenants=["tenant-a"])

    with pytest.raises(auth_mod.AuthError):
        auth_mod.verify_bearer(token, cfg)


def test_verify_bearer_rejects_expired_token(monkeypatch, rsa_keypair, fake_jwks_client):
    cfg = _make_cfg(monkeypatch=monkeypatch)
    private_pem, _, _ = rsa_keypair
    token = _mint(
        private_pem, iss=cfg.issuer, aud=cfg.audience, tenants=["tenant-a"], expires_in=-10
    )

    with pytest.raises(auth_mod.AuthError):
        auth_mod.verify_bearer(token, cfg)


def test_verify_bearer_rejects_missing_required_scope(monkeypatch, rsa_keypair, fake_jwks_client):
    cfg = _make_cfg(monkeypatch=monkeypatch)
    private_pem, _, _ = rsa_keypair
    token = _mint(
        private_pem, iss=cfg.issuer, aud=cfg.audience, tenants=["tenant-a"], scope="other"
    )

    with pytest.raises(auth_mod.AuthError, match="missing required scope"):
        auth_mod.verify_bearer(token, cfg)


def test_verify_bearer_rejects_bad_signature(monkeypatch, rsa_keypair, fake_jwks_client):
    cfg = _make_cfg(monkeypatch=monkeypatch)
    private_pem, _, _ = rsa_keypair
    token = _mint(private_pem, iss=cfg.issuer, aud=cfg.audience, tenants=["tenant-a"])
    # Tamper with the signature.
    tampered = token[:-3] + ("AAA" if token[-3:] != "AAA" else "BBB")

    with pytest.raises(auth_mod.AuthError):
        auth_mod.verify_bearer(tampered, cfg)


# ---------------------------------------------------------------------------
# extract_session_auth
# ---------------------------------------------------------------------------


def test_extract_session_auth_intersects_with_catalog(monkeypatch):
    cfg = _make_cfg(tenant_catalog="tenant-a,tenant-b,tenant-c", monkeypatch=monkeypatch)
    claims = {"tenants": ["tenant-a", "tenant-c", "tenant-z"]}

    sa = auth_mod.extract_session_auth(claims, cfg)

    assert sa.allowed_tenants == frozenset({"tenant-a", "tenant-c"})
    # tenant-z (not in catalog) is dropped, tenant-b (in catalog, not in token) too.


def test_extract_session_auth_default_tenant_used_when_allowed(monkeypatch):
    cfg = _make_cfg(tenant_catalog="tenant-a,tenant-c", monkeypatch=monkeypatch)
    claims = {"tenants": ["tenant-a", "tenant-c"], "default_tenant": "tenant-c"}

    sa = auth_mod.extract_session_auth(claims, cfg)

    assert sa.current_tenant == "tenant-c"


def test_extract_session_auth_default_tenant_falls_back_when_not_allowed(monkeypatch):
    cfg = _make_cfg(tenant_catalog="tenant-a,tenant-c", monkeypatch=monkeypatch)
    claims = {"tenants": ["tenant-a", "tenant-c"], "default_tenant": "tenant-z"}

    sa = auth_mod.extract_session_auth(claims, cfg)

    # default_tenant not in allowed → falls back to first sorted member.
    assert sa.current_tenant == "tenant-a"


def test_extract_session_auth_rejects_empty_intersection(monkeypatch):
    cfg = _make_cfg(tenant_catalog="tenant-a,tenant-b", monkeypatch=monkeypatch)
    claims = {"tenants": ["tenant-z"]}

    with pytest.raises(auth_mod.AuthError, match="no overlap"):
        auth_mod.extract_session_auth(claims, cfg)


def test_extract_session_auth_rejects_missing_tenants_claim(monkeypatch):
    cfg = _make_cfg(monkeypatch=monkeypatch)
    claims = {}

    with pytest.raises(auth_mod.AuthError):
        auth_mod.extract_session_auth(claims, cfg)


def test_extract_session_auth_rejects_empty_catalog(monkeypatch):
    cfg = _make_cfg(tenant_catalog="", monkeypatch=monkeypatch)
    claims = {"tenants": ["tenant-a"]}

    with pytest.raises(auth_mod.AuthError, match="MCP_TENANT_CATALOG is empty"):
        auth_mod.extract_session_auth(claims, cfg)


def test_extract_session_auth_tolerates_string_tenants_claim(monkeypatch):
    cfg = _make_cfg(tenant_catalog="tenant-a,tenant-b", monkeypatch=monkeypatch)
    claims = {"tenants": "tenant-a"}  # single string instead of array

    sa = auth_mod.extract_session_auth(claims, cfg)

    assert sa.allowed_tenants == frozenset({"tenant-a"})
    assert sa.current_tenant == "tenant-a"
