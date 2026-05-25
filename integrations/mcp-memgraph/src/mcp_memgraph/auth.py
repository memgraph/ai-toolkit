"""JWT validation and per-session authentication state.

Validates Keycloak-issued bearer tokens locally via JWKS (no per-request
introspection round-trip) and derives a :class:`SessionAuth` carrying the
caller's allowed tenants intersected with the server's MCP_TENANT_CATALOG.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jwt
from jwt import PyJWKClient

if TYPE_CHECKING:
    from mcp_memgraph.config import MCPAuthConfig


class AuthError(Exception):
    """Raised when a bearer token fails validation or maps to no allowed tenant."""


@dataclass
class SessionAuth:
    """Per-MCP-session authentication state.

    The set of tenants the caller may access is derived from the JWT and frozen
    for the life of the session. ``current_tenant`` is the active database for
    subsequent tool calls and can be flipped within the allowed set via
    ``use_database``.
    """

    allowed_tenants: frozenset[str]
    current_tenant: str
    subject: str = ""
    raw_claims: dict = field(default_factory=dict)

    def can_use(self, name: str) -> bool:
        return name in self.allowed_tenants


# Module-level JWKS client cache, keyed by (jwks_url). Reused across requests so
# Keycloak is only contacted once per key-rotation cycle.
_JWKS_CLIENTS: dict[str, PyJWKClient] = {}


def _get_jwks_client(jwks_url: str) -> PyJWKClient:
    client = _JWKS_CLIENTS.get(jwks_url)
    if client is None:
        client = PyJWKClient(jwks_url, cache_keys=True)
        _JWKS_CLIENTS[jwks_url] = client
    return client


def verify_bearer(token: str, cfg: MCPAuthConfig) -> dict:
    """Validate a bearer token and return its claims.

    Raises :class:`AuthError` on any validation failure: bad signature, wrong
    issuer/audience/expiry, missing required scope.
    """
    if not cfg.jwks_url:
        raise AuthError("MCP_AUTH_JWKS_URL is not configured")

    try:
        signing_key = _get_jwks_client(cfg.jwks_url).get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256", "ES256"],
            issuer=cfg.issuer or None,
            audience=cfg.audience or None,
            options={
                "require": ["exp", "iat", "iss", "aud"],
                "verify_aud": bool(cfg.audience),
                "verify_iss": bool(cfg.issuer),
            },
        )
    except jwt.PyJWTError as e:
        raise AuthError(f"token validation failed: {e}") from e

    required = cfg.required_scope
    if required:
        scopes = set(str(claims.get("scope", "")).split())
        if required not in scopes:
            raise AuthError(f"missing required scope '{required}'")
    return claims


def extract_session_auth(claims: dict, cfg: MCPAuthConfig) -> SessionAuth:
    """Build :class:`SessionAuth` from already-validated claims.

    Intersects the JWT's ``tenants`` claim with the server-side
    ``MCP_TENANT_CATALOG``. Raises :class:`AuthError` when the intersection is
    empty (i.e., this user has no overlap with anything the server can serve).
    """
    raw_tenants = claims.get(cfg.tenants_claim, [])
    if isinstance(raw_tenants, str):
        # Tolerate single-value strings even though we expect arrays.
        raw_tenants = [raw_tenants]
    jwt_tenants = {str(t) for t in raw_tenants if t}

    catalog = cfg.tenant_catalog
    if not catalog:
        raise AuthError("MCP_TENANT_CATALOG is empty")

    allowed = frozenset(jwt_tenants & catalog)
    if not allowed:
        raise AuthError("user has no overlap with MCP_TENANT_CATALOG")

    requested_default = claims.get(cfg.default_tenant_claim)
    current = str(requested_default) if requested_default in allowed else sorted(allowed)[0]

    return SessionAuth(
        allowed_tenants=allowed,
        current_tenant=current,
        subject=str(claims.get("sub", "")),
        raw_claims=claims,
    )


# Request-scoped accessor. The middleware sets this contextvar on each
# authenticated HTTP request; tool handlers read it via current_session_auth().
_CURRENT_SESSION_AUTH: contextvars.ContextVar[SessionAuth | None] = contextvars.ContextVar(
    "mcp_memgraph_session_auth", default=None
)


def set_current_session_auth(auth: SessionAuth | None) -> contextvars.Token:
    return _CURRENT_SESSION_AUTH.set(auth)


def reset_current_session_auth(token: contextvars.Token) -> None:
    _CURRENT_SESSION_AUTH.reset(token)


def current_session_auth() -> SessionAuth | None:
    """Return the SessionAuth for the in-flight request, or None when auth is off."""
    return _CURRENT_SESSION_AUTH.get()
