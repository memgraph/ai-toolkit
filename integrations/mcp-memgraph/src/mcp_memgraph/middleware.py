"""ASGI middleware that enforces Keycloak JWT auth on the streamable-http app.

Responsibilities:

* Validate the bearer on every request (except ``/health`` and the OAuth /
  OIDC discovery endpoints).
* Derive a :class:`SessionAuth` and stash it on a contextvar so tool handlers
  can read it via :func:`mcp_memgraph.auth.current_session_auth`.
* Preserve the caller's ``current_tenant`` across requests bearing the same
  ``Mcp-Session-Id`` so a successful ``use_database`` call survives until the
  session closes.
* Serve enough discovery documents that current MCP clients (Claude, Cursor,
  VS Code) can find the Keycloak authorization server:

    - RFC 9728 Protected Resource Metadata at
      ``/.well-known/oauth-protected-resource`` (for clients that read
      ``WWW-Authenticate``'s ``resource_metadata`` parameter).
    - RFC 8414 Authorization Server Metadata at
      ``/.well-known/oauth-authorization-server`` plus the ``/mcp/...`` and
      ``.../mcp`` variants some clients probe. Body is Keycloak's own
      OIDC discovery document, fetched lazily and cached.
    - OIDC discovery at ``/.well-known/openid-configuration`` (same body, same
      variants). Clients use this to learn ``authorization_endpoint``,
      ``token_endpoint``, ``registration_endpoint``, etc., all pointing at
      Keycloak.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import TYPE_CHECKING

import httpx

from mcp_memgraph.auth import (
    AuthError,
    SessionAuth,
    extract_session_auth,
    reset_current_session_auth,
    set_current_session_auth,
    verify_bearer,
)

if TYPE_CHECKING:
    from mcp_memgraph.config import MCPAuthConfig

logger = logging.getLogger("mcp-memgraph.auth")


# Session-state map keyed by Mcp-Session-Id. Used to preserve current_tenant
# across requests in the same MCP session. In-memory in v1; swap for Redis
# when the server is scaled horizontally.
_SESSION_STATE: dict[str, SessionAuth] = {}
_SESSION_LOCK = threading.Lock()


def _read_session_state(session_id: str | None) -> SessionAuth | None:
    if not session_id:
        return None
    with _SESSION_LOCK:
        return _SESSION_STATE.get(session_id)


def _store_session_state(session_id: str | None, auth: SessionAuth) -> None:
    if not session_id:
        return
    with _SESSION_LOCK:
        _SESSION_STATE[session_id] = auth


def _clear_session_state(session_id: str) -> None:
    with _SESSION_LOCK:
        _SESSION_STATE.pop(session_id, None)


def _header_value(scope: dict, name: bytes) -> str:
    for k, v in scope.get("headers", []):
        if k == name:
            try:
                return v.decode("latin-1")
            except UnicodeDecodeError:
                return ""
    return ""


async def _read_request_body(receive) -> bytes:
    """Drain an ASGI request body into a single bytes blob."""
    chunks: list[bytes] = []
    more = True
    while more:
        message = await receive()
        if message["type"] != "http.request":
            continue
        chunks.append(message.get("body", b""))
        more = message.get("more_body", False)
    return b"".join(chunks)


# Paths whose responses are derived from the upstream OIDC discovery document.
# Many MCP clients probe these directly on the resource server rather than
# following the PRM URL from WWW-Authenticate, so we have to answer them too.
_AS_METADATA_SUFFIXES = (
    "/.well-known/oauth-authorization-server",
    "/.well-known/openid-configuration",
)


def _is_as_metadata_path(path: str) -> bool:
    """Return True for any flavor of AS-metadata / OIDC-discovery URL.

    Recognized: the canonical paths, ``/.well-known/<x>/<anything>`` (clients
    that append the MCP route as a suffix), and ``<anything>/.well-known/<x>``
    (clients that prefix it with the MCP route).
    """
    for suffix in _AS_METADATA_SUFFIXES:
        if path == suffix:
            return True
        if path.startswith(suffix + "/"):
            return True
        if path.endswith(suffix):
            return True
    return False


class AuthMiddleware:
    """Starlette/ASGI middleware enforcing JWT auth + discovery endpoints."""

    BYPASS_PREFIXES = ("/health",)

    def __init__(self, app, cfg: MCPAuthConfig) -> None:
        self.app = app
        self.cfg = cfg
        self._oidc_doc: dict | None = None
        self._oidc_lock = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Health stays unauthenticated for K8s probes.
        if any(path == p or path.startswith(p + "/") for p in self.BYPASS_PREFIXES):
            await self.app(scope, receive, send)
            return

        # Discovery: PRM tells clients which authorization server to use.
        if path == "/.well-known/oauth-protected-resource":
            await self._serve_prm(scope, send)
            return

        # Discovery: AS metadata / OIDC config — re-serve Keycloak's doc so
        # clients can find the authorize/token/registration endpoints.
        if _is_as_metadata_path(path):
            await self._serve_oidc_doc(scope, send)
            return

        # DCR intercept (only when MCP_AUTH_STATIC_CLIENT_ID is set).
        # Implements the "shared public client" pattern: every IDE install
        # that DCRs ends up with the same pre-registered client_id, because
        # user identity (sub + tenants claim) is the actual authorization
        # signal, not which IDE the user happens to be running.
        if self.cfg.dcr_intercept_enabled and method == "POST" and path == "/register":
            await self._serve_dcr_response(scope, receive, send)
            return

        # Everything else requires a valid bearer.
        auth_header = _header_value(scope, b"authorization")
        if not auth_header.lower().startswith("bearer "):
            await self._send_401(scope, send, "missing bearer token")
            return
        token = auth_header[len("bearer ") :].strip()

        try:
            claims = verify_bearer(token, self.cfg)
            session_auth = extract_session_auth(claims, self.cfg)
        except AuthError as e:
            logger.info("rejecting request: %s", e)
            await self._send_401(scope, send, "invalid token")
            return

        session_id = _header_value(scope, b"mcp-session-id") or None
        prior = _read_session_state(session_id)
        if prior is not None and prior.current_tenant in session_auth.allowed_tenants:
            session_auth.current_tenant = prior.current_tenant
        _store_session_state(session_id, session_auth)

        cv_token = set_current_session_auth(session_auth)

        # Treat DELETE on the MCP root as the client's session-close signal so
        # we don't leak SessionAuth entries forever.
        delete_close = scope.get("method") == "DELETE" and session_id

        try:
            await self.app(scope, receive, send)
        finally:
            reset_current_session_auth(cv_token)
            if delete_close:
                _clear_session_state(session_id)

    # ------------------------------------------------------------------ Discovery

    async def _fetch_oidc_doc(self) -> dict:
        """Fetch Keycloak's OIDC discovery doc once and cache it."""
        if self._oidc_doc is not None:
            return self._oidc_doc
        async with self._oidc_lock:
            if self._oidc_doc is not None:
                return self._oidc_doc
            url = self.cfg.issuer.rstrip("/") + "/.well-known/openid-configuration"
            logger.info("fetching upstream OIDC discovery: %s", url)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                self._oidc_doc = resp.json()
        return self._oidc_doc

    async def _serve_oidc_doc(self, scope, send) -> None:
        try:
            doc = await self._fetch_oidc_doc()
        except Exception as e:  # network or Keycloak down
            logger.error("failed to fetch OIDC discovery doc: %s", e)
            body = json.dumps({"error": "upstream OIDC discovery unavailable"}).encode("utf-8")
            await self._raw_send(send, 502, body)
            return

        # If DCR intercept is on, advertise *our* /register so the client
        # POSTs to us instead of Keycloak. Everything else (authorize/token/
        # jwks/etc) still points at Keycloak — we only intercept registration.
        if self.cfg.dcr_intercept_enabled:
            doc = dict(doc)
            local = self._public_url(scope).rstrip("/") + "/register"
            doc["registration_endpoint"] = local
            mtls = doc.get("mtls_endpoint_aliases")
            if isinstance(mtls, dict) and "registration_endpoint" in mtls:
                mtls = dict(mtls)
                mtls["registration_endpoint"] = local
                doc["mtls_endpoint_aliases"] = mtls

        body = json.dumps(doc).encode("utf-8")
        await self._raw_send(send, 200, body)

    # ------------------------------------------------------------------ DCR intercept

    async def _serve_dcr_response(self, scope, receive, send) -> None:
        """Pretend to register a client; actually hand back the one shared public client.

        RFC 7591 response shape, with redirect_uris and grant_types echoed from
        the request so the calling MCP client uses whatever it asked for. No
        client_secret is returned (public client).

        Note: returning the same client_id for every caller is intentional.
        See the MCP_AUTH_STATIC_CLIENT_ID docstring in config.py for the
        rationale (shared-client design + Claude DCR workaround).
        """
        request_body = await _read_request_body(receive)
        try:
            req = json.loads(request_body) if request_body else {}
        except json.JSONDecodeError:
            req = {}

        client_id = self.cfg.static_client_id
        logger.info("DCR intercept: returning static client_id=%s", client_id)

        response = {
            "client_id": client_id,
            "client_id_issued_at": int(time.time()),
            "token_endpoint_auth_method": req.get("token_endpoint_auth_method", "none"),
            "grant_types": req.get("grant_types", ["authorization_code", "refresh_token"]),
            "response_types": req.get("response_types", ["code"]),
            "redirect_uris": req.get("redirect_uris", []),
            "scope": req.get("scope", "mcp:tools offline_access"),
        }
        # Optional fields some clients look for.
        if req.get("client_name"):
            response["client_name"] = req["client_name"]
        body = json.dumps(response).encode("utf-8")
        await self._raw_send(send, 201, body)

    async def _serve_prm(self, scope, send) -> None:
        public = self._public_url(scope).rstrip("/")
        # When DCR intercept is on, advertise *ourselves* as the authorization
        # server. The client then fetches AS metadata from us — and we serve
        # Keycloak's doc with the registration_endpoint rewritten to our
        # /register, so DCR comes to us. Authorize/token/jwks still point at
        # Keycloak, so the actual auth flow runs there.
        if self.cfg.dcr_intercept_enabled:
            authorization_servers = [public]
        else:
            authorization_servers = [self.cfg.issuer] if self.cfg.issuer else []
        doc = {
            "resource": public,
            "authorization_servers": authorization_servers,
            "scopes_supported": [self.cfg.required_scope] if self.cfg.required_scope else [],
            "bearer_methods_supported": ["header"],
        }
        body = json.dumps(doc).encode("utf-8")
        await self._raw_send(send, 200, body)

    # ------------------------------------------------------------------ 401

    async def _send_401(self, scope, send, message: str) -> None:
        prm_url = self._public_url(scope) + "/.well-known/oauth-protected-resource"
        # RFC 6750 §3: include scope so the client knows what to request
        # rather than having to fetch PRM just to learn it (avoids a
        # round-trip on every fresh auth attempt).
        parts = ['realm="mcp"', f'resource_metadata="{prm_url}"']
        if self.cfg.required_scope:
            parts.append(f'scope="{self.cfg.required_scope}"')
        challenge = "Bearer " + ", ".join(parts)
        body = json.dumps({"error": message}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", challenge.encode("latin-1")),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    # ------------------------------------------------------------------ Helpers

    def _public_url(self, scope) -> str:
        """Best-effort public URL the client used to reach us (for PRM/411)."""
        host = _header_value(scope, b"host")
        if not host:
            return self.cfg.audience.rstrip("/")
        scheme = "https" if scope.get("scheme") == "https" else "http"
        return f"{scheme}://{host}"

    async def _raw_send(self, send, status: int, body: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
