# SSO (Single Sign-On) Design for unstructured2graph

**Status:** Discovery / Requirements Gathering
**Date:** 2026-03-31

## 1. System Landscape

The SSO solution must unify authentication across three components:

| Component | Protocol/Interface | Current Auth |
|---|---|---|
| **Memgraph** (database) | Bolt (port 7687) | Username/password via env vars (`MEMGRAPH_URL`, `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD`). Enterprise supports OIDC (Entra ID, Okta), SAML, and LDAP via external auth modules. |
| **Memgraph Lab** (UI) | HTTP (browser) | Enterprise Docker deployment supports OIDC and SAML SSO with Entra ID and Okta. Desktop version has no SSO. |
| **LightRAG** (AI/RAG API) | REST/FastAPI (port 9621) | API key (`X-API-Key` header) and/or JWT tokens (OAuth2 password flow at `/login`). Supports bcrypt-hashed passwords, token expiry, auto-renewal, and path whitelisting. |
| **unstructured2graph** (this project) | Python library | No auth layer — inherits Memgraph connection credentials from env vars and LightRAG wrapper config. |

## 2. Approach Options

### Option A: Centralized Identity Provider (IdP) with OIDC

A single IdP (e.g., Entra ID, Okta, Keycloak, Auth0) issues tokens consumed by all components.

- Memgraph Enterprise: native OIDC support via external auth module
- Memgraph Lab Enterprise (Docker): native OIDC support
- LightRAG: would need a custom auth dependency or reverse-proxy layer that validates OIDC tokens
- unstructured2graph: token-passing middleware or service account credentials

### Option B: Token Broker / API Gateway

An API gateway (e.g., Kong, APISIX, Nginx + OAuth2-proxy) sits in front of LightRAG and unstructured2graph, handling authentication centrally. Memgraph uses its own OIDC/LDAP integration.

### Option C: Shared JWT with a Lightweight Auth Service

A custom auth service issues JWTs after authenticating against the IdP. All services validate the same JWT. Memgraph's external auth module is extended to validate these JWTs on Bolt connections.

### Option D: Service Accounts + User SSO (Hybrid)

User-facing SSO via OIDC for Memgraph Lab and a web frontend, while backend services (unstructured2graph, LightRAG) use service account credentials or short-lived tokens minted from the IdP's client-credentials flow.

---

## 3. Token Type Considerations

| Token Type | Pros | Cons |
|---|---|---|
| **OIDC ID Token (JWT)** | Standard claims (`sub`, `email`, `roles`), widely supported, Memgraph native support | Short-lived, needs refresh flow, not designed for API authorization |
| **OIDC Access Token (opaque or JWT)** | Designed for API auth, scoped, Memgraph accepts it | May require introspection endpoint if opaque |
| **Custom JWT** | Full control over claims, single validation across services | Must build issuance and rotation; not natively recognized by Memgraph |
| **API Key** | Simple, LightRAG supports it natively | No identity, no expiry by default, hard to revoke at scale |
| **SAML Assertion** | Enterprise-friendly, Memgraph supports it | XML-based, not practical for API-to-API calls |

---

## 4. Client Connection Patterns

| Client | Connects To | Mechanism |
|---|---|---|
| Browser user | Memgraph Lab | OIDC redirect flow (authorization code + PKCE) |
| Browser user | LightRAG WebUI | OIDC redirect or JWT from `/login` |
| Python SDK (unstructured2graph) | Memgraph (Bolt) | Token passed via Neo4j driver auth (`access_token=...;id_token=...`) |
| Python SDK (unstructured2graph) | LightRAG API | Bearer token or API key in HTTP header |
| CI/CD / batch jobs | Memgraph + LightRAG | Client-credentials (machine-to-machine) tokens |
| Memgraph Lab | Memgraph (Bolt) | Forwards user's SSO token to database |

---

## 5. LightRAG AI Authentication

LightRAG's API provides RAG queries, document management, and an Ollama-compatible chat interface. Authentication options:

1. **Extend LightRAG's FastAPI auth dependency** — replace `get_combined_auth_dependency()` in `lightrag/api/utils_api.py` with OIDC token validation (e.g., using `python-jose` or `authlib`).
2. **Reverse proxy with token validation** — Nginx/Envoy validates OIDC tokens before forwarding to LightRAG; LightRAG runs unauthenticated behind the proxy.
3. **Keep LightRAG's JWT but federate login** — LightRAG's `/login` endpoint delegates to the IdP instead of local username/password. The `AuthHandler` class in `lightrag/api/auth.py` is the extension point.

---

## 6. Follow-Up Questions

Before designing the solution, I need clarity on the following:

### Deployment & Infrastructure

1. **Are you running Memgraph Enterprise or Community?** OIDC/SAML SSO and RBAC are Enterprise-only features. If Community, we're limited to basic username/password on Bolt, and the SSO scope shrinks to LightRAG + any web layer.

2. **How is the stack deployed?** Docker Compose? Kubernetes? Bare metal? This affects whether an API gateway or sidecar proxy approach is practical.

3. **Is Memgraph Lab deployed via Docker or used as a desktop app?** SSO in Lab requires the Docker deployment.

4. **Is there an existing Identity Provider (IdP)?** If so, which one — Entra ID, Okta, Keycloak, Google Workspace, or something else? Or do we need to stand one up (e.g., Keycloak)?

### Users & Access Patterns

5. **Who are the users?** Internal team only, or external/customer-facing? This affects the complexity of the auth flow and whether self-registration is needed.

6. **Do you need role-based access control (RBAC)?** For example: some users can only query (read), others can ingest documents (write), admins manage the system. If so, where should roles be enforced — at the IdP level, Memgraph level, LightRAG level, or all?

7. **Are there machine-to-machine (M2M) clients?** For example, CI/CD pipelines, cron jobs, or other services calling unstructured2graph programmatically. These typically need client-credentials flow rather than user login.

8. **Is multi-tenancy a concern?** Should different teams/orgs see different subsets of the knowledge graph?

### LightRAG Specifics

9. **Are you running LightRAG as a standalone server (`lightrag-server`) or only as an embedded Python library?** The SSO approach differs significantly — a server has HTTP endpoints to protect, while a library delegates auth to the calling application.

10. **Do you need to secure the LightRAG WebUI separately from the API?** The WebUI currently falls back to guest access if no auth accounts are configured.

11. **Which LightRAG endpoints need protection?** All of them, or only write operations (document upload, graph mutation) while queries remain open?

### Token & Session Management

12. **What token lifetime is acceptable?** Short-lived (minutes) with refresh, or longer-lived (hours)? This affects UX for interactive users vs. batch processes.

13. **Do you need token revocation?** For example, if someone leaves the team, can you wait for tokens to expire, or must you revoke immediately?

14. **Should the Python SDK (unstructured2graph) handle token refresh transparently**, or is it acceptable to require callers to manage their own tokens?

### Scope & Priority

15. **Is this for a production deployment or a development/POC setup?** This determines how much infrastructure (IdP, gateway, TLS) to invest in upfront.

16. **What's the priority order?** If we can't do everything at once, which is most important:
    - Securing Memgraph Bolt connections?
    - SSO for Memgraph Lab?
    - Protecting LightRAG API/WebUI?
    - Unified identity across all three?

17. **Are there compliance requirements** (SOC2, HIPAA, GDPR) that constrain token storage, session length, or audit logging?
