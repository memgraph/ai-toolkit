# Shared Token Auth (Quick & Dirty)

**Status:** Proposal
**Date:** 2026-03-31

## Goal

Minimal auth that protects LightRAG WebUI and API with a single shared token. No IdP, no user accounts, no infrastructure changes. Something you can set up in 10 minutes.

## How It Works

LightRAG already supports API key auth natively. The flow:

```
 User (browser/SDK)
       |
       |  X-API-Key: <shared-token>
       v
 LightRAG Server (port 9621)
       |
       |  bolt:// (MEMGRAPH_USER / MEMGRAPH_PASSWORD)
       v
 Memgraph (port 7687)
```

1. Generate a random token (e.g., `openssl rand -hex 32`)
2. Set `LIGHTRAG_API_KEY=<token>` on the LightRAG server
3. Share the token with authorized users
4. Users pass it via `X-API-Key` header (API) or configure it in the WebUI

Memgraph stays on username/password over Bolt — no changes needed there.

## Configuration

### LightRAG Server

```bash
# .env or environment
LIGHTRAG_API_KEY=a3f8c1...your-random-token
```

Or via CLI:

```bash
lightrag-server --key a3f8c1...your-random-token
```

### Python SDK (unstructured2graph callers)

```python
import httpx

headers = {"X-API-Key": os.environ["LIGHTRAG_API_KEY"]}
response = httpx.post("http://localhost:9621/query", headers=headers, json={...})
```

### WebUI

LightRAG's built-in WebUI checks for the API key. When `LIGHTRAG_API_KEY` is set, unauthenticated requests get a 401. The WebUI should prompt for the key — but note the caveat below.

### curl

```bash
curl -H "X-API-Key: $LIGHTRAG_API_KEY" http://localhost:9621/query -d '{"query": "..."}'
```

## Caveats

1. **WebUI guest fallback** — LightRAG's WebUI falls back to guest access if no `AUTH_ACCOUNTS` are configured, even when `LIGHTRAG_API_KEY` is set. To fully lock down the WebUI, you may also need to set `AUTH_ACCOUNTS` or put a reverse proxy in front.

2. **No identity** — everyone shares the same token. You can't distinguish who did what in logs or enforce per-user permissions.

3. **No expiry** — the token lives until you rotate it manually. If it leaks, you must regenerate and redistribute.

4. **No revocation per user** — rotating the token locks out everyone; you can't revoke one person's access.

5. **HTTP in cleartext** — without TLS, the token travels in plaintext. In production, either:
   - Enable LightRAG's built-in SSL (`--ssl --ssl-certfile cert.pem --ssl-keyfile key.pem`)
   - Put it behind a TLS-terminating reverse proxy (Nginx, Caddy)

6. **Memgraph and Memgraph Lab are separate** — this only covers LightRAG. Memgraph Bolt stays on its own username/password. Memgraph Lab has no shared-token mechanism (it uses SSO or its own login).

## Token Rotation

### Current limitation: restart required

The API key **cannot be hot-reloaded** in upstream LightRAG. It is read once at startup and captured in a closure across three layers:

1. `config.py` — `parse_args()` reads `LIGHTRAG_API_KEY` env var once into `global_args.key`
2. `lightrag_server.py` — `create_app()` captures it into a local `api_key` via `os.getenv()` once
3. `utils_api.py` — `get_combined_auth_dependency(api_key)` stores it in a closure compared on every request

Even mutating `os.environ` at runtime has no effect — the closure still holds the old value.

### Rotation without hot-reload

```bash
# 1. Generate new token
NEW_TOKEN=$(openssl rand -hex 32)

# 2. Update LightRAG server env and restart
export LIGHTRAG_API_KEY=$NEW_TOKEN
# restart lightrag-server

# 3. Distribute new token to users
```

### Proposed patch: hot-reloadable API key

A small change to `lightrag/api/utils_api.py` allows the key to be rotated by updating the env var without restarting the server. See `sso-shared-token-patch.md` for the full patch.

## When to Outgrow This

Move to the full SSO design (`sso-design.md`) when any of these become true:

- You need to know *who* is making requests (audit trail)
- Different users need different permissions (RBAC)
- You have more than ~10 users sharing the token
- Compliance requires individual credentials
- You need to revoke a single user's access without disrupting others
