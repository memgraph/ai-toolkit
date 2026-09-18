# Patch: Hot-Reloadable API Key for LightRAG

**Status:** Proposal
**Date:** 2026-04-01
**Targets:** [LightRAG](https://github.com/HKUDS/LightRAG) upstream

## Problem

`LIGHTRAG_API_KEY` is read once at startup and captured in a closure. Rotating the key requires restarting the server, which drops in-flight requests and interrupts users.

## Root Cause

In `lightrag/api/utils_api.py`, `get_combined_auth_dependency(api_key)` receives the key as a parameter and closes over it:

```python
# Current code (simplified from utils_api.py)
def get_combined_auth_dependency(api_key: Optional[str] = None):
    api_key_configured = api_key is not None

    async def combined_dependency(
        request: Request,
        api_key_header_value: Optional[str] = Security(api_key_header),
        token: Optional[str] = Depends(oauth2_scheme_optional),
    ):
        # ...
        if api_key_configured and api_key_header_value and api_key_header_value == api_key:
            return  # API key validation successful
        # ...

    return combined_dependency
```

The `api_key` variable is bound at dependency-creation time, not at request time.

## Fix

Re-read `os.environ` on each request instead of comparing against the closure-captured value.

### Patch for `lightrag/api/utils_api.py`

```diff
 def get_combined_auth_dependency(api_key: Optional[str] = None):
-    api_key_configured = api_key is not None
+    # Store the initial key as fallback, but prefer the live env var
+    initial_api_key = api_key

     async def combined_dependency(
         request: Request,
         api_key_header_value: Optional[str] = Security(api_key_header),
         token: Optional[str] = Depends(oauth2_scheme_optional),
     ):
+        # Re-read env var on each request to support hot-reload
+        current_api_key = os.environ.get("LIGHTRAG_API_KEY") or initial_api_key
+        api_key_configured = current_api_key is not None
+
         # ... (whitelist check unchanged) ...

         if api_key_configured and api_key_header_value and api_key_header_value == current_api_key:
             return  # API key validation successful

         # ... (rest of auth logic unchanged) ...

     return combined_dependency
```

### Patch for `lightrag/api/lightrag_server.py`

No change needed. `create_app()` still passes `api_key` to `get_combined_auth_dependency()` — it just becomes the fallback if the env var is unset.

## Rotation flow after patching

```bash
# No restart needed — just update the env var in the running process
# Option 1: If running via a process manager that supports env updates (e.g., systemd, Docker)
docker exec <container> sh -c 'export LIGHTRAG_API_KEY=<new-token>'

# Option 2: Add a small admin endpoint (optional, see below)
curl -X POST http://localhost:9621/admin/rotate-key \
  -H "X-API-Key: $OLD_KEY" \
  -d '{"new_key": "<new-token>"}'
```

### Optional: Admin endpoint for key rotation

If you can't easily update env vars in the running process (e.g., Docker without exec), add a small admin endpoint:

```python
# Add to lightrag/api/lightrag_server.py in create_app()

@app.post("/admin/rotate-key")
async def rotate_key(
    request: Request,
    body: dict,
    _=Depends(combined_auth),  # must authenticate with current key
):
    new_key = body.get("new_key")
    if not new_key or len(new_key) < 32:
        raise HTTPException(400, "new_key must be at least 32 characters")
    os.environ["LIGHTRAG_API_KEY"] = new_key
    return {"status": "rotated"}
```

This is self-securing: you need the current key to set a new one.

## Performance Impact

`os.environ.get()` is a dict lookup in CPython — negligible cost per request (nanoseconds). No caching or locking needed.

## Contribution Path

This is a small, backwards-compatible change suitable for an upstream PR to [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG). If upstream doesn't accept it, it can be monkey-patched in `lightrag-memgraph` or applied as a fork patch.
