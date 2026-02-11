---
name: release
description: Used to release all toolbox, integrations, agents. Use when releasing a subproject to PyPI or when the user asks to release or publish.
---

# Release a subproject to PyPI

## 1. Check latest version on PyPI

Before bumping or publishing, confirm the current published version so the new release is higher.

**Option A – pip (any subproject):**
```bash
pip index versions <package-name>
```
Example: `pip index versions lightrag-memgraph` → shows e.g. `0.1.4`.

**Option B – PyPI JSON API:**
```bash
curl -s https://pypi.org/pypi/<package-name>/json | python -c "import sys,json; d=json.load(sys.stdin); print(d['info']['version'])"
```
Use the same `package-name` as in `pyproject.toml` (e.g. `lightrag-memgraph`).

## 2. Bump version if needed

In the subproject’s `pyproject.toml`, set `version` to a value **greater** than the latest on PyPI (e.g. patch: 0.1.4 → 0.1.5).

## 3. Build and publish

From the **repo root** (ai-toolkit), source env then run from the **subproject directory**:

```bash
source .env
cd <subproject-path>   # e.g. integrations/lightrag-memgraph
uv build
uv publish
```

**Build output location:** `uv build` may write artifacts to the **top-level ai-toolkit `dist/`** directory rather than the subproject’s. If `uv publish` reports "No files found to publish", run publish from the **ai-toolkit root** and pass the built files explicitly:

```bash
cd /path/to/ai-toolkit
source .env
uv publish dist/<package>_<normalized>-<version>-*.whl dist/<package>_<normalized>-<version>.tar.gz
```

Example for lightrag-memgraph 0.1.4: `uv publish dist/lightrag_memgraph-0.1.4-py3-none-any.whl dist/lightrag_memgraph-0.1.4.tar.gz`

Requires `UV_PUBLISH_TOKEN` in `.env` (and `UV_PUBLISH_USERNAME` for CI; local token auth may not need it).

## Subproject paths (from ai-toolkit root)

| Package             | Path                          |
|---------------------|-------------------------------|
| memgraph-toolbox    | memgraph-toolbox              |
| mcp-memgraph        | integrations/mcp-memgraph    |
| langchain-memgraph  | integrations/langchain-memgraph |
| lightrag-memgraph   | integrations/lightrag-memgraph  |
| unstructured2graph  | unstructured2graph           |