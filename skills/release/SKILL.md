---
name: release
description: Used to release all toolbox, integrations, agents. Use when releasing a subproject to PyPI, Docker Hub, or when the user asks to release or publish.
---

# Release Workflows

All release workflows are triggered via **`workflow_dispatch`** (manual) from the GitHub Actions tab. Bump the version in the subproject's `pyproject.toml` before dispatching.

## Workflow files

| Workflow file                     | Package            | What it does                         | Secrets used                                          |
| --------------------------------- | ------------------ | ------------------------------------ | ----------------------------------------------------- |
| `release-mcp-memgraph.yaml`       | mcp-memgraph       | Build & publish to PyPI + Docker Hub | `PYPI_TOKEN`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` |
| `release-toolbox.yaml`            | memgraph-toolbox   | Build & publish to PyPI              | `PYPI_TOKEN`                                          |
| `release-langchain-memgraph.yaml` | langchain-memgraph | Build & publish to PyPI              | `PYPI_TOKEN`                                          |
| `release-lightrag-memgraph.yaml`  | lightrag-memgraph  | Build & publish to PyPI              | `PYPI_TOKEN`                                          |
| `release-unstructured2graph.yaml` | unstructured2graph | Build & publish to PyPI              | `PYPI_TOKEN`                                          |

## Subproject paths

| Package            | Path                              | pyproject.toml version field |
| ------------------ | --------------------------------- | ---------------------------- |
| memgraph-toolbox   | `memgraph-toolbox`                | `[project] version`          |
| mcp-memgraph       | `integrations/mcp-memgraph`       | `[project] version`          |
| langchain-memgraph | `integrations/langchain-memgraph` | `[project] version`          |
| lightrag-memgraph  | `integrations/lightrag-memgraph`  | `[project] version`          |
| unstructured2graph | `unstructured2graph`              | `[project] version`          |

## Required GitHub secrets

| Secret               | Used by                | Description                     |
| -------------------- | ---------------------- | ------------------------------- |
| `PYPI_TOKEN`         | All release workflows  | PyPI API token for `uv publish` |
| `DOCKERHUB_USERNAME` | `release-mcp-memgraph` | Docker Hub username             |
| `DOCKERHUB_TOKEN`    | `release-mcp-memgraph` | Docker Hub access token         |

## Notes

- **lightrag-memgraph** and **unstructured2graph** use `uv build --out-dir dist` to work around a uv artifact path issue.
- The **mcp-memgraph** Docker image is built from the repo root (the Dockerfile copies both `memgraph-toolbox/` and `integrations/mcp-memgraph/`).
- The mcp-memgraph workflow tags the Docker image with both the version and `latest`.
