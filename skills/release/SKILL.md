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
| `release-hygm.yaml`               | hygm               | Build & publish to PyPI              | `PYPI_TOKEN`                                          |
| `release-agent-context-graph.yaml` | agent-context-graph | Build & publish to PyPI             | `PYPI_TOKEN`                                          |
| `release-actions-graph.yaml`      | actions-graph      | Build & publish to PyPI              | `PYPI_TOKEN`                                          |
| `release-skills-graph.yaml`       | skills-graph       | Build & publish to PyPI              | `PYPI_TOKEN`                                          |
| `release-resources-graph.yaml`     | resources-graph     | Build & publish to PyPI              | `PYPI_TOKEN`                                          |
| `release-sessions-graph.yaml`     | sessions-graph     | Build & publish to PyPI              | `PYPI_TOKEN`                                          |

## Subproject paths

| Package            | Path                              | pyproject.toml version field |
| ------------------ | --------------------------------- | ---------------------------- |
| memgraph-toolbox   | `memgraph-toolbox`                | `[project] version`          |
| mcp-memgraph       | `integrations/mcp-memgraph`       | `[project] version`          |
| langchain-memgraph | `integrations/langchain-memgraph` | `[project] version`          |
| lightrag-memgraph  | `integrations/lightrag-memgraph`  | `[project] version`          |
| unstructured2graph | `unstructured2graph`              | `[project] version`          |
| hygm               | `hygm`                            | `[project] version`          |
| agent-context-graph | `context-graph/agent-context-graph` | `[project] version`        |
| actions-graph      | `context-graph/actions-graph`     | `[project] version`          |
| skills-graph       | `context-graph/skills-graph`      | `[project] version`          |
| resources-graph    | `context-graph/resources-graph`    | `[project] version`          |
| sessions-graph     | `context-graph/sessions-graph`    | `[project] version`          |

## Required GitHub secrets

| Secret               | Used by                | Description                     |
| -------------------- | ---------------------- | ------------------------------- |
| `PYPI_TOKEN`         | All release workflows  | PyPI API token for `uv publish` |
| `DOCKERHUB_USERNAME` | `release-mcp-memgraph` | Docker Hub username             |
| `DOCKERHUB_TOKEN`    | `release-mcp-memgraph` | Docker Hub access token         |

## Notes

- **Publish dependencies first.** PyPI resolves the published floor of every dependency, so release in dependency order: `memgraph-toolbox` → `hygm` → `unstructured2graph`; `agent-context-graph` → `actions-graph` / `skills-graph` → `sessions-graph`. Publish `memgraph-toolbox` and `agent-context-graph` before `resources-graph` (its optional harness integration).
- **Context Graph plugins** install from PyPI in `context-graph/plugins/*/scripts/bootstrap.sh`. When a release is one the plugins need, raise the floors there and bump both plugin manifests (`.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`) plus the root `.claude-plugin/marketplace.json` entry, and merge that only once the packages are on PyPI: the plugins install from `main`.
- **lightrag-memgraph**, **unstructured2graph** and **hygm** use `uv build --out-dir dist` to work around a uv artifact path issue.
- The **mcp-memgraph** Docker image is built from the repo root (the Dockerfile copies both `memgraph-toolbox/` and `integrations/mcp-memgraph/`).
- The mcp-memgraph workflow tags the Docker image with both the version and `latest`.
