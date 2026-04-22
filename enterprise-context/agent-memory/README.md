# Agent Memory Tests

Integration tests for memory frameworks with Memgraph. Each test suite validates that a memory framework works correctly using Memgraph as its graph backend.

## Frameworks

| Framework                                                                                   | Description                                                                         | Upstream                                  |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------- |
| [Cognee](https://github.com/topoteretes/cognee-community/tree/main/packages/graph/memgraph) | Knowledge graph adapter for the Cognee framework                                    | `cognee-community-graph-adapter-memgraph` |
| [Neo4j Agent Memory](https://github.com/neo4j-labs/agent-memory)                            | Graph-native memory system for AI agents (testing Bolt-compatibility with Memgraph) | `neo4j-agent-memory`                      |

## Prerequisites

- Python >= 3.10
- A running Memgraph instance (default: `bolt://localhost:7687`)

```bash
docker run -p 7687:7687 memgraph/memgraph-mage:latest --schema-info-enabled=True
```

## Setup

Install all test dependencies:

```bash
uv sync --all-extras
```

Or install only a specific framework:

```bash
uv sync --extra cognee
uv sync --extra neo4j-agent-memory
```

## Running Tests

Run all tests:

```bash
uv run pytest
```

Run tests for a specific framework:

```bash
uv run pytest -m cognee
uv run pytest -m neo4j_agent_memory
```
