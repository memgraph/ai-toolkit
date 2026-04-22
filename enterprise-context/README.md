# Enterprise Context

A collection of standalone components for enriching enterprise data with graph-powered context. Each component is an independent library or CLI tool with its own dependencies and configuration.

## Components

| Component                      | Description                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| [sic-agent](./sic-agent)       | SIC Classification MCP Server using Memgraph vector search                               |
| [agent-memory](./agent-memory) | Integration tests for memory frameworks (Cognee, Mem0, neo4j-agent-memory) with Memgraph |

## Structure

Each component lives in its own subdirectory and includes:

- `pyproject.toml` — project metadata and dependencies
- `README.md` — component-specific documentation
- `LICENSE` — license file

Components are independent and can be installed/run separately.

## Getting Started

Navigate into any component directory and follow its README for setup instructions.
