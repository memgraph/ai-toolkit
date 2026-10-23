# Memgraph AI Toolkit

Python packages for building AI applications with [Memgraph](https://memgraph.com/): agent memory, knowledge graphs, GraphRAG, and framework integrations.

Use the packages independently, or combine them to bring documents, databases, and agent activity into a graph.

[Documentation](https://memgraph.com/docs/ai-ecosystem) · [Examples](#choose-your-starting-point) · [Discord](https://discord.gg/memgraph) · [Issues](https://github.com/memgraph/ai-toolkit/issues)

## Choose your starting point

| I want to… | Start here |
|---|---|
| Give coding agents memory across sessions and tools | [Context Graph](context-graph/) |
| Extract entities and relationships from documents | [unstructured2graph](unstructured2graph/) |
| Query Memgraph in natural language or build a LangChain agent | [langchain-memgraph](integrations/langchain-memgraph/) |
| Connect an MCP client to Memgraph | [mcp-memgraph](integrations/mcp-memgraph/) |
| Use Memgraph as LightRAG storage | [lightrag-memgraph](integrations/lightrag-memgraph/) |
| Migrate a MySQL or PostgreSQL database into a graph | [sql2graph](agents/sql2graph/) |
| Work directly with Memgraph from Python | [memgraph-toolbox](memgraph-toolbox/) |
| Define and validate a graph's entity and relation types | [hygm](hygm/) |

## Agent memory with Context Graph

Keep what your agents learn in a database you control. Context Graph captures sessions, tool calls, and skill usage, stores explicit memory, and retrieves relevant past conversations for the next session.

It supports **Claude Code, Codex, Copilot CLI, Cursor, OpenCode, Antigravity CLI, and Grok Build**, plus applications built with the Claude Agent SDK or OpenAI Agents SDK.

```bash
curl -fsSL https://raw.githubusercontent.com/memgraph/ai-toolkit/main/context-graph/scripts/install.sh | bash
```

Setup installs the CLI and connectors, starts its own Memgraph container or connects to one you specify, and wires the agents you select. Docker is needed if setup starts Memgraph for you.

[Get started with Context Graph](context-graph/#getting-started) · [Components](context-graph/#the-components) · [Recall benchmarks](context-graph/eval/#official-numbers)

## Knowledge graphs from documents

`unstructured2graph` parses documents and extracts entities and relationships. Its default backend, **GLiNER2**, runs locally without an LLM API key; LightRAG is available as an LLM-based alternative.

Start Memgraph:

```bash
docker run -p 7687:7687 --name memgraph \
  memgraph/memgraph-mage:latest --schema-info-enabled=true
```

Install the package:

```bash
pip install unstructured2graph
```

Extract a graph from text:

```python
import asyncio
from memgraph_toolbox.api.memgraph import Memgraph
from unstructured2graph import from_texts
from unstructured2graph.gliner2_backend import GLiNER2Backend


async def main():
    await from_texts(
        texts=["Alice works for Acme in London."],
        memgraph=Memgraph(),
        extraction_backend=GLiNER2Backend(),
        enforce_ontology=True,
    )


asyncio.run(main())
```

The extraction model downloads on first use. See [unstructured2graph](unstructured2graph/) for files, URLs, document-format extras, and custom ontologies.

## Framework integrations

Install only the integration you need:

```bash
pip install langchain-memgraph  # graph store, QA chain, and agent toolkit
pip install mcp-memgraph       # MCP server for database queries and schema tools
pip install lightrag-memgraph  # LightRAG storage backends
```

Each package's README covers its connection settings and examples. LLM-based features need credentials for the model provider you choose; local extraction, capture, and recall do not require an LLM key.

## Developing locally

This is a Python 3.10+ [uv](https://docs.astral.sh/uv/) workspace. From the repository root:

```bash
uv venv
uv sync --all-packages --all-extras
uv run ruff check .
uv run ruff format --check .
uv run ty check .
```

Context Graph, HyGM, and unstructured2graph use an isolated test Memgraph managed by the development script:

```bash
./scripts/dev-memgraph.sh test
./scripts/dev-memgraph.sh test sessions-graph  # one package
./scripts/dev-memgraph.sh test-down           # remove the test container
```

For other packages, see their READMEs and the [test workflow](.github/workflows/tests.yaml) for dependencies and credentials. The [development script](scripts/dev-memgraph.sh) also manages a separate exploration database for dogfooding.

## Community

Ask questions on [Discord](https://discord.gg/memgraph), report bugs or request features in [GitHub Issues](https://github.com/memgraph/ai-toolkit/issues), and explore the [Memgraph AI documentation](https://memgraph.com/docs/ai-ecosystem).

[MIT license](LICENSE)
