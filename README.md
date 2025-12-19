# Memgraph AI Toolkit

A unified mono-repo for integrating AI-powered graph tools on top of [Memgraph](https://memgraph.com/).  
This repository contains the following libraries:

1. [**memgraph-toolbox**](/memgraph-toolbox/)
   Core Python utilities and CLI tools for querying and analyzing a Memgraph database. The package is available on the [PyPi](https://pypi.org/project/memgraph-toolbox/)

2. [**langchain-memgraph**](/integrations/langchain-memgraph/)
   A LangChain integration package, exposing Memgraph operations as LangChain tools and toolkits. The package is available on the [PyPi](https://pypi.org/project/langchain-memgraph/)

3. [**mcp-memgraph**](/integrations/mcp-memgraph/)
   An MCP (Model Context Protocol) server implementation, exposing Memgraph tools over a lightweight STDIO protocol. The package is available on the [PyPi](https://pypi.org/project/mcp-memgraph/)

4. [**unstructured2graph**](/unstructured2graph/)
   A library that enables you to convert any data into a graph within Memgraph, enabling various GraphRAG techniques for enhanced information retrieval.

5. [**agents**](/agents/sql2graph/)
   An intelligent database migration agent that automates the process of migrating from MySQL or Postgresql to Memgraph. Features automated schema analysis, intelligent graph modeling with interactive refinement, and data migration with validation.

## Usage examples

For individual examples on how to use the toolbox, LangChain, MCP, or agents, refer to our documentation:

- [Langchain examples](https://memgraph.com/docs/ai-ecosystem/integrations#langchain)
- [MCP examples](https://memgraph.com/docs/ai-ecosystem/integrations#model-context-protocol-mcp)
- [unstructured2graph examples](https://memgraph.com/docs/ai-ecosystem/unstructured2graph)
- [SQL2Graph migration examples](https://memgraph.com/docs/ai-ecosystem/agents#sql2graph-agent)

## Developing locally

You can build and test each package directly from your repo. First, start a Memgraph MAGE instance with schema info enabled:

```bash
docker run -p 7687:7687 \
  --name memgraph \
  memgraph/memgraph-mage:latest \
  --schema-info-enabled=true
```

Once Memgraph is running, install any package in “editable” mode and run its test suite. For example, to test the core toolbox:

```
uv pip install -e memgraph-toolbox[test]
pytest -s memgraph-toolbox/src/memgraph_toolbox/tests
```

### Core tests

To test the core toolbox, just run:

```
uv pip install -e memgraph-toolbox[test]
pytest -s memgraph-toolbox/src/memgraph_toolbox/tests
```

### Langchain integration tests

To run the LangChain tests, create a .env file with your OPENAI_API_KEY, as the tests depend on LLM calls:

```
uv pip install -e integrations/langchain-memgraph[test]
pytest -s integrations/langchain-memgraph/tests
```

### MCP integration tests

```
uv pip install -e integrations/mcp-memgraph[test]
pytest -s integrations/mcp-memgraph/tests
```

### Agent integration tests

```
uv pip install -e integrations/agents[test]
pytest -s integrations/agents/tests
```

To run a complete migration workflow with the agent:

```
cd integrations/agents
uv run main.py
```

**Note:** The agent requires both MySQL and Memgraph connections. Set up your environment variables in `.env` based on `.env.example`.

If you are running any test on MacOS in zsh, add `""` to the command:

```
uv pip install -e memgraph-toolbox"[test]"
```
