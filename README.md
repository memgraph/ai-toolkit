# Memgraph AI Toolkit

A unified mono-repo for integrating AI-powered graph tools on top of [Memgraph](https://memgraph.com/).  
This repository contains the following libraries:

1. [**memgraph-toolbox**](/memgraph-toolbox/)
   Core Python utilities and CLI tools for querying and analyzing a Memgraph database. The package is available on the [PyPi](https://pypi.org/project/memgraph-toolbox/)

2. [**langchain-memgraph**](/integrations/langchain-memgraph/)
   A LangChain integration package, exposing Memgraph operations as LangChain tools and toolkits. The package is available on the [PyPi](https://pypi.org/project/langchain-memgraph/)

3. [**mcp-memgraph**](/integrations/mcp-memgraph/)
   An MCP (Model Context Protocol) server implementation, exposing Memgraph tools over a lightweight STDIO protocol. The package is available on the [PyPi](https://pypi.org/project/mcp-memgraph/)

## Usage examples

For individual examples on how to use the toolbox, LangChain, or MCP, refer to our documentation:

- [Langchain examples](https://memgraph.com/docs/ai-ecosystem/integrations#langchain)
- [MCP examples](https://memgraph.com/docs/ai-ecosystem/integrations#model-context-protocol-mcp)

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

If you are running any test on MacOS in zsh, add `""` to the command:

```
uv pip install -e memgraph-toolbox"[test]"
```
