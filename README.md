# Memgraph AI Toolkit

A unified mono-repo for building and integrating AI-powered graph tools on top of [Memgraph](https://memgraph.com/).  
This repository contains the following libraries:

1. **memgraph-toolbox**  
   Core Python utilities and CLI tools for querying and analyzing a Memgraph database.

2. **integrations/langchain-memgraph**  
   A LangChain integration package, exposing Memgraph operations as LangChain tools and toolkits.

3. **integrations/mcp-memgraph**  
   An MCP (Model Context Protocol) server implementation, exposing Memgraph tools over a lightweight STDIO protocol.

## Usage examples

For individual examples on how to use the toolbox, LangChain, or MCP, refer to our documentation:

- [Langchain examples](https://memgraph.com/docs/ai-ecosystem/integrations#langchain)
- [MCP examples](https://memgraph.com/docs/ai-ecosystem/integrations#model-context-protocol-mcp)

## Running tests locally

To run the tests locally, ensure you have a local Memgraph MAGE instance running on port 7687:

`docker run -p 7687:7687 --name memgraph memgraph/memgraph-mage:latest`

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
