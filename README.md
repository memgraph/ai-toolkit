# Memgraph AI Toolkit

A unified mono-repo for building and integrating AI-powered graph tools on top of [Memgraph](https://memgraph.com/).  
This repository contains:

1. **memgraph-toolbox**  
   Core Python utilities and CLI tools for querying and analyzing a Memgraph database.

2. **integrations/langchain-memgraph**  
   A LangChain integration package, exposing Memgraph operations as LangChain tools and toolkits.

3. **integrations/mcp-memgraph**  
   An MCP (Model Context Protocol) server implementation, exposing Memgraph tools over a lightweight STDIO protocol.

## Usage examples

For individual examples on how to use the toolbox, LangChain or MCP, take a look at our documentation:

- [Langchain examples](https://memgraph.com/docs/ai-ecosystem/integrations#langchain)
- [MCP examples](https://memgraph.com/docs/ai-ecosystem/integrations#model-context-protocol-mcp)

## Running tests locally

In order to run tests locally,

```
# Core toolbox tests
uv pip install -e memgraph-toolbox[test]
pytest -s memgraph-toolbox/src/memgraph_toolbox/tests

# LangChain integration tests
uv pip install -e integrations/langchain-memgraph[test]
pytest -s integrations/langchain-memgraph/tests

# MCP integration tests
uv pip install -e integrations/mcp-memgraph[test]
pytest -s integrations/mcp-memgraph/tests
```

If you are running on MacOS in zsh, add `""` to the command `uv pip install -e memgraph-toolbox"[test]"`
