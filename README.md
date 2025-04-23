# Memgraph Toolbox

This should be a make file

### Dependencies for running:

```
uv pip install -e memgraph-toolbox
uv pip install -e integrations/langchain-memgraph
uv pip install -e integrations/mcp-memgraph
```

### Dependencies for tests:

```
uv pip install -e memgraph-toolbox[test]
uv pip install -e integrations/langchain-memgraph[test]
uv pip install -e integrations/mcp-memgraph[test]
# In zsh it is
uv pip install -e memgraph-toolbox"[test]"
uv pip install -e integrations/langchain-memgraph"[test]"
uv pip install -e integrations/mcp-memgraph"[test]"
```

### Test for core

```
pytest -s memgraph-toolbox/src/memgraph-toolbox/tests
pytest -s integrations/langchain-memgraph/tests
pytest -s integrations/mcp-memgraph/tests

```
