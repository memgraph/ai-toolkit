# AI toolkit

This should be a make file

### Dependencies for running:

```
uv pip install -e core
uv pip install -e integrations/langchain-memgraph
uv pip install -e integrations/mcp-memgraph
```

### Dependencies for tests:

```
uv pip install -e core[test]
uv pip install -e integrations/langchain-memgraph[test]
uv pip install -e integrations/mcp-memgraph[test]
# In zsh it is
uv pip install -e core"[test]"
```

### Test for core

```
pytest -s core/src/core/tests
pytest -s integrations/langchain-memgraph/tests
pytest -s integrations/mcp-memgraph/tests

```
