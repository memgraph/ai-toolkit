FROM python:3.13-slim

WORKDIR /app

COPY memgraph-toolbox /app/memgraph-toolbox
COPY integrations/mcp-memgraph /app/integrations/mcp-memgraph

RUN pip install --upgrade pip uv
RUN uv pip install --system -e /app/memgraph-toolbox
RUN uv pip install --system -e /app/integrations/mcp-memgraph

WORKDIR /app/integrations/mcp-memgraph/src

ENV FASTMCP_HOST=0.0.0.0
ENV MCP_TRANSPORT=streamable-http
ENV MEMGRAPH_URL=bolt://host.docker.internal:7687

EXPOSE 8000

CMD ["python", "-m", "mcp_memgraph.main"]
