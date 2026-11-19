"""A docker-compose file for a spaces server: a licensed Memgraph with MAGE, and the shared-space worker."""

from __future__ import annotations

#: Headroom between Memgraph's own limit and the container's: what Memgraph doesn't count, such as the embedding model.
_UNCOUNTED_MIB = 512
SPEC_IN_CONTAINER = "/spec/spaces.toml"


def compose_file(*, bind: str, port: int, memory_gib: int, spec: str) -> str:
    """The compose file's text.

    *bind* is the host address Memgraph's Bolt port listens on (e.g. a private network's), *port* that port,
    *memory_gib* the container's memory, and *spec* the host path of the spaces spec the worker reconciles.

    Raises:
        ValueError: *memory_gib* leaves Memgraph no memory once the uncounted headroom is taken.
    """
    from importlib.metadata import version

    from memgraph_context_graph.setup.memgraph import IMAGE

    memgraph_mib = memory_gib * 1024 - _UNCOUNTED_MIB
    if memgraph_mib < 1024:
        raise ValueError("--memory-gib must be at least 2: recall's embedding model alone needs about a gigabyte")
    worker = f"context-graph-spaces=={version('context-graph-spaces')}"
    return f"""\
# A Context Graph spaces server, written by `context-graph-spaces compose`.
#
#   1. Put the licence in .env next to this file:
#        MEMGRAPH_ENTERPRISE_LICENSE=...
#        MEMGRAPH_ORGANIZATION_NAME=...
#   2. docker compose up -d
#   3. context-graph-spaces --url bolt://<this host>:{port} init        (the first admin login)
#   4. context-graph-spaces ... apply spaces.toml                     (prints CG_SERVICE_PASSWORD once)
#   5. Add CG_SERVICE_PASSWORD=... to .env, then: docker compose --profile worker up -d
services:
  memgraph:
    image: {IMAGE}
    restart: unless-stopped
    environment:
      MEMGRAPH_ENTERPRISE_LICENSE: ${{MEMGRAPH_ENTERPRISE_LICENSE:?put the licence in .env}}
      MEMGRAPH_ORGANIZATION_NAME: ${{MEMGRAPH_ORGANIZATION_NAME:?put the organization name in .env}}
    ports:
      - "{bind}:{port}:7687"
    volumes:
      - memgraph-data:/var/lib/memgraph
      - memgraph-log:/var/log/memgraph
    # Memgraph sizes its limit from the host's RAM, not the container's; without --memory-limit below mem_limit
    # the container is killed instead of the query that would overflow it being refused.
    mem_limit: {memory_gib}g
    command: ["--schema-info-enabled=True", "--memory-limit={memgraph_mib}", "--also-log-to-stderr=true"]

  # Writes the derived graph (chunks, entities, embeddings) members can't write in shared spaces.
  # It needs the password apply prints once, so it starts only with --profile worker.
  worker:
    profiles: ["worker"]
    image: ghcr.io/astral-sh/uv:python3.12-bookworm-slim
    restart: unless-stopped
    depends_on: [memgraph]
    environment:
      CG_SERVICE_PASSWORD: ${{CG_SERVICE_PASSWORD:-}}
      # PyPI's Linux torch is the CUDA build (about 7 GB with its nvidia wheels); the worker runs on CPU (about 1.6 GB).
      UV_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu
      UV_INDEX_STRATEGY: unsafe-best-match
    volumes:
      - {spec}:{SPEC_IN_CONTAINER}:ro
      # The package, torch and the GLiNER2 model, downloaded once instead of on every restart.
      - worker-cache:/root/.cache
    entrypoint: ["uvx", "--from", "{worker}", "context-graph-spaces"]
    command: ["--url", "bolt://memgraph:7687", "reconcile", "{SPEC_IN_CONTAINER}", "--watch"]

volumes:
  memgraph-data:
  memgraph-log:
  worker-cache:
"""
