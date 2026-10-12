#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR uv is not installed or not on PATH.

Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh

uv manages Python for the tool. If uv-managed Python downloads are blocked, install Python 3.10+.

Then restart your shell and rerun this script.
EOF
  exit 1
fi

uv tool install "memgraph-context-graph[mcp]>=0.4.0" \
  --with "skills-graph[memgraph-context-graph]>=0.2.1" \
  --with "actions-graph[memgraph-context-graph]>=0.4.1" \
  --with "sessions-graph[memgraph-context-graph]>=0.7.1" \
  --upgrade \
  --refresh-package memgraph-context-graph \
  --refresh-package skills-graph \
  --refresh-package actions-graph \
  --refresh-package sessions-graph

exec memgraph-context-graph bootstrap --runtime codex --connector skills-graph --connector actions-graph --connector sessions-graph --no-reinstall "$@"
