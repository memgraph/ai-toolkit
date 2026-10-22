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

# --python: uv otherwise takes the first Python it finds, often Apple's 3.9, which the packages refuse.
uv tool install --python 3.12 "memgraph-context-graph[mcp]>=0.11.0" \
  --with "skills-graph[memgraph-context-graph]>=0.3.0" \
  --with "actions-graph[memgraph-context-graph]>=0.7.1" \
  --with "sessions-graph[memgraph-context-graph]>=0.11.1" \
  --upgrade \
  --refresh-package memgraph-context-graph \
  --refresh-package skills-graph \
  --refresh-package actions-graph \
  --refresh-package sessions-graph

# setup finds or starts Memgraph, sets your identity and checks the hooks; it skips what is already done.
exec memgraph-context-graph setup claude-code --no-tui "$@"
