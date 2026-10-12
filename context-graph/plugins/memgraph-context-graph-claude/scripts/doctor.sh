#!/usr/bin/env bash
set -euo pipefail

if ! command -v memgraph-context-graph >/dev/null 2>&1; then
  echo "memgraph-context-graph not found on PATH" >&2
  exit 1
fi

exec memgraph-context-graph doctor \
  --runtime claude-code \
  --connector skills-graph \
  --connector actions-graph \
  --connector sessions-graph
