#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR uv is not installed or not on PATH.

Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then restart your shell and rerun this script.
EOF
  exit 1
fi

if command -v agent-context-graph >/dev/null 2>&1 \
  && agent-context-graph bootstrap --help >/dev/null 2>&1; then
  exec agent-context-graph bootstrap --runtime codex --connector skills-graph "$@"
fi

exec uvx \
  --from agent-context-graph \
  --with "skills-graph[agent-context-graph]" \
  agent-context-graph bootstrap --runtime codex --connector skills-graph "$@"
