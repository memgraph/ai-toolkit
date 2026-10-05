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

uv tool install "agent-context-graph[mcp]>=0.3.1" \
  --with "skills-graph[agent-context-graph]>=0.1.3" \
  --with "actions-graph[agent-context-graph]>=0.3.0" \
  --with "sessions-graph[agent-context-graph]>=0.6.0" \
  --upgrade \
  --refresh-package agent-context-graph \
  --refresh-package skills-graph \
  --refresh-package actions-graph \
  --refresh-package sessions-graph

exec agent-context-graph bootstrap --runtime claude-code --connector skills-graph --connector actions-graph --connector sessions-graph --no-reinstall "$@"
