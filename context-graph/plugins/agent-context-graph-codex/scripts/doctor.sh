#!/usr/bin/env bash
set -euo pipefail

if ! command -v agent-context-graph >/dev/null 2>&1; then
  echo "agent-context-graph not found on PATH" >&2
  exit 1
fi

exec agent-context-graph doctor \
  --runtime codex \
  --connector skills-graph \
  --connector actions-graph \
  --connector sessions-graph
