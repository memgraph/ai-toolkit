#!/usr/bin/env bash
set -euo pipefail

if ! command -v agent-context-graph >/dev/null 2>&1; then
  echo "agent-context-graph not found on PATH" >&2
  exit 1
fi

printf '{"hook_event_name":"Stop","session_id":"doctor"}' \
  | agent-context-graph hook run codex --connector skills-graph --strict

echo
echo "agent-context-graph Codex hook command is reachable"
