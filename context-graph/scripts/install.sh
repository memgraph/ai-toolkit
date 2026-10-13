#!/usr/bin/env bash
# Install Context Graph for your coding agents. This script only gets Python's
# tooling in place; `memgraph-context-graph setup` does the rest: it finds or
# starts Memgraph, wires every agent it finds, sets your identity and checks
# the result. On a terminal it is interactive; otherwise it prints its plan
# and runs it.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/memgraph/ai-toolkit/main/context-graph/scripts/install.sh | bash
#   ... | bash -s -- codex --memgraph-url bolt://my-host:7687    # any `setup` flags
#   ./context-graph/scripts/install.sh --help                   # setup's own help
#
# Undo with `memgraph-context-graph uninstall`.

set -euo pipefail

fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (it manages Python for the memgraph-context-graph tool)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv was installed but isn't on PATH; open a new shell and run this again."
fi

# The same set setup checks for (memgraph_context_graph/setup/tool.py) and the plugins' bootstrap installs.
echo "Installing memgraph-context-graph with skills-graph, actions-graph and sessions-graph"
uv tool install "memgraph-context-graph[mcp]>=0.6.0" \
  --with "skills-graph[memgraph-context-graph]>=0.2.1" \
  --with "actions-graph[memgraph-context-graph]>=0.4.1" \
  --with "sessions-graph[memgraph-context-graph]>=0.8.0" \
  --upgrade

# uv's tool shims live in `uv tool dir --bin` (usually ~/.local/bin), which may not be on PATH yet.
export PATH="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin"):$PATH"
command -v memgraph-context-graph >/dev/null 2>&1 ||
  fail "memgraph-context-graph was installed but isn't on PATH; add $(uv tool dir --bin) to PATH and run this again."

# Piped into bash, stdin is the rest of this script, so a TUI would draw but never receive a key.
# Reattach the terminal when there is one; without one, setup runs its plain, non-interactive path.
if (exec </dev/tty) 2>/dev/null; then
  exec memgraph-context-graph setup "$@" </dev/tty
fi
exec memgraph-context-graph setup --no-tui "$@"
