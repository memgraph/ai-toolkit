#!/usr/bin/env bash
# One-command setup for Context Graph on Claude Code: start Memgraph if
# nothing is reachable, register the plugin marketplace, install the plugin
# (this is what actually wires hooks into Claude Code -- `bootstrap` alone
# does not), install the agent-context-graph CLI with all three connectors,
# set your identity, and verify with `doctor`.
#
# Usage:
#   ./context-graph/scripts/install.sh
#   curl -fsSL https://raw.githubusercontent.com/memgraph/ai-toolkit/main/context-graph/scripts/install.sh | bash
#
# Env overrides:
#   AGENT_CONTEXT_GRAPH_USER_ID   identity to record (default: git user.name, else $USER)
#   MEMGRAPH_HOST / MEMGRAPH_PORT default localhost:7687
#   SKIP_MEMGRAPH=1               don't start a local Memgraph even if none is reachable
#   SKIP_UV_INSTALL=1             don't auto-install uv if missing (just fail with instructions)
#
# Codex is NOT covered: unlike Claude Code, Codex has no non-interactive
# plugin-install command today (its marketplace add is scriptable, but
# enabling the plugin itself is a manual step in the Codex UI). Run this
# script anyway to get Memgraph + the CLI + connectors installed, then finish
# Codex's plugin step by hand -- see ../plugins/agent-context-graph-codex/README.md.

set -euo pipefail

RUNTIME="claude-code"
CONNECTORS=(skills-graph actions-graph sessions-graph)
MEMGRAPH_HOST="${MEMGRAPH_HOST:-localhost}"
MEMGRAPH_PORT="${MEMGRAPH_PORT:-7687}"
CONTAINER_NAME="context-graph-memgraph"
MAGE_IMAGE="memgraph/memgraph-mage:latest"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

bolt_reachable() {
  (exec 3<>"/dev/tcp/${MEMGRAPH_HOST}/${MEMGRAPH_PORT}") 2>/dev/null
}

# A plain TCP accept happens well before Memgraph can actually complete a Bolt
# handshake -- the MAGE image in particular preloads heavy ML query-module
# dependencies after the socket is already listening. Once we start the
# container ourselves, gate readiness on a real query via the mgconsole
# bundled in the image, not just the socket.
bolt_query_ready() {
  docker exec "$CONTAINER_NAME" mgconsole --host 127.0.0.1 --port 7687 <<<"RETURN 1;" >/dev/null 2>&1
}

# ---- 1. Memgraph ------------------------------------------------------------
log "Checking Memgraph at bolt://${MEMGRAPH_HOST}:${MEMGRAPH_PORT}"
if bolt_reachable; then
  echo "Already reachable -- using it as-is."
elif [[ "${SKIP_MEMGRAPH:-0}" == "1" ]]; then
  fail "memgraph: not reachable and SKIP_MEMGRAPH=1 -- start one and rerun."
else
  command -v docker >/dev/null 2>&1 || fail "docker is required to auto-start Memgraph: https://docs.docker.com/get-docker/ (or start your own and rerun with SKIP_MEMGRAPH=1)"
  echo "Nothing reachable -- starting a local Memgraph ($MAGE_IMAGE)."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER_NAME" \
    -p "${MEMGRAPH_PORT}:7687" -p 7444:7444 \
    "$MAGE_IMAGE" --schema-info-enabled=True >/dev/null
  ready=0
  for _ in $(seq 1 90); do
    if bolt_query_ready; then ready=1; break; fi
    sleep 2
  done
  [[ "$ready" == "1" ]] || { docker logs "$CONTAINER_NAME" || true; fail "memgraph did not become ready in time"; }
  echo "Memgraph is up on bolt://${MEMGRAPH_HOST}:${MEMGRAPH_PORT} (container: $CONTAINER_NAME)"
fi

# ---- 2. uv -------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  if [[ "${SKIP_UV_INSTALL:-0}" == "1" ]]; then
    fail "uv not found and SKIP_UV_INSTALL=1: https://astral.sh/uv/install.sh"
  fi
  log "Installing uv (manages Python for the agent-context-graph tool)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv install did not put 'uv' on PATH -- open a new shell and rerun."
fi
echo "OK uv: $(command -v uv)"

# ---- 3. Claude Code plugin (marketplace + install) ---------------------------
# This is the step the plain `bootstrap` command cannot do for you: it's what
# wires hooks into Claude Code so sessions actually get captured. Without it,
# `doctor` below will look all-green but no real session will write anything.
command -v claude >/dev/null 2>&1 || fail "claude (Claude Code CLI) not found on PATH -- install Claude Code first."

log "Registering the ai-toolkit plugin marketplace"
if claude plugin marketplace list --json 2>/dev/null | grep -q '"context-graph-plugins"'; then
  echo "Already registered."
else
  claude plugin marketplace add memgraph/ai-toolkit --sparse .claude-plugin
fi

log "Installing the context-graph plugin"
if claude plugin list --json 2>/dev/null | grep -q '"context-graph"'; then
  echo "Already installed."
else
  claude plugin install context-graph@context-graph-plugins -y
fi

# ---- 4. agent-context-graph CLI + connectors ---------------------------------
log "Installing agent-context-graph with skills-graph, actions-graph, sessions-graph"
uv tool install "agent-context-graph>=0.1.9" \
  --with "skills-graph[agent-context-graph]>=0.1.3" \
  --with "actions-graph[agent-context-graph]>=0.1.1" \
  --with "sessions-graph[agent-context-graph]>=0.1.2" \
  --upgrade \
  --refresh-package agent-context-graph \
  --refresh-package skills-graph \
  --refresh-package actions-graph

# uv tool shims land in `uv tool dir --bin` (usually ~/.local/bin), which may
# not be on PATH yet even when uv itself came from elsewhere (Homebrew, pipx).
export PATH="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin"):$PATH"
command -v agent-context-graph >/dev/null 2>&1 || fail "agent-context-graph installed but not on PATH -- add $(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin") to PATH and rerun."

log "Bootstrapping ($RUNTIME)"
connector_flags=()
for c in "${CONNECTORS[@]}"; do connector_flags+=(--connector "$c"); done
MEMGRAPH_HOST="$MEMGRAPH_HOST" MEMGRAPH_PORT="$MEMGRAPH_PORT" \
  agent-context-graph bootstrap --runtime "$RUNTIME" "${connector_flags[@]}" --no-reinstall

# ---- 5. Identity --------------------------------------------------------------
USER_ID="${AGENT_CONTEXT_GRAPH_USER_ID:-$(git config --get user.name 2>/dev/null || true)}"
USER_ID="${USER_ID:-$USER}"
log "Setting identity.user_id = ${USER_ID}"
agent-context-graph config set identity.user_id "$USER_ID"

# ---- 6. Verify -----------------------------------------------------------------
log "Verifying"
agent-context-graph doctor --runtime "$RUNTIME" "${connector_flags[@]}"

echo ""
echo -e "\033[1;32m✓ Context Graph is live.\033[0m Use Claude Code normally -- every session now"
echo "writes Actions/Skills/Memories to bolt://${MEMGRAPH_HOST}:${MEMGRAPH_PORT}."
echo ""
echo "Explore the graph: docker run -d -p 3000:3000 --add-host=host.docker.internal:host-gateway \\"
echo "  -e QUICK_CONNECT_MG_HOST=host.docker.internal -e QUICK_CONNECT_MG_PORT=${MEMGRAPH_PORT} memgraph/lab:latest"
echo "Docs: https://github.com/memgraph/ai-toolkit/tree/main/context-graph"
