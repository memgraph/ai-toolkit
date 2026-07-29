#!/usr/bin/env bash
# Local dev lifecycle for an isolated Memgraph instance used to test
# unstructured2graph and the context-graph family (agent-context-graph,
# actions-graph, skills-graph, sessions-graph) -- and, separately, to point
# the real live Claude Code plugin at that same instance for dogfooding.
#
# See `./scripts/dev-memgraph.sh --help` for the intended workflow.
set -euo pipefail

CONTAINER_NAME="ai-toolkit-dev-memgraph"
HOST_PORT="7688"
IMAGE="memgraph/memgraph-mage:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Deliberately a dedicated port/container, distinct from any other Memgraph
# you may already have running locally for unrelated purposes -- this script
# never touches anything but $CONTAINER_NAME.
LOCAL_MEMGRAPH_URL="bolt://localhost:${HOST_PORT}"
LOCAL_MEMGRAPH_USER=""
LOCAL_MEMGRAPH_PASSWORD=""
LOCAL_MEMGRAPH_DATABASE="memgraph"

CONFIG_DIR="${HOME}/.config/context-graph"
CONFIG_FILE="${CONFIG_DIR}/config.toml"
CONFIG_BACKUP="${CONFIG_DIR}/config.toml.pre-local-test-backup"

ALL_PACKAGES=(unstructured2graph actions-graph agent-context-graph skills-graph sessions-graph)

_HELP="usage: $(basename "$0") <command> [args]

Manage an isolated local Memgraph instance for developing/testing
unstructured2graph and the context-graph family, and for pointing the real
Claude Code plugin at it to see live session data land for real.

Commands:
  up                 Start (or resume) the local Memgraph container (port ${HOST_PORT}).
  down               Stop and remove the local Memgraph container.
  status             Show container state and whether hook config is local or real.
  test [pkg...]      Run test suites against the local container.
                     Default packages: ${ALL_PACKAGES[*]}
  inspect            Print a node-count summary of what is currently in the local graph.
  reconcile [args]    Run 'sessions-graph reconcile' against the local container.
                     Defaults to --pending (sweeps every session marked pending);
                     pass --session <id> to target one. Pulls OPENAI_API_KEY from
                     the environment, or from .env at the repo root if unset.
  hooks-local        Point your REAL, live Claude Code plugin at the local container
                     (backs up your current ~/.config/context-graph/config.toml first).
  hooks-restore       Restore your real hook config from the hooks-local backup.

Typical workflow:
  $(basename "$0") up
  $(basename "$0") test              # proves correctness -- graph ends EMPTY by design
                                       # (the test fixtures clean up before/after each test)
  $(basename "$0") hooks-local        # now drive a REAL Claude Code session to see data land
  $(basename "$0") inspect
  $(basename "$0") reconcile          # runs real entity extraction on pending sessions (costs an LLM call)
  $(basename "$0") inspect
  $(basename "$0") hooks-restore      # point your real hooks back before you forget
  $(basename "$0") down
"

_wait_ready() {
  echo "Waiting for Memgraph to accept Bolt connections on localhost:${HOST_PORT}..."
  for i in $(seq 1 30); do
    if (echo >"/dev/tcp/localhost/${HOST_PORT}") 2>/dev/null; then
      echo "Memgraph is ready."
      return 0
    fi
    echo "Waiting for Memgraph (attempt $i)..."
    sleep 2
  done
  echo "ERROR: Memgraph did not become ready in time." >&2
  docker logs "${CONTAINER_NAME}" 2>&1 | tail -30 >&2 || true
  exit 1
}

_require_container_reachable() {
  if ! (echo >"/dev/tcp/localhost/${HOST_PORT}") 2>/dev/null; then
    echo "ERROR: no Memgraph reachable on localhost:${HOST_PORT}. Run '$(basename "$0") up' first." >&2
    exit 1
  fi
}

# Some packages' own test suites are not perfectly hermetic about
# ~/.config/context-graph/config.toml (a real, discovered bug: a bootstrap
# test in agent-context-graph's suite failed to mock its config-writing call
# and silently overwrote a real, live, credentialed config with test
# artifacts). Belt-and-suspenders: protect whatever is currently in that file
# across every `test` run, independent of whether hooks-local/hooks-restore
# are ever used, so a similar bug anywhere else can't do the same thing.
_TEST_CONFIG_SAFETY_BACKUP="${CONFIG_FILE}.test-run-safety-backup"

_protect_hook_config() {
  if [ -f "${CONFIG_FILE}" ]; then
    cp "${CONFIG_FILE}" "${_TEST_CONFIG_SAFETY_BACKUP}"
    chmod 600 "${_TEST_CONFIG_SAFETY_BACKUP}" 2>/dev/null || true
  fi
}

_restore_hook_config_after_test() {
  if [ -f "${_TEST_CONFIG_SAFETY_BACKUP}" ]; then
    mv "${_TEST_CONFIG_SAFETY_BACKUP}" "${CONFIG_FILE}"
  fi
}

cmd_up() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
      echo "Already running: ${CONTAINER_NAME} on port ${HOST_PORT}"
    else
      echo "Starting existing container ${CONTAINER_NAME}..."
      docker start "${CONTAINER_NAME}" >/dev/null
    fi
  else
    echo "Creating ${CONTAINER_NAME} (${IMAGE}) on port ${HOST_PORT}..."
    docker run -d -p "${HOST_PORT}:7687" --name "${CONTAINER_NAME}" "${IMAGE}" \
      --schema-info-enabled=True --telemetry-enabled=false >/dev/null
  fi
  _wait_ready
}

cmd_down() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
    echo "Removed ${CONTAINER_NAME}."
  else
    echo "${CONTAINER_NAME} does not exist."
  fi
}

cmd_status() {
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container: running (${CONTAINER_NAME}, localhost:${HOST_PORT})"
  elif docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container: stopped (${CONTAINER_NAME})"
  else
    echo "Container: not created"
  fi

  if [ -f "${CONFIG_BACKUP}" ]; then
    echo "Hook config: LOCAL mode (your real config is backed up at ${CONFIG_BACKUP})"
  else
    echo "Hook config: real/production mode (no local-mode backup present)"
  fi
}

cmd_inspect() {
  _require_container_reachable
  MEMGRAPH_URL="${LOCAL_MEMGRAPH_URL}" \
    MEMGRAPH_USER="${LOCAL_MEMGRAPH_USER}" \
    MEMGRAPH_PASSWORD="${LOCAL_MEMGRAPH_PASSWORD}" \
    MEMGRAPH_DATABASE="${LOCAL_MEMGRAPH_DATABASE}" \
    "${REPO_ROOT}/.venv/bin/python3" - <<'PYEOF'
from memgraph_toolbox.api.memgraph import Memgraph

m = Memgraph()
try:
    rows = m.query("MATCH (n) RETURN labels(n) AS labels, count(*) AS count ORDER BY count DESC")
    if not rows:
        print("(empty graph)")
    else:
        print("Node counts by label:")
        for row in rows:
            print(f"  {row['labels']}: {row['count']}")
    total = m.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
    print(f"Total nodes: {total}")
finally:
    m.close()
PYEOF
}

# Resolves OPENAI_API_KEY for cmd_reconcile: prefer whatever is already in the
# environment, otherwise pull it from .env at the repo root. Never prints the
# value anywhere.
_resolve_openai_api_key() {
  if [ -n "${OPENAI_API_KEY:-}" ]; then
    return 0
  fi
  local env_file="${REPO_ROOT}/.env"
  if [ -f "${env_file}" ]; then
    OPENAI_API_KEY="$(grep -E '^OPENAI_API_KEY=' "${env_file}" | head -1 | cut -d'=' -f2-)"
    export OPENAI_API_KEY
  fi
  [ -n "${OPENAI_API_KEY:-}" ]
}

cmd_reconcile() {
  _require_container_reachable

  if ! _resolve_openai_api_key; then
    echo "ERROR: OPENAI_API_KEY is not set and was not found in ${REPO_ROOT}/.env" >&2
    echo "Session reconciliation calls a real LLM (via LightRAG) and needs it." >&2
    exit 1
  fi

  local target=("$@")
  if [ "${#target[@]}" -eq 0 ]; then
    target=(--pending)
  fi

  echo "Running: sessions-graph reconcile ${target[*]}"
  (cd "$REPO_ROOT" && env \
    "MEMGRAPH_URL=${LOCAL_MEMGRAPH_URL}" \
    "MEMGRAPH_USER=${LOCAL_MEMGRAPH_USER}" \
    "MEMGRAPH_PASSWORD=${LOCAL_MEMGRAPH_PASSWORD}" \
    "MEMGRAPH_DATABASE=${LOCAL_MEMGRAPH_DATABASE}" \
    "OPENAI_API_KEY=${OPENAI_API_KEY}" \
    uv run --package sessions-graph --extra reconciliation sessions-graph reconcile "${target[@]}")
}

cmd_test() {
  _require_container_reachable
  _protect_hook_config
  trap _restore_hook_config_after_test EXIT

  local packages=("$@")
  if [ "${#packages[@]}" -eq 0 ]; then
    packages=("${ALL_PACKAGES[@]}")
  fi

  # Only exported for the packages that actually talk to Memgraph in their own
  # tests. agent-context-graph is a pure event-routing/adapter layer -- its
  # own suite is fully mocked and some tests assert on the hardcoded DEFAULT
  # bolt://localhost:7687, so leaking MEMGRAPH_URL into it would break them;
  # it's run with those vars explicitly unset instead, regardless of what's
  # already in the ambient shell environment.
  local with_local_memgraph=(
    env
    "MEMGRAPH_URL=${LOCAL_MEMGRAPH_URL}"
    "MEMGRAPH_USER=${LOCAL_MEMGRAPH_USER}"
    "MEMGRAPH_PASSWORD=${LOCAL_MEMGRAPH_PASSWORD}"
    "MEMGRAPH_DATABASE=${LOCAL_MEMGRAPH_DATABASE}"
  )
  local without_memgraph_env=(env -u MEMGRAPH_URL -u MEMGRAPH_USER -u MEMGRAPH_PASSWORD -u MEMGRAPH_DATABASE)

  local failed=()
  for pkg in "${packages[@]}"; do
    echo ""
    echo "=== Testing ${pkg} ==="
    case "$pkg" in
      unstructured2graph)
        (cd "$REPO_ROOT" && "${with_local_memgraph[@]}" uv run --package unstructured2graph --extra test \
          pytest unstructured2graph/tests/ -v) || failed+=("$pkg")
        ;;
      actions-graph)
        (cd "$REPO_ROOT" && "${with_local_memgraph[@]}" uv run --package actions-graph --extra test --extra agent-context-graph \
          pytest context-graph/actions-graph/tests/ -v) || failed+=("$pkg")
        ;;
      agent-context-graph)
        (cd "$REPO_ROOT" && "${without_memgraph_env[@]}" uv run --package agent-context-graph --extra test \
          pytest context-graph/agent-context-graph/tests/ -v) || failed+=("$pkg")
        ;;
      skills-graph)
        (cd "$REPO_ROOT" && "${with_local_memgraph[@]}" uv run --package skills-graph --extra test --extra agent-context-graph \
          pytest context-graph/skills-graph/tests/ -v) || failed+=("$pkg")
        ;;
      sessions-graph)
        (cd "$REPO_ROOT" && "${with_local_memgraph[@]}" uv run --package sessions-graph --extra test --extra reconciliation --extra agent-context-graph \
          pytest context-graph/sessions-graph/tests/ -v) || failed+=("$pkg")
        ;;
      *)
        echo "Unknown package: $pkg (expected one of: ${ALL_PACKAGES[*]})" >&2
        failed+=("$pkg")
        ;;
    esac
  done

  echo ""
  if [ "${#failed[@]}" -eq 0 ]; then
    echo "All test suites passed: ${packages[*]}"
  else
    echo "FAILED: ${failed[*]}" >&2
    exit 1
  fi
}

cmd_hooks_local() {
  if ! command -v agent-context-graph >/dev/null 2>&1; then
    echo "ERROR: agent-context-graph is not on PATH." >&2
    echo "Install it first, e.g. via context-graph/plugins/agent-context-graph-claude/scripts/bootstrap.sh" >&2
    exit 1
  fi

  mkdir -p "${CONFIG_DIR}"
  if [ -f "${CONFIG_BACKUP}" ]; then
    echo "Backup already exists at ${CONFIG_BACKUP} -- assuming you're already in local mode; re-applying local config."
  elif [ -f "${CONFIG_FILE}" ]; then
    cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
    echo "Backed up your real config to ${CONFIG_BACKUP}"
  else
    echo "No existing config file at ${CONFIG_FILE} -- nothing to back up."
  fi

  agent-context-graph config set memgraph.url "${LOCAL_MEMGRAPH_URL}"
  agent-context-graph config set memgraph.user ""
  agent-context-graph config set memgraph.password ""
  agent-context-graph config set memgraph.database "${LOCAL_MEMGRAPH_DATABASE}"

  echo ""
  echo "Live hooks now point at the local instance (${LOCAL_MEMGRAPH_URL})."
  echo "When you're done, restore your real config with:"
  echo "  $(basename "$0") hooks-restore"
}

cmd_hooks_restore() {
  if [ ! -f "${CONFIG_BACKUP}" ]; then
    echo "ERROR: no backup found at ${CONFIG_BACKUP} -- nothing to restore." >&2
    echo "(Are you sure hooks-local was run, and hasn't already been restored?)" >&2
    exit 1
  fi
  mv "${CONFIG_BACKUP}" "${CONFIG_FILE}"
  echo "Restored your real hook config from backup."
  agent-context-graph config show 2>/dev/null || true
}

main() {
  local cmd="${1:-}"
  if [ "$#" -gt 0 ]; then
    shift
  fi
  case "$cmd" in
    up) cmd_up ;;
    down) cmd_down ;;
    status) cmd_status ;;
    inspect) cmd_inspect ;;
    test) cmd_test "$@" ;;
    reconcile) cmd_reconcile "$@" ;;
    hooks-local) cmd_hooks_local ;;
    hooks-restore) cmd_hooks_restore ;;
    -h | --help | "") echo "$_HELP" ;;
    *)
      echo "Unknown command: $cmd" >&2
      echo "$_HELP" >&2
      exit 2
      ;;
  esac
}

main "$@"
