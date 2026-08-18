#!/usr/bin/env bash
# Local dev lifecycle for exploring/dogfooding unstructured2graph and the
# context-graph family (agent-context-graph, actions-graph, skills-graph,
# sessions-graph) against an isolated local Memgraph -- including pointing
# the real live Claude Code plugin at it. That exploration instance's data
# persists until you explicitly `down` it. `test` is a separate concern with
# its own disposable container, so running the automated suites can never
# wipe whatever you're exploring.
#
# See `./scripts/dev-memgraph.sh --help` for the intended workflow.
set -euo pipefail

CONTAINER_NAME="ai-toolkit-dev-memgraph"
HOST_PORT="7688"
IMAGE="memgraph/memgraph-mage:latest"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Deliberately a dedicated port/container, distinct from any other Memgraph
# you may already have running locally for unrelated purposes -- this script
# never touches anything but $CONTAINER_NAME with `up`/`down`. This is the
# exploration instance: `hooks-local` points your real Claude Code plugin at
# it, and its data is meant to persist across everything except an explicit
# `down` -- nothing in this script ever runs cleanup queries against it.
LOCAL_MEMGRAPH_URL="bolt://localhost:${HOST_PORT}"
LOCAL_MEMGRAPH_USER=""
LOCAL_MEMGRAPH_PASSWORD=""
LOCAL_MEMGRAPH_DATABASE="memgraph"

# `test` runs against a SEPARATE, disposable container/port, never the one
# above. Community Edition Memgraph has no multi-database/multi-tenancy (that
# needs an Enterprise license -- `SHOW DATABASES` errors without one), so this
# is the only way to give the test suites' own hermetic per-test cleanup
# (every package's e2e conftest.py does MATCH (n) DETACH DELETE n before/after
# each test, by design, matching skills-graph/actions-graph's own pre-existing
# convention) a place to run without ever being able to wipe whatever you're
# exploring/dogfooding on the instance above. Managed transparently by `test`
# itself -- you never need to think about it.
TEST_CONTAINER_NAME="ai-toolkit-test-memgraph"
TEST_HOST_PORT="7689"
TEST_MEMGRAPH_URL="bolt://localhost:${TEST_HOST_PORT}"
TEST_MEMGRAPH_USER=""
TEST_MEMGRAPH_PASSWORD=""
TEST_MEMGRAPH_DATABASE="memgraph"

CONFIG_DIR="${HOME}/.config/context-graph"
CONFIG_FILE="${CONFIG_DIR}/config.toml"
CONFIG_BACKUP="${CONFIG_DIR}/config.toml.pre-local-test-backup"

ALL_PACKAGES=(unstructured2graph actions-graph agent-context-graph skills-graph sessions-graph)

_HELP="usage: $(basename "$0") <command> [args]

Runs an isolated local Memgraph (port ${HOST_PORT}) to explore/dogfood
unstructured2graph and the context-graph family against -- its data persists
until you explicitly \`down\` it. \`test\` is a separate concern: it runs
against its own disposable container (port ${TEST_HOST_PORT}), so it can
never wipe whatever you're exploring on the main one.

Commands:
  up                 Start (or resume) the main local Memgraph container.
  down               Stop and remove the main local Memgraph container.
  status             Show both containers' state and whether hook config is local or real.
  inspect            Print a node-count summary of what is currently in the main graph.
  reconcile [args]    Run 'sessions-graph reconcile' against the main container.
                     Defaults to --pending (sweeps every session marked pending);
                     pass --session <id> to target one. Pulls OPENAI_API_KEY from
                     the environment, or from .env at the repo root if unset.
  hooks-local        Point your REAL, live Claude Code plugin at the main container
                     (backs up your current ~/.config/context-graph/config.toml first).
  hooks-restore       Restore your real hook config from the hooks-local backup.
  test-graph-model    Drive a real, non-interactive Claude Code session that spawns
                     exactly one subagent and reads one real skill file, then assert
                     a broad slice of the actions-graph/skills-graph model against
                     the result: Session-HAS_AGENT->Agent, the real spawning tool
                     call-SPAWNED->Agent, Agent-HAS_ACTION->its own tool calls,
                     per-container FOLLOWED_BY chains (and that they never cross),
                     PARENT_OF (ToolCall->ToolResult) at both levels, USED_TOOL,
                     and USED_SKILL attaching to the Agent (not the Session).
                     Verifies the SPAWNED inference rule and the skill-attachment
                     rule against a real session instead of synthetic e2e data.
                     Needs ANTHROPIC_API_KEY (env, or .env at the repo root) and
                     'claude' + 'agent-context-graph' on PATH. Costs a real LLM
                     call; Tier 1 only (one subagent, no concurrent-subagent
                     disambiguation).
  dogfood-env         Print export statements enabling auto_reconcile (true, automatic,
                     event-driven reconciliation on SESSION_END) for a claude session
                     launched afterward. Usage: eval \"\$(./scripts/dev-memgraph.sh dogfood-env)\" && claude
  test [pkg...]      Run test suites against a SEPARATE, disposable container --
                     never touches the main one. Default packages: ${ALL_PACKAGES[*]}

Typical workflow (exploration -- the actual goal of this script):
  $(basename "$0") up
  $(basename "$0") hooks-local        # drive a REAL Claude Code session to see data land
  $(basename "$0") inspect
  $(basename "$0") reconcile          # runs real entity extraction on pending sessions (costs an LLM call)
  $(basename "$0") inspect
  $(basename "$0") hooks-restore      # point your real hooks back before you forget
  $(basename "$0") down               # only when fully done -- this deletes your explored data

Automatic reconciliation instead of the manual \`reconcile\` step above --
takes effect for a NEW claude session, not one already running:
  eval \"\$($(basename "$0") dogfood-env)\" && claude

Running the automated suites any time (does not touch the above):
  $(basename "$0") test
  $(basename "$0") test-down          # optional -- reclaim the disposable test container
"

_wait_ready() {
  local port="$1" container="$2"
  echo "Waiting for Memgraph to accept Bolt connections on localhost:${port}..."
  for i in $(seq 1 30); do
    if (echo >"/dev/tcp/localhost/${port}") 2>/dev/null; then
      echo "Memgraph is ready."
      return 0
    fi
    echo "Waiting for Memgraph (attempt $i)..."
    sleep 2
  done
  echo "ERROR: Memgraph did not become ready in time." >&2
  docker logs "${container}" 2>&1 | tail -30 >&2 || true
  exit 1
}

_require_container_reachable() {
  if ! (echo >"/dev/tcp/localhost/${HOST_PORT}") 2>/dev/null; then
    echo "ERROR: no Memgraph reachable on localhost:${HOST_PORT}. Run '$(basename "$0") up' first." >&2
    exit 1
  fi
}

# Idempotent start for any named container/port pair -- shared by the main
# `up` command and `test`'s own separate throwaway container.
_container_up() {
  local name="$1" port="$2"
  if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    if docker ps --format '{{.Names}}' | grep -qx "${name}"; then
      echo "Already running: ${name} on port ${port}"
    else
      echo "Starting existing container ${name}..."
      docker start "${name}" >/dev/null
    fi
  else
    echo "Creating ${name} (${IMAGE}) on port ${port}..."
    docker run -d -p "${port}:7687" --name "${name}" "${IMAGE}" \
      --schema-info-enabled=True --telemetry-enabled=false >/dev/null
  fi
  _wait_ready "${port}" "${name}"
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
  _container_up "${CONTAINER_NAME}" "${HOST_PORT}"
}

_container_down() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    docker rm -f "${name}" >/dev/null
    echo "Removed ${name}."
  else
    echo "${name} does not exist."
  fi
}

cmd_down() {
  _container_down "${CONTAINER_NAME}"
}

cmd_test_down() {
  _container_down "${TEST_CONTAINER_NAME}"
}

_describe_container() {
  local label="$1" name="$2" port="$3"
  if docker ps --format '{{.Names}}' | grep -qx "${name}"; then
    echo "${label}: running (${name}, localhost:${port})"
  elif docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    echo "${label}: stopped (${name})"
  else
    echo "${label}: not created"
  fi
}

cmd_status() {
  _describe_container "Main container " "${CONTAINER_NAME}" "${HOST_PORT}"
  _describe_container "Test container " "${TEST_CONTAINER_NAME}" "${TEST_HOST_PORT}"

  if [ -f "${CONFIG_BACKUP}" ]; then
    echo "Hook config:     LOCAL mode (your real config is backed up at ${CONFIG_BACKUP})"
  else
    echo "Hook config:     real/production mode (no local-mode backup present)"
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

# Resolves ANTHROPIC_API_KEY for cmd_test_graph_model: prefer whatever is
# already in the environment, otherwise pull it from .env at the repo root.
# Never prints the value anywhere.
_resolve_anthropic_api_key() {
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    return 0
  fi
  local env_file="${REPO_ROOT}/.env"
  if [ -f "${env_file}" ]; then
    ANTHROPIC_API_KEY="$(grep -E '^ANTHROPIC_API_KEY=' "${env_file}" | head -1 | cut -d'=' -f2-)"
    export ANTHROPIC_API_KEY
  fi
  [ -n "${ANTHROPIC_API_KEY:-}" ]
}

# Prints export statements for the vars SessionsGraphConnector's own
# auto_reconcile fallback (SESSIONS_GRAPH_AUTO_RECONCILE) and the detached
# reconcile subprocess it spawns actually need. Deliberately NOT something
# this script sets permanently in a shell profile: env vars set on an
# already-running `claude` process can't retroactively reach it (or its hook
# subprocesses) -- they only take effect for a `claude` you launch fresh
# afterward, from a shell that has run this first. Usage:
#   eval "$(./scripts/dev-memgraph.sh dogfood-env)" && claude
# Scoped to that one shell/session; a plain new terminal is unaffected.
cmd_dogfood_env() {
  if ! _resolve_openai_api_key; then
    echo "ERROR: OPENAI_API_KEY is not set and was not found in ${REPO_ROOT}/.env" >&2
    exit 1
  fi
  cat <<EOF
export SESSIONS_GRAPH_AUTO_RECONCILE=1
export MEMGRAPH_URL="${LOCAL_MEMGRAPH_URL}"
export MEMGRAPH_USER="${LOCAL_MEMGRAPH_USER}"
export MEMGRAPH_PASSWORD="${LOCAL_MEMGRAPH_PASSWORD}"
export MEMGRAPH_DATABASE="${LOCAL_MEMGRAPH_DATABASE}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
EOF
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
  # Its own separate, disposable container -- see the comment on
  # TEST_CONTAINER_NAME above for why. Never the main $CONTAINER_NAME.
  _container_up "${TEST_CONTAINER_NAME}" "${TEST_HOST_PORT}"
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
    "MEMGRAPH_URL=${TEST_MEMGRAPH_URL}"
    "MEMGRAPH_USER=${TEST_MEMGRAPH_USER}"
    "MEMGRAPH_PASSWORD=${TEST_MEMGRAPH_PASSWORD}"
    "MEMGRAPH_DATABASE=${TEST_MEMGRAPH_DATABASE}"
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

# Verifies a broad slice of the actions-graph/skills-graph model against a
# REAL Claude Code session, not synthetic e2e data: the SPAWNED subagent-
# nesting inference rule, per-container HAS_ACTION/FOLLOWED_BY containment,
# and Agent-scoped skill-usage attachment. Tier 1 only: one subagent, no
# concurrent-subagent disambiguation.
#
# Generates its own self-contained hooks config (via claude_code.py's
# build_hooks_config, with the same --connector flags the installed
# marketplace plugin uses) rather than depending on that plugin already being
# installed -- this needs to work unattended on a bare CI runner, not just a
# dogfooding dev machine. Only the Memgraph *target* still goes through the
# existing hooks-local/hooks-restore swap, shared with the installed plugin.
cmd_test_graph_model() {
  _container_up "${CONTAINER_NAME}" "${HOST_PORT}"

  if ! command -v claude >/dev/null 2>&1; then
    echo "ERROR: claude is not on PATH." >&2
    exit 1
  fi
  if ! command -v agent-context-graph >/dev/null 2>&1; then
    echo "ERROR: agent-context-graph is not on PATH." >&2
    echo "Install it first, e.g. via context-graph/plugins/agent-context-graph-claude/scripts/bootstrap.sh" >&2
    exit 1
  fi
  if ! _resolve_anthropic_api_key; then
    echo "ERROR: ANTHROPIC_API_KEY is not set and was not found in ${REPO_ROOT}/.env" >&2
    echo "This drives a real Claude Code session and needs it." >&2
    exit 1
  fi

  local settings_file
  settings_file="$(mktemp -t test-graph-model-settings.XXXXXX.json)"
  uv run --package agent-context-graph python3 - "${settings_file}" <<'PYEOF'
import json
import sys

from agent_context_graph.adapters.claude_code import build_hooks_config

# `uv run --package agent-context-graph agent-context-graph ...` rather than a
# bare `agent-context-graph` -- a real, discovered-the-hard-way ambiguity: a
# stale globally `uv tool install`ed copy can shadow (or be shadowed by) this
# repo's own workspace venv on PATH, silently testing the wrong version of
# this code. This form always resolves to the exact checkout being verified,
# in CI or locally, regardless of what else is installed. --connector flags
# match the installed marketplace plugin's own generated command
# (context-graph-plugins/context-graph/*/hooks/hooks.json) -- without them,
# `hook run` does nothing (a second real, silent gap this verification
# tripped over).
command = (
    "uv run --package agent-context-graph agent-context-graph hook run claude-code "
    "--connector skills-graph --connector actions-graph --connector sessions-graph"
)
with open(sys.argv[1], "w") as f:
    json.dump({"hooks": build_hooks_config(command)}, f)
PYEOF

  # Only swap (and later restore) if hooks-local hasn't already been applied
  # by the caller -- same convention cmd_hooks_local itself uses.
  local we_swapped_hooks=0
  if [ ! -f "${CONFIG_BACKUP}" ]; then
    cmd_hooks_local
    we_swapped_hooks=1
  else
    echo "Hook config already in local mode -- leaving as-is."
  fi
  # Only arm the restore trap if a backup actually exists now -- on a machine
  # with no pre-existing real config (e.g. a fresh CI runner), cmd_hooks_local
  # has nothing to back up and creates no backup file, so there is nothing to
  # restore either. Without this check, the trap would call cmd_hooks_restore
  # unconditionally on exit, which errors ("no backup found") and calls
  # `exit 1` from inside the trap -- clobbering even a successful run's exit
  # code.
  if [ "${we_swapped_hooks}" -eq 1 ] && [ -f "${CONFIG_BACKUP}" ]; then
    trap cmd_hooks_restore EXIT
  fi
  if [ "${we_swapped_hooks}" -eq 1 ]; then
    # sessions-graph only MERGEs (:User)-[:HAD_SESSION]->(:Session) when a
    # user_id actually resolves (by design -- see connector.py's `if
    # user_id:` guard). On a fresh machine with none configured, this is the
    # difference between that check ever being able to pass at all. Only
    # done when we own this swap -- if hooks-local was already active
    # (someone else's real, still-running local-mode session), we have no
    # restore path for this and must not clobber their identity.
    agent-context-graph config set identity.user_id "test-graph-model"
  fi

  local session_id
  session_id="$(uv run --package agent-context-graph python3 -c 'import uuid; print(uuid.uuid4())')"
  echo "Driving a real Claude Code session (session_id=${session_id})..."

  # Restricting the top-level session to ONLY the Task tool -- Claude Code's
  # CLI/UI-facing name for subagent launch; it reports as tool_name "Agent" in
  # the actual hook payloads, a real, non-obvious naming split this
  # verification effort discovered -- forces genuine delegation: the model has
  # no other way to accomplish anything, so it must spawn a subagent rather
  # than maybe doing so. The skill-file read is folded into the SAME subagent
  # call (rather than a second session) so this still costs exactly one LLM
  # call while also exercising skills-graph's Agent-scoped USED_SKILL.
  #
  # Deliberately does NOT spell out the skill's literal path here: Claude
  # Code propagates this top-level prompt into the parent "Agent" tool
  # call's own metadata, and SkillGraphConnector scans metadata for any
  # string that looks like a resolvable SKILL.md path -- so a literal path
  # in THIS prompt gets mistaken for a top-level skill read, even though
  # nothing at this level actually read anything (a real, pre-existing
  # false-positive in SkillGraphConnector's detection, caught by this exact
  # verification -- tracked separately, not fixed here). Making the
  # subagent locate the file itself keeps the path out of this prompt.
  local prompt="Use the Task tool to launch exactly one subagent (subagent_type: Explore) to do two things: (1) find this repository's 'release' skill (look for a directory under skills/ containing a SKILL.md describing it), read that file, and note what it's for, and (2) search this repository for the string 'TODO'. Report back a short summary combining both. Do not do either yourself -- delegate the entire thing to that one subagent."

  local output_file
  output_file="$(mktemp -t test-graph-model-output.XXXXXX.json)"
  set +e
  # Explicit cd: the hooks config's command embeds `uv run --package
  # agent-context-graph ...`, which resolves relative to the hook
  # subprocess's cwd -- which Claude Code sets to wherever `claude` itself
  # was launched from. Must be $REPO_ROOT regardless of where this script
  # was invoked from, or `uv run` won't find the workspace.
  (cd "${REPO_ROOT}" && claude -p "${prompt}" \
    --settings "${settings_file}" \
    --setting-sources project \
    --session-id "${session_id}" \
    --output-format json \
    --permission-mode bypassPermissions \
    --allowedTools=Task) >"${output_file}" 2>&1
  local claude_exit=$?
  set -e

  # Always show what the session actually did -- a zero exit only means the
  # CLI itself didn't crash, not that hooks fired or the model used the Task
  # tool as instructed. Needed to diagnose graph-shape failures below without
  # re-running (subagent spawning is model-decided, so failures aren't always
  # reproducible).
  echo "--- claude -p output (session_id=${session_id}) ---"
  cat "${output_file}"
  echo "--- end claude -p output ---"

  if [ "${claude_exit}" -ne 0 ]; then
    echo "ERROR: claude exited with status ${claude_exit}" >&2
    exit 1
  fi

  echo "Session complete. Checking graph shape for session_id=${session_id}..."
  MEMGRAPH_URL="${LOCAL_MEMGRAPH_URL}" \
    MEMGRAPH_USER="${LOCAL_MEMGRAPH_USER}" \
    MEMGRAPH_PASSWORD="${LOCAL_MEMGRAPH_PASSWORD}" \
    MEMGRAPH_DATABASE="${LOCAL_MEMGRAPH_DATABASE}" \
    VERIFY_SESSION_ID="${session_id}" \
    uv run --package actions-graph python3 - <<'PYEOF'
import os
import sys

from actions_graph import ActionsGraph

session_id = os.environ["VERIFY_SESSION_ID"]
graph = ActionsGraph()
db = graph._db

checks = []

session = graph.get_session(session_id)
checks.append(("Session node exists", session is not None))

# sessions-graph is wired into the same live run (--connector sessions-graph
# above) and produces this data today -- previously unchecked here even
# though it's free (same session, no extra LLM cost).
user_rows = db.query(
    "MATCH (:User)-[:HAD_SESSION]->(:Session {session_id: $sid}) RETURN count(*) AS c",
    params={"sid": session_id},
)
checks.append(("sessions-graph: User -[:HAD_SESSION]-> Session", user_rows[0]["c"] == 1))

# TEMPORARY DIAGNOSTIC: does ANY :User node exist at all, regardless of which
# session it's linked to? Distinguishes "SessionStart never fired for
# sessions-graph" from "it fired but linked to an unexpected session_id."
_all_users = db.query("MATCH (u:User) RETURN u.user_id AS uid")
print(f"DIAGNOSTIC: all :User nodes in graph: {_all_users}", file=sys.stderr)

status_rows = db.query(
    "MATCH (s:Session {session_id: $sid}) RETURN s.reconciliation_status AS status",
    params={"sid": session_id},
)
checks.append(
    ("sessions-graph: Session.reconciliation_status is set", bool(status_rows) and status_rows[0]["status"] is not None)
)

agents = graph.get_session_agents(session_id) if session is not None else []
checks.append(("HAS_AGENT: Session -> at least one Agent", len(agents) >= 1))

agent = agents[0] if agents else None

spawning_action_id = None
spawned_ok = False
if agent:
    rows = db.query(
        "MATCH (call:Action)-[:SPAWNED]->(:Agent {agent_id: $agent_id}) "
        "RETURN call.tool_name AS tool_name, call.action_id AS action_id",
        params={"agent_id": agent.agent_id},
    )
    spawned_ok = len(rows) == 1 and rows[0]["tool_name"] in graph.agent_spawning_tool_names
    if rows:
        spawning_action_id = rows[0]["action_id"]
checks.append(("SPAWNED: the real spawning tool call -> Agent", spawned_ok))

agent_actions = []
if agent:
    agent_actions = db.query(
        "MATCH (:Agent {agent_id: $agent_id})-[:HAS_ACTION]->(a:Action) "
        "RETURN a.action_id AS action_id, a.action_type AS action_type, a.timestamp AS ts "
        "ORDER BY ts",
        params={"agent_id": agent.agent_id},
    )
checks.append(("HAS_ACTION: Agent -> its own tool calls", len(agent_actions) >= 1))

# FOLLOWED_BY: the agent's own actions form one connected chain (n-1 edges
# purely within that set -- order doesn't need re-deriving, connectivity does).
followed_by_ok = True
if len(agent_actions) >= 2:
    agent_ids = [a["action_id"] for a in agent_actions]
    rows = db.query(
        "MATCH (a:Action)-[:FOLLOWED_BY]->(b:Action) "
        "WHERE a.action_id IN $ids AND b.action_id IN $ids "
        "RETURN count(*) AS c",
        params={"ids": agent_ids},
    )
    followed_by_ok = rows[0]["c"] == len(agent_ids) - 1
checks.append(("FOLLOWED_BY: Agent's own actions form one ordered chain", followed_by_ok))

# FOLLOWED_BY: the top-level session's own chain never crosses into the Agent's.
top_level_actions = db.query(
    "MATCH (:Session {session_id: $sid})-[:HAS_ACTION]->(a:Action) RETURN a.action_id AS action_id",
    params={"sid": session_id},
)
chain_separation_ok = True
if agent and top_level_actions and agent_actions:
    top_ids = [a["action_id"] for a in top_level_actions]
    agent_ids = [a["action_id"] for a in agent_actions]
    rows = db.query(
        "MATCH (a:Action)-[:FOLLOWED_BY]->(b:Action) "
        "WHERE (a.action_id IN $top_ids AND b.action_id IN $agent_ids) "
        "   OR (a.action_id IN $agent_ids AND b.action_id IN $top_ids) "
        "RETURN count(*) AS c",
        params={"top_ids": top_ids, "agent_ids": agent_ids},
    )
    chain_separation_ok = rows[0]["c"] == 0
checks.append(("FOLLOWED_BY: top-level chain never crosses into the Agent's chain", chain_separation_ok))

# PARENT_OF: the spawning tool call itself resolves to its own ToolResult.
parent_of_top_ok = False
if spawning_action_id:
    rows = db.query(
        "MATCH (:Action {action_id: $aid})-[:PARENT_OF]->(:Action:ToolResult) RETURN count(*) AS c",
        params={"aid": spawning_action_id},
    )
    parent_of_top_ok = rows[0]["c"] == 1
checks.append(("PARENT_OF: the spawning tool call -> its ToolResult", parent_of_top_ok))

# PARENT_OF: at least one of the Agent's own tool calls resolves the same way.
parent_of_nested_ok = False
nested_calls = [a["action_id"] for a in agent_actions if a["action_type"] == "tool_call"]
if nested_calls:
    rows = db.query(
        "MATCH (:Action {action_id: $aid})-[:PARENT_OF]->(:Action:ToolResult) RETURN count(*) AS c",
        params={"aid": nested_calls[0]},
    )
    parent_of_nested_ok = rows[0]["c"] == 1
checks.append(("PARENT_OF: a nested tool call inside the Agent -> its ToolResult", parent_of_nested_ok))

# USED_TOOL: the Agent's own tool calls link to real Tool nodes.
used_tool_ok = False
if agent:
    rows = db.query(
        "MATCH (:Agent {agent_id: $agent_id})-[:HAS_ACTION]->(:Action)-[:USED_TOOL]->(t:Tool) "
        "RETURN count(DISTINCT t) AS c",
        params={"agent_id": agent.agent_id},
    )
    used_tool_ok = rows[0]["c"] >= 1
checks.append(("USED_TOOL: Agent's tool calls link to Tool nodes", used_tool_ok))

# USED_SKILL (skills-graph): attaches to the Agent, not the Session --
# reading skills/release/SKILL.md happened inside the subagent, so the
# either-container design should route it there, not to the top-level Session.
used_skill_agent_ok = False
used_skill_session_absent_ok = True
if agent:
    rows = db.query(
        "MATCH (:Agent {agent_id: $agent_id})-[:USED_SKILL]->(sk:Skill) RETURN sk.name AS name",
        params={"agent_id": agent.agent_id},
    )
    used_skill_agent_ok = len(rows) >= 1
    session_rows = db.query(
        "MATCH (:Session {session_id: $sid})-[:USED_SKILL]->(:Skill) RETURN count(*) AS c",
        params={"sid": session_id},
    )
    used_skill_session_absent_ok = session_rows[0]["c"] == 0
checks.append(("USED_SKILL: attaches to the Agent, not the Session", used_skill_agent_ok))
checks.append(("USED_SKILL: correctly NOT also attached to the Session", used_skill_session_absent_ok))

print()
all_ok = True
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all_ok and ok

print()
if all_ok:
    print("Graph-model verification PASSED against a real Claude Code session.")
    sys.exit(0)
print("Graph-model verification FAILED.")
sys.exit(1)
PYEOF
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
    test-down) cmd_test_down ;;
    reconcile) cmd_reconcile "$@" ;;
    dogfood-env) cmd_dogfood_env ;;
    hooks-local) cmd_hooks_local ;;
    hooks-restore) cmd_hooks_restore ;;
    test-graph-model) cmd_test_graph_model ;;
    -h | --help | "") echo "$_HELP" ;;
    *)
      echo "Unknown command: $cmd" >&2
      echo "$_HELP" >&2
      exit 2
      ;;
  esac
}

main "$@"
