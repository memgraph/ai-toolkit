#!/bin/bash
# Live e2e for the command-hook runtimes: run real agent CLIs (Claude Code,
# Codex, Copilot CLI, OpenCode, Antigravity CLI, Grok Build) against hooks from
# this checkout, and check the graph they produce.
#
# Everything writes to a disposable Memgraph and a throwaway config file chosen
# with CONTEXT_GRAPH_CONFIG, so ~/.config/context-graph/config.toml and any real
# Memgraph are never touched. Each hook goes through a wrapper that also logs
# its raw stdin to <workdir>/<runtime>.payloads.jsonl, for comparing real
# payloads against what the adapters expect.
#
# Usage:
#   live-hooks-e2e.sh up      <workdir> [port]   start Memgraph, build projects + wiring
#   live-hooks-e2e.sh run     <workdir> [runtime...]   one headless session per runtime
#   live-hooks-e2e.sh verify  <workdir> [--check] [runtime...]   print (and with
#                             --check, assert) each session's graph shape
#   live-hooks-e2e.sh down    <workdir>          remove the Memgraph container
#
# Interactive use: before the first `run`, grant each runtime's project trust
# once in <workdir>/proj-<runtime>: Codex hook trust (`codex`), Copilot folder
# trust (`copilot`), and Grok `/hooks-trust`. Runtimes that are not installed
# are skipped. Default port is 7699; check `docker ps` first, since a test port
# shared with another workstream gets wiped. An installed context-graph Codex
# plugin also fires during Codex runs, against your real config; its Memgraph
# target is untouched by this harness, but expect its hooks in Codex's output.
#
# Unattended use (CI), via environment:
#   LIVE_HOOKS_TRUST_HOOKS=1      run Codex with --dangerously-bypass-hook-trust;
#                                 the hooks are the ones `up` just generated. Codex
#                                 also loads a project's .codex/ config only once
#                                 the project itself is trusted in its config.toml
#   LIVE_HOOKS_CODEX_SANDBOX      Codex sandbox (default read-only)
#   LIVE_HOOKS_OPENCODE_MODEL     e.g. anthropic/claude-haiku-4-5; `up` writes an
#                                 OpenCode provider config that reads the key
#                                 from <PROVIDER>_API_KEY
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO_ROOT/.venv/bin/agent-context-graph"
PYTHON="$REPO_ROOT/.venv/bin/python"
IMAGE="memgraph/memgraph-mage:latest"
PROMPT="Run the shell command 'ls', then read README.md and summarize it in one sentence."
RUNTIMES=(claude-code codex copilot-cli opencode antigravity-cli grok)

command_name=${1:?usage: live-hooks-e2e.sh up|run|verify|down <workdir> ...}
WORK=${2:?workdir required}
shift 2
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
container="ai-toolkit-live-hooks-e2e-$(basename "$WORK")"

port() { cat "$WORK/.port"; }

up() {
  local port=${1:-7699}
  if (exec 3<>"/dev/tcp/localhost/$port") 2>/dev/null; then
    echo "port $port is already in use; pick another" >&2
    exit 1
  fi
  docker run -d --rm -p "$port:7687" --name "$container" "$IMAGE" \
    --schema-info-enabled=True --telemetry-enabled=false >/dev/null
  # A TCP accept comes well before Memgraph can complete a Bolt handshake.
  "$PYTHON" - "$port" <<'PY'
import sys, time
from neo4j import GraphDatabase
deadline = time.monotonic() + 120
while True:
    try:
        with GraphDatabase.driver(f"bolt://localhost:{sys.argv[1]}", auth=("", "")) as driver:
            driver.verify_connectivity()
        break
    except Exception:
        if time.monotonic() > deadline:
            raise
        time.sleep(1)
PY
  echo "$port" > "$WORK/.port"

  printf '[identity]\nuser_id = "live-hooks-e2e"\n[memgraph]\nurl = "bolt://localhost:%s"\n' "$port" > "$WORK/config.toml"
  cat > "$WORK/capture-hook.sh" <<EOF
#!/bin/bash
payload=\$(cat)
printf '{"argv": "%s", "payload": %s}\n' "\$*" "\${payload:-null}" >> "$WORK/\$1.payloads.jsonl"
printf '%s' "\$payload" | CONTEXT_GRAPH_CONFIG="$WORK/config.toml" "$CLI" hook run "\$@" \\
  --connector actions-graph --connector sessions-graph --memgraph-url bolt://localhost:$port \\
  --memgraph-user "" --memgraph-password "" --memgraph-database memgraph --strict 2>>"$WORK/\$1.stderr.log"
EOF
  chmod +x "$WORK/capture-hook.sh"

  for runtime in "${RUNTIMES[@]}"; do
    local dir="$WORK/proj-$runtime"
    mkdir -p "$dir"
    ( cd "$dir"
      # A temp-dir cleanup can empty .git without removing it.
      git rev-parse --git-dir >/dev/null 2>&1 || { rm -rf .git; git init -q; }
      printf '# Demo project\n\nA tiny project used to test hooks.\n' > README.md
      printf 'print("hi")\n' > app.py
      git add README.md app.py
      git -c user.email=e2e@example.com -c user.name=e2e commit -qm init --allow-empty )
    if [ "$runtime" != claude-code ]; then
      CONTEXT_GRAPH_CONFIG="$WORK/config.toml" "$CLI" hook init "$runtime" --project-dir "$dir" \
        --connector actions-graph --connector sessions-graph \
        --hook-command "$WORK/capture-hook.sh $runtime" --timeout 30 --force >/dev/null
    fi
  done
  if [ -n "${LIVE_HOOKS_OPENCODE_MODEL:-}" ]; then
    local provider=${LIVE_HOOKS_OPENCODE_MODEL%%/*}
    local key_var
    key_var="$(printf '%s' "$provider" | tr '[:lower:]-' '[:upper:]_')_API_KEY"
    cat > "$WORK/proj-opencode/opencode.jsonc" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "model": "$LIVE_HOOKS_OPENCODE_MODEL",
  "providers": { "$provider": { "settings": { "apiKey": "{env:$key_var}" } } }
}
EOF
  fi
  "$PYTHON" - "$WORK" <<'PY'
import json, sys
from agent_context_graph.adapters.claude_code import PLUGIN
work = sys.argv[1]
hooks = PLUGIN.build_hooks_config(f"{work}/capture-hook.sh claude-code", timeout=30)
json.dump({"hooks": hooks}, open(f"{work}/claude-code-settings.json", "w"), indent=2)
PY
  CONTEXT_GRAPH_CONFIG="$WORK/config.toml" "$CLI" hook init grok --project-dir "$WORK/.schema" \
    --connector actions-graph --connector sessions-graph --hook-command x --force --setup-schema \
    --memgraph-url "bolt://localhost:$port" >/dev/null
  rm -rf "${WORK:?}/.schema"
  echo "ready: $WORK (Memgraph on $port, container $container)"
}

run_one() {
  local runtime=$1
  local dir="$WORK/proj-$runtime"
  local -a cmd
  case "$runtime" in
    claude-code) cmd=(claude -p "$PROMPT" --settings "$WORK/claude-code-settings.json" --setting-sources project
                      --allowedTools "Bash(ls)" "Read" --output-format text) ;;
    codex)
      cmd=(codex exec -s "${LIVE_HOOKS_CODEX_SANDBOX:-read-only}")
      [ "${LIVE_HOOKS_TRUST_HOOKS:-}" = 1 ] && cmd+=(--dangerously-bypass-hook-trust)
      cmd+=("$PROMPT")
      ;;
    copilot-cli) cmd=(copilot -p "$PROMPT" --allow-tool 'shell(ls)') ;;
    # --standalone: a background OpenCode service started earlier would not see
    # this shell's provider keys.
    opencode) cmd=(opencode run --standalone "$PROMPT") ;;
    antigravity-cli) cmd=(agy -p "$PROMPT" --print-timeout 180s) ;;
    grok) cmd=(grok -p "$PROMPT") ;;
    *) echo "unknown runtime: $runtime" >&2; return 1 ;;
  esac
  if ! command -v "${cmd[0]}" >/dev/null 2>&1; then
    echo "$runtime: skipped (${cmd[0]} not installed)"
    return 0
  fi
  local code=0
  ( cd "$dir" && "${cmd[@]}" </dev/null >"$WORK/run-$runtime.out" 2>&1 ) || code=$?
  local payloads=0 errors=0
  [ -f "$WORK/$runtime.payloads.jsonl" ] && payloads=$(wc -l <"$WORK/$runtime.payloads.jsonl" | tr -d ' ')
  [ -f "$WORK/$runtime.stderr.log" ] && errors=$(grep -c Traceback "$WORK/$runtime.stderr.log" || true)
  echo "$runtime: exit=$code payloads=$payloads hook-errors=$errors (output: $WORK/run-$runtime.out)"
  # A failed agent run, a hook that raised (hooks run with --strict), or no hook
  # firing at all fails the run step. Codex, for one, skips untrusted project
  # hooks silently and still exits 0.
  [ "$code" -eq 0 ] && [ "$errors" -eq 0 ] && [ "$payloads" -gt 0 ]
}

case "$command_name" in
  up) up "$@" ;;
  run)
    targets=("${RUNTIMES[@]}")
    [ "$#" -gt 0 ] && targets=("$@")
    failed=()
    for runtime in "${targets[@]}"; do run_one "$runtime" || failed+=("$runtime"); done
    if [ "${#failed[@]}" -gt 0 ]; then
      echo "failed: ${failed[*]}" >&2
      exit 1
    fi ;;
  verify)
    MEMGRAPH_URL="bolt://localhost:$(port)" MEMGRAPH_USER="" MEMGRAPH_PASSWORD="" MEMGRAPH_DATABASE=memgraph \
      "$PYTHON" "$(dirname "$0")/verify_graph.py" "$@" ;;
  down) docker rm -f "$container" >/dev/null && echo "removed $container" ;;
  *) echo "unknown command: $command_name" >&2; exit 2 ;;
esac
