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
#   live-hooks-e2e.sh memory  <workdir> [runtime...]   Context Graph as each agent's
#                             memory: a write, a read in a new session, and the
#                             next session's start (memory index, last ask)
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
#
# `memory` registers the MCP server the way setup does, but per project or per
# run, so no agent's user config changes: Claude Code --mcp-config, Codex -c,
# Copilot --additional-mcp-config, and project files for OpenCode, Antigravity
# and Grok. Two exceptions need you: Antigravity reads tool permissions only from
# user config, so it is skipped until `mcp(context-graph/*)` is allowed there
# (setup antigravity-cli writes it); and Codex loads project hooks only for a
# trusted project, so its session-start check runs through an installed
# context-graph plugin, which this checkout's CLI stands in for on PATH. The
# agent's model matters: OpenCode's default free model often reaches for its
# own file tools instead of `memory`; set LIVE_HOOKS_OPENCODE_MODEL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI="$REPO_ROOT/.venv/bin/memgraph-context-graph"
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
from memgraph_context_graph.adapters.claude_code import PLUGIN
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

#: Each memory run's prompts. The read run's ask is what the next session must quote back.
WRITE_PROMPT="Use your memory tool to create the file /memories/notes/%s.md containing exactly: The %s e2e codeword is %s. Then reply DONE."
READ_PROMPT="What is the e2e read codeword? It is saved in my memory. Reply with just the codeword."
START_PROMPT="Answer only from what you already know, without calling any tools: which memory file paths are in my memory index, and what was the first ask of my last session in this project? Reply NONE if you don't know."

memory_setup() {
  printf '[memory]\nbackend = "context-graph"\n' >> "$WORK/config.toml"
  printf '#!/bin/bash\nexec env CONTEXT_GRAPH_CONFIG="%s" "%s" mcp "$@" 2>>"%s"\n' \
    "$WORK/config.toml" "$CLI" "$WORK/mcp.stderr.log" > "$WORK/mcp.sh"
  chmod +x "$WORK/mcp.sh"
  printf '{"mcpServers":{"context-graph":{"command":"%s","args":[]}}}\n' "$WORK/mcp.sh" > "$WORK/claude-mcp.json"
  printf '{"mcpServers":{"context-graph":{"type":"local","command":"%s","args":[],"tools":["*"]}}}\n' \
    "$WORK/mcp.sh" > "$WORK/copilot-mcp.json"
  printf '{"mcpServers":{"context-graph":{"command":"%s","args":[]}}}\n' "$WORK/mcp.sh" \
    > "$WORK/proj-antigravity-cli/.agents/mcp_config.json"
  mkdir -p "$WORK/proj-grok/.grok"
  printf '[mcp_servers.context-graph]\ncommand = "%s"\nargs = ["--session-context"]\n\n[permission]\nallow = ["MCPTool(context-graph__*)"]\n' \
    "$WORK/mcp.sh" > "$WORK/proj-grok/.grok/config.toml"
  "$PYTHON" - "$WORK/proj-opencode" "$WORK/mcp.sh" <<'PY'
import json, pathlib, sys
project, server = pathlib.Path(sys.argv[1]), sys.argv[2]
config = project / "opencode.jsonc"
document = json.loads(config.read_text()) if config.exists() else {"$schema": "https://opencode.ai/config.json"}
document["mcp"] = {"servers": {"context-graph": {"type": "local", "command": [server], "codemode": False}}}
document["permissions"] = [{"action": "context-graph_*", "resource": "*", "effect": "allow"}]
config.write_text(json.dumps(document, indent=2))
PY
  CONTEXT_GRAPH_CONFIG="$WORK/config.toml" "$PYTHON" "$(dirname "$0")/memory_check.py" seed
  touch "$WORK/.memory"
}

# One headless memory session: the agent CLI with the context-graph server for this run, output to stdout.
memory_session() {
  local runtime=$1 prompt=$2
  local dir="$WORK/proj-$runtime"
  local -a cmd
  case "$runtime" in
    claude-code) cmd=(claude -p "$prompt" --settings "$WORK/claude-code-settings.json" --setting-sources project
                      --mcp-config "$WORK/claude-mcp.json" --strict-mcp-config
                      --allowedTools mcp__context-graph__memory mcp__context-graph__recall --output-format text) ;;
    codex)
      # Named like the plugin's server, so it replaces an installed plugin's for this run: -c can't approve a
      # plugin server's tools, and under `codex exec` an approval prompt is a refusal.
      cmd=(codex exec -s "${LIVE_HOOKS_CODEX_SANDBOX:-read-only}" -c "mcp_servers.context-graph.command=\"$WORK/mcp.sh\""
           -c 'mcp_servers.context-graph.default_tools_approval_mode="approve"')
      [ "${LIVE_HOOKS_TRUST_HOOKS:-}" = 1 ] && cmd+=(--dangerously-bypass-hook-trust)
      cmd+=("$prompt")
      ;;
    copilot-cli) cmd=(env COPILOT_ALLOW_ALL=true copilot -p "$prompt" -s --no-auto-update
                      --additional-mcp-config "@$WORK/copilot-mcp.json" --allow-tool=context-graph) ;;
    opencode)
      cmd=(opencode run --standalone)
      [ -n "${LIVE_HOOKS_OPENCODE_MODEL:-}" ] && cmd+=(-m "$LIVE_HOOKS_OPENCODE_MODEL")
      cmd+=("$prompt")
      ;;
    antigravity-cli) cmd=(agy -p "$prompt" --output-format text --print-timeout 240s) ;;
    grok) cmd=(env GROK_FOLDER_TRUST=0 grok -p "$prompt") ;;
    *) echo "unknown runtime: $runtime" >&2; return 1 ;;
  esac
  # The checkout's CLI first on PATH: an installed context-graph plugin (Claude Code, Codex, and Grok, which
  # loads Claude Code's plugins) then runs this checkout too, against CONTEXT_GRAPH_CONFIG's Memgraph.
  ( cd "$dir" && PATH="$REPO_ROOT/.venv/bin:$PATH" CONTEXT_GRAPH_CONFIG="$WORK/config.toml" "${cmd[@]}" </dev/null 2>&1 )
}

memory_one() {
  local runtime=$1
  local binary
  case "$runtime" in
    claude-code) binary=claude ;;
    copilot-cli) binary=copilot ;;
    antigravity-cli) binary=agy ;;
    *) binary=$runtime ;;
  esac
  if ! command -v "$binary" >/dev/null 2>&1; then
    echo "$runtime: skipped ($binary not installed)"
    return 0
  fi
  if [ "$runtime" = antigravity-cli ] \
    && ! grep -qF 'mcp(context-graph/*)' "$HOME/.gemini/antigravity-cli/settings.json" 2>/dev/null; then
    echo "$runtime: skipped (allow mcp(context-graph/*) in ~/.gemini/antigravity-cli/settings.json first)"
    return 0
  fi
  local word failed=()
  word="$(printf '%s' "$runtime" | tr '[:lower:]-' '[:upper:]_')-$RANDOM"
  # shellcheck disable=SC2059  # the prompt is the format
  memory_session "$runtime" "$(printf "$WRITE_PROMPT" "$runtime" "$runtime" "$word")" \
    > "$WORK/memory-$runtime-write.out" || true
  CONTEXT_GRAPH_CONFIG="$WORK/config.toml" "$PYTHON" "$(dirname "$0")/memory_check.py" written "$runtime" "$word" \
    || failed+=(write)
  memory_session "$runtime" "$READ_PROMPT" > "$WORK/memory-$runtime-read.out" || true
  grep -q QUARTZ-PELICAN "$WORK/memory-$runtime-read.out" || failed+=(read)
  memory_session "$runtime" "$START_PROMPT" > "$WORK/memory-$runtime-start.out" || true
  grep -q /memories/notes/seed.md "$WORK/memory-$runtime-start.out" || failed+=(index)
  grep -q "e2e read codeword" "$WORK/memory-$runtime-start.out" || failed+=(last-ask)
  if [ "${#failed[@]}" -gt 0 ]; then
    echo "$runtime: failed ${failed[*]} (outputs: $WORK/memory-$runtime-*.out)"
    return 1
  fi
  echo "$runtime: write, read, index and last ask ok"
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
  memory)
    [ -f "$WORK/.memory" ] || memory_setup
    targets=("${RUNTIMES[@]}")
    [ "$#" -gt 0 ] && targets=("$@")
    failed=()
    for runtime in "${targets[@]}"; do memory_one "$runtime" || failed+=("$runtime"); done
    if [ "${#failed[@]}" -gt 0 ]; then
      echo "failed: ${failed[*]}" >&2
      exit 1
    fi ;;
  down) docker rm -f "$container" >/dev/null && echo "removed $container" ;;
  *) echo "unknown command: $command_name" >&2; exit 2 ;;
esac
