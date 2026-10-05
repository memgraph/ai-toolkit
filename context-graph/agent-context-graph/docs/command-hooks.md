# Command Hook Reference

Agent Context Graph translates runtime-specific hooks into the shared Event
Protocol, then routes those events to graph connectors. The installed command
is the same for every runtime:

```bash
agent-context-graph hook run <runtime> \
  --connector skills-graph \
  --connector actions-graph \
  --connector sessions-graph
```

Runtime registrations are discovered through the
`agent_context_graph.runtimes` Python entry-point group.

## Supported runtimes

| Runtime | Registration | Integration | Project-local installation |
|---|---|---|---|
| Claude Code | `claude-code` | JSON command hooks | Runtime Plugin recommended |
| OpenAI Codex | `codex` | JSON command hooks | `.codex/config.toml`, `.codex/hooks.json` |
| GitHub Copilot CLI | `copilot-cli` | JSON command hooks | `.github/hooks/agent-context-graph.json` |
| Cursor | `cursor` | JSON command hooks | `.cursor/hooks.json` |
| OpenCode | `opencode` | V2 JavaScript plugin | `.opencode/plugins/agent-context-graph/index.js` |
| Antigravity CLI | `antigravity-cli` | JSON command hooks | `.agents/hooks.json` |
| Grok Build | `grok` | JSON command hooks | `.grok/hooks/agent-context-graph.json` |

Generate wiring for runtimes with a project-local installer:

```bash
agent-context-graph hook init copilot-cli
agent-context-graph hook init cursor
agent-context-graph hook init opencode
agent-context-graph hook init antigravity-cli
agent-context-graph hook init grok
```

`hook init` enables all three built-in connectors by default. Use repeated
`--connector` flags to select a subset. Existing Copilot, Cursor, Antigravity, and Grok
JSON documents are merged without removing unrelated settings or hooks.
OpenCode's generated plugin is replaced only with `--force`.

`--setup-schema` creates the Memgraph schema for each selected connector after
the wiring is written, for every runtime. Its `--memgraph-*` overrides are used
only for that step and are never written into generated wiring; everything
else resolves from the config file.

## Runtime shapes

### Codex and Claude Code

Both use nested command entries:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "agent-context-graph hook run codex --connector actions-graph",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

Codex's initializer also writes `[features] hooks = true` to
`.codex/config.toml`. Claude Code is normally wired by the repository's
Runtime Plugin rather than `hook init`.

### GitHub Copilot CLI

Copilot stores a versioned hook document under `.github/hooks/`. Native
camel-case payloads do not include an event-name field, so the generated
command supplies `--event-name` for the generic runner:

```json
{
  "version": 1,
  "hooks": {
    "preToolUse": [
      {
        "type": "command",
        "command": "agent-context-graph hook run copilot-cli --event-name preToolUse",
        "timeoutSec": 30,
        "matcher": ".*"
      }
    ]
  }
}
```

Native payloads carry `toolArgs` as a JSON string, which is parsed back into
an object, and a `toolResult.resultType` other than `success` marks the result
as an error. Copilot sends no tool-call id. Subagent start and stop are joined
on `agentName`, the only field both payloads share. Capture hooks write no
output, which Copilot treats as its default behavior.

See the [GitHub Copilot hooks reference](https://docs.github.com/en/copilot/reference/hooks-configuration).
`agentStop` ends one turn and `sessionEnd` ends the session; see
[Turns and sessions](#turns-and-sessions).

### Cursor

Cursor stores flat command entries in `.cursor/hooks.json`. The stable Agent
Session key is `conversation_id`; `generation_id` is recorded as per-turn
metadata. The per-turn `stop` hook is a turn end; Cursor's separate
`sessionEnd` hook closes the conversation. Cursor blocks a permission
hook (`preToolUse`, `subagentStart`) whose output does not match its schema, so
those hooks answer `{"permission": "allow"}`. That is the weakest vote: any
other hook's `deny` or `ask` still wins. See the
[Cursor hooks reference](https://cursor.com/docs/hooks).

### OpenCode

OpenCode V2 does not use command-hook JSON. `hook init opencode` installs a
dependency-free V2 plugin that registers prompt and tool hooks, subscribes to
the public event stream, and pipes normalized JSON to the Python runtime
registration. It captures `session.created`, `session.deleted`,
`session.text.ended` (one completed assistant text part, not the streamed
deltas), `session.execution.succeeded` and `session.execution.failed` (turn
ends), and `permission.asked`. The hook command
is embedded as an argv array and spawned directly with `node:child_process`:
no shell or login profile sits in between, and the shim never injects Memgraph
or LLM credentials. See the
[OpenCode V2 plugin reference](https://opencode.ai/v2/docs/build/plugins).
`session.deleted` is the only session end OpenCode reports; `opencode run`
never deletes its session, so those sessions only ever see turn ends.

### Antigravity CLI

Antigravity CLI (`agy`, the successor to Gemini CLI) reads named hook groups
from `.agents/hooks.json`; `hook init` owns the `agent-context-graph` group and
merges it beside any others. Only the tool events (`PreToolUse`,
`PostToolUse`) take matcher groups. `PreInvocation` and `Stop` are written as
flat handlers, because `agy` rejects the whole file when a lifecycle event is
nested. Timeouts are in seconds.

```json
{
  "agent-context-graph": {
    "PreInvocation": [
      {"type": "command", "command": "agent-context-graph hook run antigravity-cli --event-name PreInvocation", "timeout": 30}
    ],
    "PreToolUse": [
      {"hooks": [{"type": "command", "command": "agent-context-graph hook run antigravity-cli --event-name PreToolUse", "timeout": 30}]}
    ]
  }
}
```

Payloads carry no event name, so the generated command supplies
`--event-name`. The session key is `conversationId`. There is no session-start
hook: each turn's first model invocation (`invocationNum` resets to `0` per
turn) records the session with `modelName` and `workspacePaths[0]`, and repeats
are ignored. There is no tool-call id either;
`stepIdx`, shared by a call's `PreToolUse` and `PostToolUse`, pairs them.
`Stop` ends one execution loop rather than the conversation; a loop that
leaves the agent `fullyIdle` is a turn end, and a failed one is also recorded
as an error. `agy` has no session-end hook.

Antigravity hooks carry no tool results, prompts, assistant text, or token
usage, so an Antigravity session records which tools ran and whether they
failed, but not what they returned. Capture hooks never answer a permission
decision: an empty `PreToolUse` response leaves `agy`'s own prompt in charge,
and `PostToolUse` answers the required `{}`. See the
[Antigravity hooks reference](https://antigravity.google/docs/hooks/).

### Grok Build

Grok Build (`grok`) reads project hooks from `.grok/hooks/*.json`;
`hook init grok` writes a dedicated `.grok/hooks/agent-context-graph.json` in
the Claude Code shape (`{"hooks": {Event: [{"hooks": [...]}]}}`), with timeouts
in seconds. Project hooks run only after the folder is trusted (`/hooks-trust`
or `grok --trust`).

Payloads repeat every field in camelCase and Claude Code's snake_case
(`hook_event_name`, `session_id`, `tool_name`, `tool_input`, `tool_response`,
`tool_use_id`), so tool calls pair on a real id. Tool results are typed:
shell results record `output_for_prompt` and a non-zero `exit_code` marks an
error; file reads record `FileContent.content`. `Stop` fires at the end of
every turn (`reason: end_turn`) and again at shutdown; the former is a turn
end that records `lastAssistantMessage` as the assistant reply, and
`SessionEnd` ends the session.

Grok also runs hooks from a project's `.claude/settings.json` and
`.cursor/hooks.json`, passing its own payloads. The `claude-code` and `cursor`
registrations ignore payloads carrying Grok's camelCase `hookEventName`, which
those runtimes never send, so a project wired for several runtimes records a
Grok session once, as `grok`. See the
[Grok Build hooks reference](https://docs.x.ai/build/features/hooks).

### Turns and sessions

Most runtimes fire a stop hook after every turn, and only some report a real
session end, so adapters emit two different events:

| Runtime | Turn end | Session end |
|---|---|---|
| Claude Code | `Stop` (records `last_assistant_message`) | `SessionEnd` |
| Codex | `Stop` (records `last_assistant_message`) | none |
| Copilot CLI | `agentStop` | `sessionEnd` |
| Cursor | `stop` | `sessionEnd` |
| OpenCode | `session.execution.succeeded` / `.failed` | `session.deleted` |
| Antigravity CLI | idle `Stop` | none |
| Grok Build | `Stop` with `reason: end_turn` (records `lastAssistantMessage`) | `SessionEnd` |

A turn end marks the session `reconciliation_status: pending` in
sessions-graph without spawning reconciliation. Only a session end spawns it
(when `reconcile.auto_reconcile` is on). Sessions from runtimes with no
session-end hook are reconciled by the sweep:

```bash
sessions-graph reconcile --pending
```

## Persistent hook configuration

Hook subprocesses read identity, Memgraph, LLM, and reconciliation values from
`~/.config/context-graph/config.toml`. At hook runtime, values never come from
ambient environment variables. `CONTEXT_GRAPH_CONFIG` may select a different
file, but does not supply any value itself.

```bash
agent-context-graph config set identity.user_id "your-name"
agent-context-graph config set memgraph.url "bolt://localhost:7687"
agent-context-graph config show
```

Bootstrap captures supported environment variables into that file as a
write-time convenience:

```bash
agent-context-graph bootstrap --runtime antigravity-cli \
  --connector skills-graph \
  --connector actions-graph \
  --connector sessions-graph
```

## Verification

Smoke-test a registration and its configured connectors:

```bash
agent-context-graph doctor --runtime cursor \
  --connector skills-graph \
  --connector actions-graph \
  --connector sessions-graph
```

Each runtime provides its own native probe payload; `doctor` does not assume
that every runtime calls its session-end event `Stop`.

For live integration tests, use the repository-owned disposable Memgraph:

```bash
./scripts/dev-memgraph.sh test actions-graph
./scripts/dev-memgraph.sh test agent-context-graph
./scripts/dev-memgraph.sh test-down
```

To check the hooks against the real agent CLIs, run one headless session per
installed runtime. Each runs in its own scratch project, against a disposable
Memgraph and a throwaway config file, and the graph each one produced is
printed afterwards. Grant each runtime's project trust once first; the
script's header lists which runtimes need it.

```bash
./context-graph/scripts/live-hooks-e2e/live-hooks-e2e.sh up ~/tmp/live-hooks 7699
./context-graph/scripts/live-hooks-e2e/live-hooks-e2e.sh run ~/tmp/live-hooks
./context-graph/scripts/live-hooks-e2e/live-hooks-e2e.sh verify ~/tmp/live-hooks
./context-graph/scripts/live-hooks-e2e/live-hooks-e2e.sh down ~/tmp/live-hooks
```

## Adding a runtime

Command-hook runtimes declare a `RuntimeSpec` and subclass `SpecAdapter`.
The spec owns field aliases, event rules, event-name aliases, hook responses,
metadata keys, a native doctor probe, and command-config rendering. Wrap it in
a `SpecPlugin`, passing `init=json_hook_installer(SPEC)` when the runtime reads
a project-local JSON hook file, and publish that Runtime Registration in
`pyproject.toml`:

```toml
[project.entry-points."agent_context_graph.runtimes"]
my-runtime = "my_package.adapter:PLUGIN"
```

Tests should cover recorded payloads at the adapter interface, generated
configuration through `hook init`, the cross-runtime registration contract,
and at least one real-Memgraph path through the applicable Graph Connector.
