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

Generate wiring for runtimes with a project-local installer:

```bash
agent-context-graph hook init copilot-cli
agent-context-graph hook init cursor
agent-context-graph hook init opencode
agent-context-graph hook init antigravity-cli
```

`hook init` enables all three built-in connectors by default. Use repeated
`--connector` flags to select a subset. Existing Copilot, Cursor, and Antigravity
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
`agentStop` is deliberately not installed: it marks the end of one turn,
whereas `sessionEnd` is the durable session boundary represented by the Event
Protocol.

### Cursor

Cursor stores flat command entries in `.cursor/hooks.json`. The stable Agent
Session key is `conversation_id`; `generation_id` is recorded as per-turn
metadata. The per-turn `stop` hook is not treated as `SessionEnd`; Cursor's
separate `sessionEnd` hook closes the conversation. Cursor blocks a permission
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
deltas), `session.execution.failed`, and `permission.asked`. The hook command
is embedded as an argv array and spawned directly with `node:child_process`:
no shell or login profile sits in between, and the shim never injects Memgraph
or LLM credentials. See the
[OpenCode V2 plugin reference](https://opencode.ai/v2/docs/build/plugins).
`session.idle` is a reusable-session idle boundary and is not translated into
`SessionEnd`; deletion is the terminal lifecycle event available on the public
event stream.

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
`Stop` ends one execution loop rather than the conversation, so only a failed
execution is recorded, as an error.

Antigravity hooks carry no tool results, prompts, assistant text, or token
usage, so an Antigravity session records which tools ran and whether they
failed, but not what they returned. Capture hooks never answer a permission
decision: an empty `PreToolUse` response leaves `agy`'s own prompt in charge,
and `PostToolUse` answers the required `{}`. See the
[Antigravity hooks reference](https://antigravity.google/docs/hooks/).

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
