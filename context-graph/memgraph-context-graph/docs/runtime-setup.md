# Runtime setup

[Back to Memgraph Context Graph](../README.md)

## Command Hook Runtimes

Some agent applications run hooks as external commands instead of in-process SDK callbacks. Runtime adapters should keep the product-specific JSON mapping at the edge, emit the shared `Event` protocol, and leave graph persistence in connectors such as `SkillGraphConnector`.

The installed command is runtime-dispatched:

```bash
memgraph-context-graph hook <command> [options]
```

Implemented:

| Runtime | Adapter | Hook Shape |
|---------|---------|------------|
| OpenAI Codex | `CodexHooksAdapter` | Command receives one JSON object on `stdin` |
| Claude Code | `ClaudeCodeHooksAdapter` | Command receives one JSON object on `stdin` |
| GitHub Copilot CLI | `CopilotCLIHooksAdapter` | Command receives one JSON object on `stdin` |
| Cursor | `CursorHooksAdapter` | Command receives one JSON object on `stdin` |
| OpenCode | `OpenCodeHooksAdapter` | V2 plugin normalizes callbacks into JSON for the command |
| Antigravity CLI | `AntigravityCLIHooksAdapter` | Command receives one JSON object on `stdin` |
| Grok Build | `GrokHooksAdapter` | Command receives one JSON object on `stdin` |

Codex and Claude Code have distributable Runtime Plugins. Copilot CLI, Cursor,
OpenCode, Antigravity CLI, and Grok Build use project-local wiring, which `setup`
writes (or, one runtime at a time, `hook init`):

```bash
memgraph-context-graph hook init copilot-cli
memgraph-context-graph hook init cursor
memgraph-context-graph hook init opencode
memgraph-context-graph hook init antigravity-cli
memgraph-context-graph hook init grok
```

See the [command-hook reference](command-hooks.md) for generated paths,
payload differences, and current upstream hook documentation.

## Setup

`memgraph-context-graph setup` installs Context Graph for every agent on the machine; the
[family README](../../README.md#getting-started) walks through it. In short:

```bash
memgraph-context-graph setup                     # interactive on a terminal; plain otherwise
memgraph-context-graph setup claude-code codex   # just these agents
memgraph-context-graph setup cursor --project-dir ~/work/billing
memgraph-context-graph setup --plan              # print the plan, change nothing
memgraph-context-graph update [runtime ...]      # upgrade the package, refresh the wiring and the schema
memgraph-context-graph uninstall [runtime ...]   # remove the wiring; asks about the graph, config and tool
memgraph-context-graph uninstall --all          # the wiring, the graph, the config and the tool
```

It finds or starts Memgraph (a `memgraph-mage` container with a volume, on a free port when 7687 is taken),
writes `memgraph.url` and `identity.user_id`, installs missing connectors and the viewer into the uv tool,
wires each agent, and runs `doctor`'s checks. Every step checks before it acts, so rerunning only does what's
missing. The TUI starts only when stdin and stdout are both terminals; the plugins' `scripts/bootstrap.sh`
runs `setup <runtime> --no-tui`.

**Uninstalling.** `uninstall` always removes the wiring (plugins, hook files, MCP entries) and gives each
agent its own memory back. On a terminal it then asks, one question each and every answer defaulting to
keep, whether to delete the graph (setup's Memgraph container and its volume: every session, for good), the
config and capture-failure log, and the uv tool. Without a terminal only the flags remove more: `--graph`,
`--config`, `--tool`, or `--all` for the three. A Memgraph you run yourself is never touched; only the
container setup started, by name (`--memgraph-container`). On Windows the running tool can't remove itself,
so it prints the `uv tool uninstall` command to run next.

**Updating.** Upgrading the package moves the code hooks and the MCP server run, but what setup wrote once stays
as it was. `update` upgrades the uv tool and its connectors, then, with the new release, refreshes everything
that is wired: Claude Code's and Codex's plugin (marketplace refreshed, plugin updated), each per-project agent's
hook file in `--project-dir` (default: the current directory) and its MCP registration, and the graph's indexes
and constraints. It then runs `doctor`'s checks. Name runtimes to refresh only those. Your own hooks, the config
and the graph's data are untouched, and running it again changes nothing.
Running agent sessions keep the hooks they loaded: restart them. Outside the uv tool (a virtualenv, say) the
package step only warns, and you upgrade it yourself first.

**Identity and connection.** setup writes `~/.config/context-graph/config.toml`; hooks read their configuration from that file at runtime (see [Configuration](../README.md#configuration) — env vars are **not** read at hook time). To change your identity later:

```bash
memgraph-context-graph config set identity.user_id "your-name"
```

For a remote or HA instance, pass `setup --memgraph-url`, or set it with credentials afterwards:

```bash
memgraph-context-graph config set memgraph.url "neo4j://<coordinator-host>:7687"
memgraph-context-graph config set memgraph.user "<user>"
memgraph-context-graph config set memgraph.password        # prompts; stored 0600
memgraph-context-graph config set memgraph.database "memgraph"
```

**Verify** (setup runs the same checks):

```bash
memgraph-context-graph config show
memgraph-context-graph doctor --runtime claude-code \
  --connector skills-graph --connector actions-graph --connector sessions-graph
```

Expected successful doctor output (use `--runtime codex` for Codex):

```text
OK memgraph-context-graph executable: ...
OK memgraph-context-graph: ...
OK config: identity.user_id set
OK memgraph: reachable
OK connector:skills-graph: installed=...; memgraph=reachable
OK connector:actions-graph: installed=...; memgraph=reachable
OK connector:sessions-graph: installed=...; memgraph=reachable
OK embeddings: BAAI/bge-small-en-v1.5 (384 dimensions) inside Memgraph
OK mcp: serves recall
OK runtime:claude-code: connectors accept its probe event
OK plugin:claude-code: context-graph@context-graph-plugins installed and enabled
```

The `runtime:` probe runs the connectors inside `doctor` itself; `plugin:` is what says Claude Code (or Codex)
will actually call the hooks.

> **Reconciliation is a separate step.** The connectors capture session activity, but turning a session's text into extracted entities (a `:Person`/`:Organization` graph) is done out-of-band — see [sessions-graph § reconciliation](../../sessions-graph/README.md#session-reconciliation). By default a finished session is marked `reconciliation_status = 'pending'` and `sessions-graph reconcile --pending` extracts it.

## Source development

For source development and per-project experiments, you can generate local Codex hook files:

```bash
memgraph-context-graph hook init codex --project-dir "$PWD" --setup-schema
```

This writes local, ignored files:

```text
.codex/config.toml
.codex/hooks.json
```

See [Command Hook Reference](command-hooks.md) for manual setup, non-default Memgraph values, smoke tests, and generated hook JSON details.
