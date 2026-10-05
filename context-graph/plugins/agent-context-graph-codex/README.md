# Agent Context Graph for Codex

Codex runtime plugin for Agent Context Graph.

The plugin installs Codex lifecycle hooks that call:

```bash
agent-context-graph hook run codex --connector skills-graph --connector actions-graph --connector sessions-graph
```

Flow:

```text
Codex Plugin -> Codex Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

The plugin is only deployment wiring. Agent Context Graph normalizes runtime events. Graph connectors decide what to persist.

## First Run

The plugin installs hook wiring, while the runtime package is installed by the CLI bootstrap:

```bash
./scripts/bootstrap.sh
```

Bootstrap expects `uv` and a reachable Memgraph instance. If Memgraph is not running, start it and rerun bootstrap:

```bash
docker run --rm -p 7687:7687 memgraph/memgraph-mage
```

`uv` manages Python for the tool. If uv-managed Python downloads are blocked in your environment, install Python 3.10+ and rerun bootstrap.

Bootstrap installs and verifies:

```bash
agent-context-graph bootstrap --runtime codex --connector skills-graph --connector actions-graph --connector sessions-graph
```

## Recall

The plugin also bundles an MCP server, `agent-context-graph mcp` (see `.mcp.json`), which gives the model a `recall` tool (`recall` on the `context-graph` MCP server) over the user's own past sessions. At session start the hook adds one line telling the model the tool is there; the model calls it when a question needs memory. Recall needs the `mcp` extra, which `bootstrap.sh` installs, and Memgraph with MAGE for its vector search. See [agent-context-graph § Recall](../../agent-context-graph/README.md#recall-memory-for-the-harnesss-model).

Two Codex specifics:

- **Hooks need trust.** Codex runs a plugin's hooks only after you trust them in the startup review (or `/hooks`); until then nothing is recorded and the session-start line isn't added. `codex exec` never asks, so for unattended runs pass `--dangerously-bypass-hook-trust` or trust them once interactively.
- **Recall asks for approval** on each call, like any MCP tool. To allow it without asking, add to `~/.codex/config.toml`:

  ```toml
  [plugins."context-graph@context-graph-plugins".mcp_servers.context-graph]
  default_tools_approval_mode = "approve"
  ```

Codex hands MCP servers only selected environment variables, so `.mcp.json` forwards `CONTEXT_GRAPH_CONFIG`: a session pointed at another config file recalls from the same graph its hooks write to.

## Configure

Bootstrap writes `~/.config/context-graph/config.toml`; hooks read it at runtime (not environment variables). Set your identity — **required** for sessions-graph to attach sessions to a user:

```bash
agent-context-graph config set identity.user_id "your-name"
```

For a remote/HA Memgraph, also `agent-context-graph config set memgraph.url "neo4j://<host>:7687"` (and `memgraph.user`/`memgraph.password`/`memgraph.database`). The default is `bolt://localhost:7687`. See the [agent-context-graph README](../../agent-context-graph/README.md#configuration).

## Reconciliation (optional)

Captured sessions are marked `reconciliation_status = 'pending'` on end. To extract entities from that content into the graph:

```bash
pip install "sessions-graph[reconciliation]"
export OPENAI_API_KEY=...    # or ANTHROPIC_API_KEY
sessions-graph reconcile --pending
```

## Prerequisites

- Memgraph running and reachable over Bolt.
- `uv` available on `PATH`.
- `agent-context-graph` available on `PATH` after bootstrap.

## Local Test

From the plugin directory:

```bash
./scripts/doctor.sh
```

## Global Marketplace

This repo exposes a public Git-backed marketplace at:

```text
.agents/plugins/marketplace.json
```

Register the marketplace from GitHub and install the plugin:

```bash
codex plugin marketplace add memgraph/ai-toolkit --sparse .agents/plugins
codex plugin add context-graph@context-graph-plugins
```

Both are non-interactive; `codex plugin add` installs and enables the plugin in one step.

The Agent Context Graph skill is exposed as:

```text
context-graph:agent-context-graph
```

This is a Codex marketplace only. Claude Code uses the separate marketplace at `.claude-plugin/marketplace.json`.
