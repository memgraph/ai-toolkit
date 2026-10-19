# Memgraph Context Graph for Claude Code

Claude Code runtime plugin for Memgraph Context Graph.

The plugin installs Claude Code lifecycle hooks that call:

```bash
memgraph-context-graph hook run claude-code --connector skills-graph --connector actions-graph --connector sessions-graph
```

Flow:

```text
Claude Code Plugin -> Claude Code Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

The plugin is only deployment wiring. Memgraph Context Graph normalizes runtime events. Graph connectors decide what to persist.

## First Run

The plugin installs hook wiring; the runtime package and the rest come from one script:

```bash
./scripts/bootstrap.sh
```

It installs the `memgraph-context-graph` CLI with its connectors through `uv`, then runs
`memgraph-context-graph setup claude-code --no-tui`: it uses the Memgraph your config names, or one on
`localhost:7687`, or starts one in Docker (with a volume, on a free port if 7687 is taken), sets your
identity, and checks the hooks with `doctor`. Running it again skips what is already done.

`uv` manages Python for the tool. If uv-managed Python downloads are blocked in your environment, install Python 3.10+ and rerun it.

To set up every agent on the machine at once, interactively, run `memgraph-context-graph setup` instead.
`memgraph-context-graph update claude-code` upgrades the package and refreshes the plugin; `memgraph-context-graph uninstall claude-code` removes the wiring and keeps your graph.

## Recall

The plugin also bundles an MCP server, `memgraph-context-graph mcp` (see `.mcp.json`), which gives the model a `recall` tool (`mcp__plugin_context-graph_context-graph__recall`) over the user's own past sessions. At session start the hook adds one line telling the model the tool is there; the model calls it when a question needs memory. Recall needs the `mcp` extra, which `bootstrap.sh` installs, and Memgraph with MAGE for its vector search. See [memgraph-context-graph § Recall](../../memgraph-context-graph/README.md#recall-memory-for-the-harnesss-model).

## Configure

Bootstrap writes `~/.config/context-graph/config.toml`; hooks read it at runtime (not environment variables). Set your identity — **required** for sessions-graph to attach sessions to a user:

```bash
memgraph-context-graph config set identity.user_id "your-name"
```

For a remote/HA Memgraph, also `memgraph-context-graph config set memgraph.url "neo4j://<host>:7687"` (and `memgraph.user`/`memgraph.password`/`memgraph.database`). The default is `bolt://localhost:7687`. See the [memgraph-context-graph README](../../memgraph-context-graph/README.md#configuration).

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
- `memgraph-context-graph` available on `PATH` after `bootstrap.sh`.

## Local Test

From the plugin directory:

```bash
./scripts/doctor.sh
```

## Git Marketplace

Register this repo as a Claude Code marketplace:

```text
/plugin marketplace add memgraph/ai-toolkit
```

CLI equivalent:

```bash
claude plugin marketplace add memgraph/ai-toolkit --sparse .claude-plugin
```

Install:

```text
/plugin install context-graph@context-graph-plugins
```

The Memgraph Context Graph skill is exposed as:

```text
/context-graph:setup
```

This is a Claude Code marketplace only. Codex uses the separate marketplace at `.agents/plugins/marketplace.json`.
