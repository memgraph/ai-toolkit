# Memgraph Context Graph for Codex

Codex runtime plugin for Memgraph Context Graph.

The plugin installs Codex lifecycle hooks that call:

```bash
memgraph-context-graph hook run codex --connector skills-graph --connector actions-graph --connector sessions-graph
```

Flow:

```text
Codex Plugin -> Codex Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

The plugin is only deployment wiring. Memgraph Context Graph normalizes runtime events. Graph connectors decide what to persist.

## First Run

The plugin installs hook wiring; the runtime package and the rest come from one script:

```bash
./scripts/bootstrap.sh
```

It installs the `memgraph-context-graph` CLI with its connectors through `uv`, then runs
`memgraph-context-graph setup codex --no-tui`: it uses the Memgraph your config names, or one on
`localhost:7687`, or starts one in Docker (with a volume, on a free port if 7687 is taken), sets your
identity, and checks the hooks with `doctor`. Running it again skips what is already done.

`uv` manages Python for the tool. If uv-managed Python downloads are blocked in your environment, install Python 3.10+ and rerun it.

To set up every agent on the machine at once, interactively, run `memgraph-context-graph setup` instead.
`memgraph-context-graph update codex` upgrades the package and refreshes the plugin; `memgraph-context-graph uninstall codex` removes the wiring and keeps your graph; `memgraph-context-graph uninstall --all` removes everything, graph included.

## Recall

The plugin also bundles an MCP server, `memgraph-context-graph mcp` (see `.mcp.json`), which gives the model a `recall` tool (`recall` on the `context-graph` MCP server) over the user's own past sessions. At session start the hook adds one line telling the model the tool is there; the model calls it when a question needs memory. Recall needs the `mcp` extra, which `bootstrap.sh` installs, and Memgraph with MAGE for its vector search. See [memgraph-context-graph § Recall](../../memgraph-context-graph/README.md#recall-memory-for-the-harnesss-model).

Two Codex specifics:

- **Hooks need trust.** Codex runs a plugin's hooks only after you trust them in the startup review (or `/hooks`); until then nothing is recorded and the session-start line isn't added. `codex exec` never asks, so for unattended runs pass `--dangerously-bypass-hook-trust` or trust them once interactively.
- **Recall runs without asking.** Setup allows the plugin's MCP tools in `~/.codex/config.toml`, because under `approval_policy = "never"` an approval prompt is a refusal and `recall` would fail. Uninstalling removes the entry. To be asked each time, set `default_tools_approval_mode = "prompt"` there; setup keeps a mode you chose:

  ```toml
  [plugins."context-graph@context-graph-plugins".mcp_servers.context-graph]
  default_tools_approval_mode = "approve"
  ```

Codex hands MCP servers only selected environment variables, so `.mcp.json` forwards `CONTEXT_GRAPH_CONFIG`: a session pointed at another config file recalls from the same graph its hooks write to.

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

The Memgraph Context Graph skill is exposed as:

```text
context-graph:setup
```

This is a Codex marketplace only. Claude Code uses the separate marketplace at `.claude-plugin/marketplace.json`.
