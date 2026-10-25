# Context Graph for Codex

Give Codex access to past sessions and durable memory in Memgraph. This plugin installs the hooks that capture activity and the MCP server that serves `recall` and, when enabled, `memory`.

## Get started

The [Context Graph installer](../../README.md#getting-started) installs the plugin and dependencies together. If the CLI is already installed:

```bash
memgraph-context-graph setup codex
memgraph-context-graph doctor --runtime codex
```

Setup connects to your configured Memgraph or starts its own Docker container, installs missing connectors, and sets your identity. Restart Codex afterwards.

## Install through the marketplace

```bash
codex plugin marketplace add memgraph/ai-toolkit --sparse .agents/plugins
codex plugin add context-graph@context-graph-plugins
```

The plugin supplies wiring; it also needs the Python CLI and connectors. From this plugin's directory, run `./scripts/bootstrap.sh` to install those and run setup, or use the [family installer](../../README.md#getting-started).

## Codex hook trust

Trust the plugin's hooks in Codex's startup review or `/hooks` before using it. Until trusted, hooks do not capture activity or add session-start context. `codex exec` never asks, so for unattended runs trust them once interactively or pass `--dangerously-bypass-hook-trust`.

Setup allows the plugin's MCP tools in `~/.codex/config.toml`, so recall works when approval prompts are disabled (under `approval_policy = "never"`, a prompt is a refusal). This is the entry setup writes, and uninstall removes it:

```toml
[plugins."context-graph@context-graph-plugins".mcp_servers.context-graph]
default_tools_approval_mode = "approve"
```

To be asked before each call, set it to `"prompt"`; setup keeps a mode you chose.

The plugin forwards `CONTEXT_GRAPH_CONFIG` to its MCP server so it reads from the same graph as the hooks.

## Enable curated memory

Recall searches your captured history. To also store the agent's long-term memory in Context Graph:

```bash
memgraph-context-graph setup codex --memory-backend context-graph
```

This is a user-wide setting. See the [memory guide](../../memgraph-context-graph/docs/memory.md) for import behavior, session-start context, and switching back.

## Configuration

Hooks read `~/.config/context-graph/config.toml`, rather than connection or credential environment variables:

```bash
memgraph-context-graph config set identity.user_id "your-name"
memgraph-context-graph config set memgraph.url "bolt://my-host:7687"
memgraph-context-graph config set memgraph.password  # prompts
```

Setup writes your identity; sessions-graph needs `identity.user_id` to attach sessions to you. See [configuration and privacy](../../memgraph-context-graph/docs/configuration.md) for capture controls and connection settings.

## Reconcile captured sessions

To extract entities and relationships, add the `reconciliation` extra ([how](../../README.md#turn-conversations-into-a-knowledge-graph)), then run:

```bash
sessions-graph reconcile --pending
```

Extraction uses local GLiNER2 and needs no LLM key. See [reconciliation](../../sessions-graph/docs/reconciliation.md) for automatic processing and ontology settings.

## Update or remove

```bash
memgraph-context-graph update codex
memgraph-context-graph uninstall codex
```

Restart the agent after updating. Uninstall removes the wiring and restores built-in memory settings. On a terminal, it also offers to delete the graph, config, and tool; each defaults to keeping it. See [runtime setup](../../memgraph-context-graph/docs/runtime-setup.md) for removal flags.

## Troubleshooting

Run `memgraph-context-graph doctor --runtime codex`. Check that the plugin is enabled, the CLI is on `PATH`, and Memgraph is reachable. For source development, `./scripts/doctor.sh` runs the same checks from this directory.

If this project has an older local hook file as well as the plugin, inspect it for duplicate or obsolete commands. The plugin calls `memgraph-context-graph hook run codex` and reads its connector selection from the shared config file.
