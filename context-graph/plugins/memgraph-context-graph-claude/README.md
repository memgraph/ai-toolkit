# Context Graph for Claude Code

Give Claude Code access to past sessions and durable memory in Memgraph. This plugin installs the hooks that capture activity and the MCP server that serves `recall` and, when enabled, `memory`.

## Get started

The [Context Graph installer](../../README.md#getting-started) installs the plugin and dependencies together. If the CLI is already installed:

```bash
memgraph-context-graph setup claude-code
memgraph-context-graph doctor --runtime claude-code
```

Setup connects to your configured Memgraph or starts its own Docker container, installs missing connectors, and sets your identity. Restart Claude Code afterwards.

## Install through the marketplace

```bash
claude plugin marketplace add memgraph/ai-toolkit --sparse .claude-plugin
claude plugin install context-graph@context-graph-plugins -y
```

The plugin supplies wiring; it also needs the Python CLI and connectors. From this plugin's directory, run `./scripts/bootstrap.sh` to install those and run setup, or use the [family installer](../../README.md#getting-started).

Inside Claude Code, you can also install with `/plugin marketplace add memgraph/ai-toolkit` followed by `/plugin install context-graph@context-graph-plugins`. The bundled setup skill is `/context-graph:setup`.

## Enable curated memory

Recall searches your captured history. To also store the agent's long-term memory in Context Graph:

```bash
memgraph-context-graph setup claude-code --memory-backend context-graph
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
memgraph-context-graph update claude-code
memgraph-context-graph uninstall claude-code
```

Restart the agent after updating. Uninstall removes the wiring and restores built-in memory settings. On a terminal, it also offers to delete the graph, config, and tool; each defaults to keeping it. See [runtime setup](../../memgraph-context-graph/docs/runtime-setup.md) for removal flags.

## Troubleshooting

Run `memgraph-context-graph doctor --runtime claude-code`. Check that the plugin is enabled, the CLI is on `PATH`, and Memgraph is reachable. For source development, `./scripts/doctor.sh` runs the same checks from this directory.

If this project has an older local hook file as well as the plugin, inspect it for duplicate or obsolete commands. The plugin calls `memgraph-context-graph hook run claude-code` and reads its connector selection from the shared config file.
