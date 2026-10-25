# Memgraph Context Graph

Connect coding agents and SDK applications to Context Graph. This package provides the CLI, runtime adapters, event routing, and MCP server; graph components store the activity and memory.

For an overview of what you can do with it, start with the [Context Graph guide](../README.md).

## Installation

For coding agents, the [installer](../README.md#getting-started) sets up the CLI, connectors, and agent wiring together. To install them yourself:

```bash
uv tool install 'memgraph-context-graph[mcp]' \
  --with 'skills-graph[memgraph-context-graph]' \
  --with 'actions-graph[memgraph-context-graph]' \
  --with 'sessions-graph[memgraph-context-graph]'
memgraph-context-graph setup
```

The `mcp` extra provides the MCP server that `memgraph-context-graph mcp` uses to serve `recall` and `memory` to agents.

For a Python application, install the extra for your SDK and the graph components you need; see [Memory for SDK apps](#memory-for-sdk-apps).

## Setup

```bash
memgraph-context-graph setup claude-code codex
memgraph-context-graph setup cursor --project-dir ~/work/billing
memgraph-context-graph setup --plan
memgraph-context-graph doctor --runtime codex
```

Setup connects to your configured Memgraph, or starts its own Docker container, writes your identity and connection settings, wires the selected agents, and checks the result. Claude Code and Codex use user-wide plugins; the other runtimes use project-local hooks and user-level MCP registrations.

```bash
memgraph-context-graph update codex      # upgrade and refresh wiring
memgraph-context-graph uninstall codex   # remove wiring; choose what else to remove
```

Restart running agents after an update. On a terminal, uninstall asks separately whether to delete setup's graph, the config, and the tool; those choices default to keeping them. Without a terminal it removes only wiring unless you supply removal flags. `uninstall --all` deletes all three, including the graph's data.

See [runtime setup](docs/runtime-setup.md) for the full lifecycle, manual wiring, and health checks.

## Memory for SDK apps

Apps built with the Claude Agent SDK or OpenAI Agents SDK can use the same memory as coding agents, in-process. The SDK integration adds capture, memory context, and `recall`/`memory` tools.

```python
from memgraph_context_graph.sdk import ContextGraphMemory

memory = ContextGraphMemory(user_id="alice")
```

[SDK quick starts](docs/sdk-memory.md) cover installation, wiring each SDK, and ending sessions.

## Recall: memory for the harness's model

The `recall` tool retrieves dated conversations and facts from the user's history. The agent's model answers from that evidence. You can also query it from the terminal:

```bash
memgraph-context-graph recall "What did we decide about the deploy window?"
```

See [recall and memory](docs/memory.md) for search settings, output, and runtime behavior.

## Memory: Context Graph as the harness's memory

Opt in to storing the agent's curated memory in Context Graph:

```bash
memgraph-context-graph setup codex --memory-backend context-graph
memgraph-context-graph setup codex --memory-backend native
```

The setting is user-wide. Setup imports supported existing memory before switching, then exposes the graph-backed `memory` tool and a session-start memory index. Import and hand-back behavior varies by agent; see the [memory guide](docs/memory.md#memory-context-graph-as-the-harnesss-memory).

## Configuration

Hooks read values from `~/.config/context-graph/config.toml`. `CONTEXT_GRAPH_CONFIG` can select another file; connection settings and API keys are read from that file, rather than environment variables.

```bash
memgraph-context-graph config show
memgraph-context-graph config set identity.user_id "your-name"
memgraph-context-graph config set memgraph.url "bolt://my-host:7687"
memgraph-context-graph config set memgraph.password  # prompts
```

The [configuration guide](docs/configuration.md) covers connectors, extraction, recall, and optional LLM settings.

## Capture privacy

Capture strips `<private>…</private>` text and redacts recognized credential patterns before storage. You can exclude projects or tools and add your own redaction patterns. Pattern matching cannot catch every secret. See [capture privacy](docs/configuration.md#capture-privacy).

## Forgetting sessions

```bash
memgraph-context-graph forget project . --dry-run
memgraph-context-graph forget session <session-id>
```

Forgetting deletes the session's captured activity and derived data that no other source supports. Explicit memory files remain. See [forgetting sessions](docs/configuration.md#forgetting-sessions) for deletion scope and protection against recapture.

## Architecture

```text
Runtime adapter → shared Event Protocol → graph connectors → Memgraph
```

Adapters translate runtime-specific activity. Connectors decide what to persist and live in the package that owns the graph schema. Runtime plugins distribute the hooks and MCP wiring.

| Read more | Covers |
|---|---|
| [Claude Code plugin](../plugins/memgraph-context-graph-claude/) | Marketplace install and plugin specifics |
| [Codex plugin](../plugins/memgraph-context-graph-codex/) | Marketplace install, hook trust, and tool approval |
| [Command-hook reference](docs/command-hooks.md) | Payloads, generated files, and runtime differences |
| [Architecture and extensions](docs/extending.md) | Events, custom connectors, and runtime registrations |
| [Actions Graph](../actions-graph/) | Tool calls, messages, and subagents |
| [Skills Graph](../skills-graph/) | Skill definitions and usage |
| [Sessions Graph](../sessions-graph/) | Memories, reconciliation, and recall |
| [Resources Graph](../resources-graph/) | Public GitHub resource caching |

## License

MIT
