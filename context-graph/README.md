# Context Graph

Memory and observability for your AI agents, stored in [Memgraph](https://memgraph.com/).

Context Graph connects what happens across sessions: conversations, tool calls, skill usage, and the memories agents choose to keep. Your agents can recall earlier work, and you can inspect the evidence behind it.

Works with **Claude Code, Codex, Copilot CLI, Cursor, OpenCode, Antigravity CLI, and Grok Build**, plus apps built with the Claude Agent SDK or OpenAI Agents SDK.

## Getting started

```bash
curl -fsSL https://raw.githubusercontent.com/memgraph/ai-toolkit/main/context-graph/scripts/install.sh | bash
```

Setup installs the CLI and connectors, lets you select agents, sets your identity, and checks the result. It uses the Memgraph configured for Context Graph or starts its own persistent Docker container. To pick agents or connect to a database you manage, pass setup's arguments through the pipe:

```bash
curl -fsSL https://raw.githubusercontent.com/memgraph/ai-toolkit/main/context-graph/scripts/install.sh \
  | bash -s -- codex --memgraph-url bolt://my-host:7687
```

Claude Code and Codex get user-wide plugins. The other agents get hooks in the project you select and an MCP server in their user configuration. Restart your agent after installation; Codex also asks you to trust the hooks.

Already have the CLI? You can select just the agents and projects you want:

```bash
memgraph-context-graph setup claude-code codex
memgraph-context-graph setup cursor --project-dir ~/work/billing
memgraph-context-graph doctor --runtime codex
```

Capture and recall need no LLM API key. Use Memgraph with MAGE for vector recall; setup uses the `memgraph/memgraph-mage` image. Memgraph 3.6 or newer is required.

## Give your agent lasting memory

Session history is captured automatically. To also make Context Graph the agent's curated memory store, opt in:

```bash
memgraph-context-graph setup codex --memory-backend context-graph
```

The agent gets a `memory` tool for durable notes and a `recall` tool for past conversations and facts. At session start, it sees a memory index and where you left off in the project. Memory follows the user across agents and checkouts of the same repository.

This setting is user-wide. You can return to the agent's built-in memory with `--memory-backend native`; import and hand-back support differs by agent. See [recall and memory](memgraph-context-graph/docs/memory.md).

For an application you are building, use the [SDK integration](memgraph-context-graph/docs/sdk-memory.md) to add capture, context, and memory tools in-process.

## See what was captured

```bash
memgraph-context-graph ui
```

The [terminal viewer](viewer/) shows capture health, session timelines, pending processing, and extracted entities. Setup installs it by default. You can also query the graph in Memgraph Lab or any Bolt client:

```cypher
MATCH (:User {user_id: "your-name"})-[:HAD_SESSION]->(s:Session)-[:HAS_ACTION|HAS_AGENT*1..2]->(a:ToolCall)
RETURN s.session_id, a.tool_name, count(*) AS calls
ORDER BY calls DESC;
```

## Turn conversations into a knowledge graph

Capture records activity immediately. Reconciliation extracts entities and relationships from that activity in a separate process, using local GLiNER2:

```bash
sessions-graph reconcile --pending
```

Setup does not install the `reconciliation` extra. For the installer's uv tool, add it by reinstalling the tool; `uv tool install` replaces the tool's packages, so name all of them:

```bash
uv tool install 'memgraph-context-graph[mcp]' \
  --with 'skills-graph[memgraph-context-graph]' \
  --with 'actions-graph[memgraph-context-graph]' \
  --with 'sessions-graph[memgraph-context-graph,reconciliation]' \
  --with context-graph-viewer \
  --with-executables-from sessions-graph
```

Later `update` runs keep the extra. No LLM key is needed for extraction; an optional key enables learning the user's ontology from their sessions.

For agents that emit a session-end event, `reconcile.auto_reconcile` can trigger this automatically. Turn ends still queue work for agents that do not emit session end, so use a scheduled `--pending` sweep to process those. See [reconciliation](sessions-graph/docs/reconciliation.md) for configuration, ontology versions, and embedding catch-up.

## The components

Most users install the family together. Each component is also a Python package you can use directly.

| Component | What it does |
|---|---|
| [memgraph-context-graph](memgraph-context-graph/) | CLI, runtime adapters, event routing, and the MCP server |
| [actions-graph](actions-graph/) | Records tool calls, results, messages, errors, and subagents |
| [skills-graph](skills-graph/) | Stores skill definitions and tracks which sessions use them |
| [sessions-graph](sessions-graph/) | Owns users, session provenance, memory files, reconciliation, and recall |
| [resources-graph](resources-graph/) | Remembers public GitHub resources an agent explored; opt-in |
| [context-graph-viewer](viewer/) | Read-only terminal viewer for capture health and the graph |
| [context-graph-spaces](spaces/) | Enterprise personal and shared spaces, with database-enforced access control |
| [context-graph-eval](eval/) | Recall benchmarks and graph-construction evaluation |

All capture components join on `(:Session {session_id})`. Sessions Graph owns the user relationship; other components attach their data to that session.

```text
User → Session → Actions and subagents
              → Skills used
              → Memories written
              → Chunks → Extracted entities and relationships
```

[unstructured2graph](../unstructured2graph/) supplies extraction, and [hygm](../hygm/) defines the ontology. Both are shared toolkit packages outside this family.

## How well does recall work?

Measured end to end on public long-memory benchmarks, through the shipped read path (hybrid recall over a graph built locally by GLiNER2, no LLM in the build or the recall):

| Benchmark | Answer score | Evidence recall | Retrieval tokens | Recall latency (mean) |
|---|---|---|---|---|
| LongMemEval-S, all 500 questions | **87.4%** | 0.96 | 5,863 | 1.56 s |
| BEAM 100K, all 400 questions | **63.1%** | 0.70 | 5,865 | 0.50 s |

Answer score is the share of the answer key the answer gets right, by each benchmark's own judge (LongMemEval's, BEAM's), so read the two rows as one metric on two judges.

The settings, per-type breakdowns and caveats are in the [eval README](./eval/README.md#official-numbers).

## Configuration and privacy

Hooks read `~/.config/context-graph/config.toml`. Use `config set` to change identity, connection settings, capture policy, or recall settings:

```bash
memgraph-context-graph config show
memgraph-context-graph config set identity.user_id "your-name"
memgraph-context-graph config set capture.exclude_projects "~/clients/*"
```

Capture removes `<private>…</private>` text and redacts recognized credential patterns before storage. Pattern matching cannot catch every secret. You can exclude tools or projects, and forget a captured session or project later. See [configuration and privacy](memgraph-context-graph/docs/configuration.md).

## Update or uninstall

```bash
memgraph-context-graph update
memgraph-context-graph uninstall
```

Update refreshes packages, wiring, and schema. Restart running agents afterwards. Uninstall removes agent wiring and restores built-in memory settings; on a terminal it asks separately about deleting the graph, config, and tool, defaulting to keeping them. `uninstall --all` deletes those too, including setup's graph data. See [runtime setup](memgraph-context-graph/docs/runtime-setup.md).

## Local development

Run from the repository root. The development script keeps exploration data separate from disposable test data:

```bash
./scripts/dev-memgraph.sh test
./scripts/dev-memgraph.sh test sessions-graph
./scripts/dev-memgraph.sh test-down
```

Use `./scripts/dev-memgraph.sh --help` for exploration, dogfooding, and Enterprise test commands.
