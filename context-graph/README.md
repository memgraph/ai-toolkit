# Context Graph

**Turn your Claude Code and Codex sessions into a queryable knowledge graph in [Memgraph](https://memgraph.com/).**

Context Graph is a family of components that capture what your coding agents actually do — the tools they call, the skills they use, the memories they record — and persist it into a single Memgraph graph you can query across every session. Install it as a plugin, and every session your agent runs quietly builds up a graph of your work.

```text
Claude Code / Codex  ──hooks──▶  agent-context-graph  ──▶  Memgraph
                                   (routes events)          (:User)-[:HAD_SESSION]->(:Session)
                                                              ├─[:HAS_ACTION]->(:Action)   ← actions-graph
                                                              ├─[:USED_SKILL]->(:Skill)    ← skills-graph
                                                              └─[:PRODUCED_MEMORY]->(:Memory) ← sessions-graph
```

## Why

Every agent session today is ephemeral — it scrolls off and the context is gone. Context Graph makes that context durable and connected:

- **Observability** — see which tools and skills a session used, and how actions followed one another.
- **Cross-session recall** — agents write **Memories** (durable free-form assertions) they can search in future sessions.
- **A knowledge graph of your work** — optionally reconcile session content into extracted entities (people, projects, files, concepts) linked back to the sessions that mentioned them.

Everything is joined by a shared `(:Session {session_id})` node, so one graph answers questions that span tools, skills, memories, and entities.

## The components

| Component | What it captures | Owns |
|---|---|---|
| **[agent-context-graph](./agent-context-graph/)** | The event hub. Normalizes runtime hooks/SDK activity into a shared event stream and routes it to the connectors below. | Adapters + connectors wiring |
| **[actions-graph](./actions-graph/)** | Tool calls, tool results, messages, subagent/error events — the raw activity of a session. | `(:Action)`, `(:Tool)` |
| **[skills-graph](./skills-graph/)** | Which reusable [Agent Skills](https://docs.claude.com/en/docs/claude-code/skills) a session used, and stored skill definitions. | `(:Skill)` |
| **[sessions-graph](./sessions-graph/)** | User/session provenance, durable **Memories**, and session **reconciliation** into entities. | `(:User)`, `(:Session)`, `(:Memory)` |

> `(:Session)` is a shared coordination point — every component `MERGE`s it idempotently, but only sessions-graph owns `(:User)` and the `HAD_SESSION` edge. Entity extraction is powered by [unstructured2graph](../unstructured2graph/), which lives outside this family.

## Getting started (Claude Code)

The fastest path is the plugin — it wires all three connectors into Claude Code's hooks so capture is automatic.

### 1. Start Memgraph

```bash
docker run --rm -p 7687:7687 memgraph/memgraph-mage:latest --schema-info-enabled=true
```

Requires **Memgraph ≥ 3.6** (sessions-graph uses text search, stable from that release). The MAGE image is recommended if you'll also use reconciliation.

### 2. Install the plugin

Inside Claude Code:

```text
/plugin marketplace add memgraph/ai-toolkit
/plugin install context-graph@context-graph-plugins
```

### 3. Bootstrap and configure

The plugin's first run installs the `agent-context-graph` tool (with all three connector extras), writes `~/.config/context-graph/config.toml`, and runs `doctor`:

```bash
agent-context-graph bootstrap --runtime claude-code \
  --connector skills-graph --connector actions-graph --connector sessions-graph
```

Set your identity — **required** for sessions-graph to attach sessions to a user:

```bash
agent-context-graph config set identity.user_id "your-name"
```

The default Memgraph connection is `bolt://localhost:7687`. To point at a remote or HA instance, set it in the config file (hooks read this file, **not** environment variables, at runtime — see [ADR 0002](./agent-context-graph/docs/adr/0002-config-file-only-hook-resolution.md)):

```bash
agent-context-graph config set memgraph.url "neo4j://<coordinator-host>:7687"
agent-context-graph config set memgraph.user "<user>"
agent-context-graph config set memgraph.password        # prompts; stored 0600
agent-context-graph config set memgraph.database "memgraph"
```

### 4. Verify

```bash
agent-context-graph config show
agent-context-graph doctor --runtime claude-code \
  --connector skills-graph --connector actions-graph --connector sessions-graph
```

You want every line `OK` — config (user_id set), Memgraph reachable, each connector, and the strict hook smoke test.

### 5. Use Claude Code normally

From here, every session writes to the graph automatically: `(:User)-[:HAD_SESSION]->(:Session)`, tool actions, skill usage, and any Memories the agent records.

> **Codex** follows the same flow with `--runtime codex` and `codex plugin marketplace add memgraph/ai-toolkit --sparse .agents/plugins`. See [agent-context-graph](./agent-context-graph/) for both runtimes and for the in-process SDK path (no plugin).

## Turning sessions into an entity graph (reconciliation)

By default, a finished session is marked `reconciliation_status = 'pending'` — its content is captured but not yet extracted into entities (LLM extraction is too slow/costly to run inside a hook). Run it out-of-band to build the entity graph:

```bash
pip install "sessions-graph[reconciliation]"    # actions-graph + unstructured2graph + LightRAG
export OPENAI_API_KEY=...                        # or ANTHROPIC_API_KEY

# extract entities from every pending session
sessions-graph reconcile --pending
```

This pulls each session's actions and memories, runs them through [unstructured2graph](../unstructured2graph/)'s LightRAG pipeline, and writes extracted entities (with typed labels like `:Person`, `:Organization`) linked back via `(:Action|:Memory)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(:Entity)`. See [sessions-graph § reconciliation](./sessions-graph/README.md#session-reconciliation).

## Querying the graph

Because every component writes to the same `(:Session {session_id})`, one query language spans all of them:

| Question | Cypher path |
|---|---|
| All actions a user triggered | `(:User)-[:HAD_SESSION]->(:Session)-[:HAS_ACTION]->(:Action)` |
| Which skills a user has used | `(:User)-[:HAD_SESSION]->(:Session)-[:USED_SKILL]->(:Skill)` |
| Memories produced during a session | `(:Session)-[:PRODUCED_MEMORY]->(:Memory)` |
| All memories owned by a user | `(:User)-[:HAS_MEMORY]->(:Memory)` |
| Sessions where a specific tool was called | `(:Session)-[:HAS_ACTION]->(:Action {tool_name: "..."})` |
| Entities surfaced across a user's sessions | `(:User)-[:HAD_SESSION]->(:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(:Entity)` |

Run these in [Memgraph Lab](https://memgraph.com/docs/data-visualization) or any Bolt client.

## Using components directly (without a plugin)

Each component is a standalone Python library with its own README and can be used on its own or wired into `agent-context-graph` for the SDK path (Claude Agent SDK, OpenAI Agents SDK):

- [agent-context-graph](./agent-context-graph/) — the adapter/connector layer and SDK quick starts
- [actions-graph](./actions-graph/) — record and query session actions
- [skills-graph](./skills-graph/) — persist and track skills
- [sessions-graph](./sessions-graph/) — memories and reconciliation

## Local development

`scripts/dev-memgraph.sh` (repo root) starts an isolated Memgraph, runs each component's test suite against it, and can point your live plugin at it for dogfooding:

```bash
./scripts/dev-memgraph.sh up        # start an isolated local Memgraph
./scripts/dev-memgraph.sh test      # run all component test suites against it
./scripts/dev-memgraph.sh reconcile # run reconciliation on pending sessions
./scripts/dev-memgraph.sh down      # tear it down
```
