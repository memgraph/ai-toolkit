# Context Graph

**Turn your coding-agent sessions (Claude Code, Codex, GitHub Copilot CLI, Cursor, OpenCode, Antigravity CLI, Grok Build) into a queryable knowledge graph in [Memgraph](https://memgraph.com/).**

Context Graph is a family of components that capture what your coding agents actually do — the tools they call, the skills they use, the memories they record — and persist it into a single Memgraph graph you can query across every session. Install it as a plugin, and every session your agent runs quietly builds up a graph of your work.

```text
Agent CLI hooks     ──────────▶  memgraph-context-graph  ──▶  Memgraph
(Claude Code, Codex, …)            (routes events)          (:User)-[:HAD_SESSION]->(:Session)
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
| **[memgraph-context-graph](./memgraph-context-graph/)** | The event hub. Normalizes runtime hooks/SDK activity into a shared event stream and routes it to the connectors below. | Adapters + connectors wiring |
| **[actions-graph](./actions-graph/)** | Tool calls, tool results, messages, subagent/error events — the raw activity of a session. | `(:Action)` |
| **[skills-graph](./skills-graph/)** | Which reusable [Agent Skills](https://docs.claude.com/en/docs/claude-code/skills) a session used, and stored skill definitions. | `(:Skill)` |
| **[resources-graph](./resources-graph/)** | GitHub resources touched by agents, their shared cache, and the `resource` MCP read tool. Enabled explicitly; see its installation guide. | `(:Resource)`, `(:Touch)`, `(:Listing)` |
| **[sessions-graph](./sessions-graph/)** | User/session provenance, durable **Memories**, and session **reconciliation** into entities. | `(:User)`, `(:Session)`, `(:Memory)` |
| **[context-graph-viewer](./viewer/)** | Nothing: a read-only terminal viewer (`memgraph-context-graph ui`) for capture health, pipeline state, sessions and the entity graph. | — |

> `(:Session)` is a shared coordination point — every component `MERGE`s it idempotently, but only sessions-graph owns `(:User)` and the `HAD_SESSION` edge. Entity extraction is powered by [unstructured2graph](../unstructured2graph/), which lives outside this family.

## How well does recall work?

Measured end to end on public long-memory benchmarks, through the shipped read path (hybrid recall over a graph built locally by GLiNER2, no LLM in the build or the recall):

| Benchmark | Answer score | Evidence recall | Retrieval tokens | Recall latency (mean) |
|---|---|---|---|---|
| LongMemEval-S, all 500 questions | **87.0%** | 0.92 | 6,465 | 2.85 s |
| BEAM 100K, all 400 questions | **63.6%** | 0.60 | 9,451 | 8.8 s |

Answer score is the share of the answer key the answer gets right, by each benchmark's own judge (LongMemEval's, BEAM's), so read the two rows as one metric on two judges.

The settings, per-type breakdowns and caveats are in the [eval README](./eval/README.md#official-numbers).

## Getting started

One command sets up every agent on the machine:

```bash
curl -fsSL https://raw.githubusercontent.com/memgraph/ai-toolkit/main/context-graph/scripts/install.sh | bash
# or, from a checkout:
./context-graph/scripts/install.sh
```

The script installs `uv` if it's missing and the `memgraph-context-graph` CLI with its three connectors, then
hands over to `memgraph-context-graph setup`. On a terminal setup is interactive: it shows the agents it found
on `PATH` (ticked), the project it will wire, your identity and the Memgraph it will use, updates the plan as
you change them, and changes nothing until you press Install. Without a terminal it prints the plan and runs it.

What it does, in order (each step checks first, so rerunning only does what's missing):

1. **Memgraph**: the one your config names, else one already on `localhost:7687`, else it starts
   `memgraph-mage` in a `context-graph-memgraph` container with a data volume. If 7687 is taken it picks
   another free port and writes it to the config. It never adopts the AI Memory demo's container, which is
   disposable, and it never touches a remote Memgraph that doesn't answer.
2. **Config**: `memgraph.url` and `identity.user_id` (default: git `user.name`, else `$USER`).
3. **Packages**: the connectors and the terminal viewer, if any are missing.
4. **Agents**: Claude Code and Codex get the Context Graph plugin, user-wide. Copilot CLI, Cursor, OpenCode,
   Antigravity CLI and Grok Build get project-local hooks in the current directory (never `$HOME`); your own
   hooks in those files are kept.
5. **Verify**: `doctor`'s checks, including each agent's hook probe.

The report ends with the exact command to rerun for anything that failed. Every choice is also a flag
(`memgraph-context-graph setup --help`), and the script passes its arguments on:

```bash
curl -fsSL .../install.sh | bash -s -- codex --memgraph-url bolt://my-host:7687
memgraph-context-graph setup cursor --project-dir ~/work/billing --project-dir ~/work/crawler
memgraph-context-graph setup --plan          # what it would do; changes nothing
memgraph-context-graph uninstall             # remove the wiring; your graph and config stay
```

Once it finishes, use the runtime normally — every session writes to the graph automatically: `(:User)-[:HAD_SESSION]->(:Session)`, tool actions, skill usage, and any Memories the agent records.

<details>
<summary>The same steps by hand</summary>

### 1. Start Memgraph

```bash
docker run --rm -p 7687:7687 memgraph/memgraph-mage:latest --schema-info-enabled=true
```

Requires **Memgraph ≥ 3.6** (sessions-graph uses text search, stable from that release). The MAGE image is recommended if you'll also use reconciliation.

### 2. Install the plugin

This is what wires the hooks into the runtime so sessions are actually captured. Inside Claude Code:

```text
/plugin marketplace add memgraph/ai-toolkit
/plugin install context-graph@context-graph-plugins
```

Or non-interactively, from any shell — Claude Code:

```bash
claude plugin marketplace add memgraph/ai-toolkit --sparse .claude-plugin
claude plugin install context-graph@context-graph-plugins -y
```

Codex:

```bash
codex plugin marketplace add memgraph/ai-toolkit --sparse .agents/plugins
codex plugin add context-graph@context-graph-plugins
```

### 3. Install the CLI and configure

```bash
uv tool install 'memgraph-context-graph[mcp]' \
  --with 'skills-graph[memgraph-context-graph]' \
  --with 'actions-graph[memgraph-context-graph]' \
  --with 'sessions-graph[memgraph-context-graph]'
```

Set your identity — **required** for sessions-graph to attach sessions to a user:

```bash
memgraph-context-graph config set identity.user_id "your-name"
```

The default Memgraph connection is `bolt://localhost:7687`. To point at a remote or HA instance, set it in the config file (hooks read this file, **not** environment variables, at runtime — see [ADR 0002](./memgraph-context-graph/docs/adr/0002-config-file-only-hook-resolution.md)):

```bash
memgraph-context-graph config set memgraph.url "neo4j://<coordinator-host>:7687"
memgraph-context-graph config set memgraph.user "<user>"
memgraph-context-graph config set memgraph.password        # prompts; stored 0600
memgraph-context-graph config set memgraph.database "memgraph"
```

### 4. Verify

```bash
memgraph-context-graph config show
memgraph-context-graph doctor --runtime claude-code \
  --connector skills-graph --connector actions-graph --connector sessions-graph
# Codex: --runtime codex
```

You want every line `OK` — config (user_id set), Memgraph reachable, each connector, and the strict hook smoke test.

### 5. Use the runtime normally

From here, every session writes to the graph automatically: `(:User)-[:HAD_SESSION]->(:Session)`, tool actions, skill usage, and any Memories the agent records.

</details>

See [memgraph-context-graph](./memgraph-context-graph/) for both runtimes' adapter details and for the in-process SDK path (no plugin).

## Turning sessions into an entity graph (reconciliation)

By default, a finished session is marked `reconciliation_status = 'pending'` — its content is captured but not yet extracted into entities (local extraction and the LLM summary run outside the hook timeout). Run it out-of-band to build the entity graph:

```bash
pip install "sessions-graph[reconciliation]"    # actions-graph + unstructured2graph + LightRAG
export OPENAI_API_KEY=...                        # or ANTHROPIC_API_KEY

# extract entities from every pending session
sessions-graph reconcile --pending
```

This pulls each session's actions and memories, runs them through [unstructured2graph](../unstructured2graph/)'s chunk + entity-extraction pipeline, and writes extracted entities (with typed labels like `:Person`, `:Organization`) linked back via `(:Action|:Memory)-[:HAS_CHUNK]->(:Chunk)<-[:MENTIONED_IN]-(:Entity)`. The `sessions-graph reconcile` CLI extracts locally with GLiNER2 over the user's adopted HyGM model, or `hygm.default_model()` when none is adopted. LightRAG supplies the separate LLM session summary. Programmatic callers can swap in another `ExtractionBackend` — see [sessions-graph § reconciliation](./sessions-graph/README.md#session-reconciliation).

## Seeing the graph

```bash
memgraph-context-graph ui
```

A read-only terminal viewer: whether capture is failing (the hooks' local failure log plus the graph), where reconciliation stands, each session's timeline and what it produced, and the entities and relations it learned. It needs the `context-graph-viewer` package in the tool environment; see [its README](./viewer/README.md).

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

Each component is a standalone Python library with its own README and can be used on its own or wired into `memgraph-context-graph` for the SDK path (Claude Agent SDK, OpenAI Agents SDK):

- [memgraph-context-graph](./memgraph-context-graph/) — the adapter/connector layer and SDK quick starts
- [actions-graph](./actions-graph/) — record and query session actions
- [skills-graph](./skills-graph/) — persist and track skills
- [sessions-graph](./sessions-graph/) — memories and reconciliation

## Local development

`scripts/dev-memgraph.sh` (repo root) manages an exploration Memgraph on port 7688 and a separate disposable test Memgraph on port 7689, and can point your live plugin at it for dogfooding:

```bash
./scripts/dev-memgraph.sh up        # start an isolated local Memgraph
./scripts/dev-memgraph.sh test      # Context Graph + HyGM + unstructured2graph on the test instance
./scripts/dev-memgraph.sh test-down # reclaim the disposable test instance
./scripts/dev-memgraph.sh reconcile # run reconciliation on pending sessions
./scripts/dev-memgraph.sh down      # tear it down
```
