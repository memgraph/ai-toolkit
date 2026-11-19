# Memgraph Context Graph

Adapter layer for Context Graph components. Normalizes agent SDK + runtime activity into events, routes to graph connectors.

## Language

**Memgraph Context Graph**:
Adapter layer normalizing agent SDK + runtime activity into events, routing to graph connectors.
_Avoid_: Graph storage, skills graph, memory store

**Graph Connector**:
Integration point, owned by a graph component, deciding which normalized events matter to it + persisting their meaning.
_Avoid_: Adapter, hook, writer

**Agent Development SDK**:
Framework for building agent apps; integrations usually attach via in-process callbacks/hook objects.
_Avoid_: Runtime hook, command hook

**Runtime Hook**:
Hook emitted by an agent runtime around an already-running session, often via external command + JSON payload.
_Avoid_: Agent development SDK

**Runtime Adapter**:
Integration point translating one SDK/runtime-hook shape into shared event protocol.
_Avoid_: Graph connector, storage adapter

**Runtime Plugin**:
Host-specific distribution package installing runtime hooks, skills, commands, setup helpers, other host-native integration files.
_Avoid_: Graph component, graph connector, storage layer

**Codex Plugin**:
Runtime Plugin for OpenAI Codex; wires Codex lifecycle hooks to an Memgraph Context Graph runtime adapter entrypoint.
_Avoid_: Codex graph, skills graph plugin

**Claude Code Plugin**:
Runtime Plugin for Claude Code; wires Claude Code lifecycle hooks to an Memgraph Context Graph runtime adapter entrypoint.
_Avoid_: Claude graph, skills graph plugin

**Event Protocol**:
Runtime-agnostic set of agent activity events, emitted by runtime adapters, consumed by graph connectors.
_Avoid_: Graph event, graph protocol, hook payload, callback data

**Agent Session**:
Runtime-side unit of agent activity grouping related events under shared session identifier.
_Avoid_: Session node

**Turn End**:
Agent finished one turn (one prompt's work) inside a still-open **Agent Session**. Distinct from session end: most runtimes fire a stop hook after every turn, and several (Codex, Antigravity, `opencode run`) fire nothing at session end. Graph Connectors mark the session for later reconciliation on a Turn End but only start it on session end.
_Avoid_: Stop (a runtime's hook name, not a protocol term), session end

**Hook Configuration**:
Persistent TOML file supplying identity + connection settings to hook subprocesses. Default path `~/.config/context-graph/config.toml`; `CONTEXT_GRAPH_CONFIG` may select another file. At hook runtime, config values come only from selected file. Env var selects file; never supplies a value from it. Its keys are one table (`config_schema.CONFIG_KEYS`): parsing, the written file, and `config get/set/show` all follow it.
_Avoid_: env config, runtime config, shell config

**Edition**:
Which multi-user model the **Hook Configuration** selects, by its top-level `edition` key. **Community** (the default): one user, one Memgraph database. **Enterprise**: many users on a shared Memgraph, isolation enforced by Memgraph Enterprise access control, implemented by the separate `context-graph-spaces` package. Chosen explicitly, never detected from the server's licence (map #501).
_Avoid_: tier, plan, licence (commercial terms, not behaviour)

**Space**:
A place memory lives. Each user has one **Personal Space**, where hooks write all their activity. Enterprise adds **Shared Spaces** that a set of members can read. A space is one Memgraph database: Memgraph can't query across databases, so reading several spaces means one query per space.
_Avoid_: tenant (admin vocabulary), workspace, team (a team is just one kind of Shared Space)

**Space Resolver**:
`SpaceResolver` (`spaces.py`). The one place a component asks who the user is, which space to write to (`personal()`), and which spaces to read (`readable()`). `spaces_for(config)` returns the configured **Edition**'s resolver. Components never read `[identity]` or `[memgraph]` themselves.
_Avoid_: reading `HookConfig.user_id` / `memgraph_*` directly

**Runtime Registration**:
Python object implementing `RuntimeCLIPlugin`. Package publishes it via `memgraph_context_graph.runtimes` entry-point group. CLI resolves runtime name through it to find adapter, hook response, hook config, optional initializer.

Not a **Runtime Plugin**: plugin installs host-facing files; registration lets Memgraph Context Graph discover runtime support via `importlib.metadata.entry_points()`. Adding one needs no central registry change.
_Avoid_: Runtime Plugin (already means distribution package), Runtime Adapter (a Runtime Registration *references* one via `adapter_class`, isn't one)

**Tool Registration**:
Object implementing `Tool` (`tools.py`): a name, a description the model reads, an input schema, the connector whose graph it reads, an optional session-start hint, and `call(arguments, config)`. A graph component publishes it via the `memgraph_context_graph.tools` entry-point group; `memgraph-context-graph mcp` serves every registered tool, and the CLI runs one by name. Identity + connection come from the **Space Resolver** over **Hook Configuration**, never from tool arguments.
_Avoid_: MCP tool (the protocol it is served over, not the thing), Graph Connector (writes events; a tool reads)

**Connector Registration**:
`ConnectorSpec` (`connectors.py`): a graph component's name, its graph class, how to build its **Graph Connector** from **Hook Configuration**, and an optional `doctor` check. Published via the `memgraph_context_graph.connectors` entry-point group; the hook runner, schema setup and `doctor` find connectors only through it. The hub keeps just what it needs before a component is installed: each installable connector's requirement (`INSTALLABLE`) and the default set.
_Avoid_: Component (the package), Graph Connector (the object a registration builds)

**Graph Epoch**:
The version of one component's shape in the graph, an integer that only a migration raises; every graph is at epoch 1 until one does. The graph states per component the oldest epoch that may write and read it, `(:ContextGraphSchema {component, epoch, min_writer_epoch, min_reader_epoch})`, and every client checks that before writing or reading (`graph_version.py`): a client older than the graph stops, never corrupts. No node means no constraint. Clients record themselves as `(:ContextGraphClient)`, so a migration can see who still writes an old shape. Not a package version: a release that leaves the graph's shape alone keeps the epoch.
_Avoid_: schema version, package version, migration (raises an epoch; isn't one)

**Refused-Event Spool**:
Where a connector's events go while the graph refuses it (**Graph Epoch**): a local, 0600, size-capped JSONL file beside the capture-failure log (`refused_events.py`), holding each event as the capture policy left it, original timestamp included. `update` replays it into the connectors the graph allows again. Its line format is versioned and read by later releases, so a change keeps a reader for every earlier format.
_Avoid_: queue, buffer, outbox (nothing sends it on its own; `update` replays it)

## Relationships

- **Memgraph Context Graph** belongs to broader **Context Graph** family.
- **Memgraph Context Graph** routes events to graph connectors owned by graph components.
- **Graph Connector** consumes normalized events emitted by **Memgraph Context Graph**.
- **Agent Development SDK** integration + **Runtime Hook** integration both use a **Runtime Adapter** to emit same event protocol.
- **Runtime Plugin** = deployment surface for runtime hooks/setup helpers; not a graph component.
- **Codex Plugin** installs Codex hook wiring invoking the Codex runtime adapter command.
- **Claude Code Plugin** installs Claude Code hook wiring invoking the Claude Code runtime adapter command.
- **Event Protocol** carries agent activity, not graph semantics.
- **Agent Session** may be persisted as a session node by a graph component; **Event Protocol** only carries the session identifier.
- Tool/message events carry `agent_name` when runtime identifies a subagent. Memgraph Context Graph only transports this id; Graph Connectors decide how to use it. The Codex adapter emits no subagent lifecycle events.
- Runtime Plugin's generated command calls `hook run <name>`. CLI resolves name via Runtime Registration; plugin never names an adapter class directly.
- Built-in Runtime Registrations: `codex`, `claude-code`, `copilot-cli`, `cursor`, `opencode`, `antigravity-cli`, and `grok`. Other packages may publish more.

## Example dialogue

> **Dev:** "Should Memgraph Context Graph write the skill usage edge?"
> **Domain expert:** "No. Memgraph Context Graph emits the tool event; Skills Graph decides whether that event represents skill usage."

> **Dev:** "Are OpenAI Agents SDK and Codex hooks the same kind of integration?"
> **Domain expert:** "No. OpenAI Agents SDK is for agent development; Codex hooks are runtime hooks. Both become normalized events before graph connectors see them."

> **Dev:** "Should the Codex plugin know how to write skill usage?"
> **Domain expert:** "No. Plugin installs hook wiring. Codex runtime adapter emits events. Skills Graph decides if those events mean skill usage."

## Flagged ambiguities

- "graph" = umbrella **Context Graph** family or specific component? Resolved: **Memgraph Context Graph** = adapter layer only, never persistence.
- "connector" can sound like generic transport plumbing. Resolved: **Graph Connector** owns graph-specific event interpretation.
- "adapter" can mean SDK integrations or runtime hooks. Resolved: **Runtime Adapter** = anything translating agent activity into shared event protocol.
- "SDK adapter" too narrow: Codex command hooks are runtime hooks, not SDK callbacks. Resolved: code/docs use **Runtime Adapter**.
- "event" shouldn't carry graph nomenclature. Resolved: graph meaning assigned by **Graph Connectors**, not **Event Protocol**.
- "Session" = activity or graph state? Use **Agent Session** for activity, "session node" for persisted state.
- "Plugin" can sound like a graph extension. Use **Runtime Plugin** for host-specific distribution; Graph Connectors own graph interpretation.
- Entry-point discovery = **Runtime Registration**, not a runtime/CLI plugin ([#269](https://github.com/memgraph/ai-toolkit/issues/269)).
- Code hasn't fully caught up to **Runtime Adapter** terminology: Event Protocol's `source_sdk` field (`events.py`, set by every adapter, read by Actions Graph's connector) still names the rejected term (ADR 0001); a rename to e.g. `source_adapter` is not done.
