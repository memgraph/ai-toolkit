# Agent Context Graph

Adapter layer for Context Graph components. Normalizes agent SDK + runtime activity into events, routes to graph connectors.

## Language

**Agent Context Graph**:
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
Runtime Plugin for OpenAI Codex; wires Codex lifecycle hooks to an Agent Context Graph runtime adapter entrypoint.
_Avoid_: Codex graph, skills graph plugin

**Claude Code Plugin**:
Runtime Plugin for Claude Code; wires Claude Code lifecycle hooks to an Agent Context Graph runtime adapter entrypoint.
_Avoid_: Claude graph, skills graph plugin

**Event Protocol**:
Runtime-agnostic set of agent activity events, emitted by runtime adapters, consumed by graph connectors.
_Avoid_: Graph event, graph protocol, hook payload, callback data

**Agent Session**:
Runtime-side unit of agent activity grouping related events under shared session identifier.
_Avoid_: Session node

**Hook Configuration**:
Persistent TOML file supplying identity + connection settings to hook subprocesses. Default path `~/.config/context-graph/config.toml`; `CONTEXT_GRAPH_CONFIG` may select another file. At hook runtime, config values come only from selected file. Env var selects file; never supplies a value from it.
_Avoid_: env config, runtime config, shell config

**Runtime Registration**:
Python object implementing `RuntimeCLIPlugin`. Package publishes it via `agent_context_graph.runtimes` entry-point group. CLI resolves runtime name through it to find adapter, hook response, hook config, optional initializer.

Not a **Runtime Plugin**: plugin installs host-facing files; registration lets Agent Context Graph discover runtime support via `importlib.metadata.entry_points()`. Adding one needs no central registry change.
_Avoid_: Runtime Plugin (already means distribution package), Runtime Adapter (a Runtime Registration *references* one via `adapter_class`, isn't one)

## Relationships

- **Agent Context Graph** belongs to broader **Context Graph** family.
- **Agent Context Graph** routes events to graph connectors owned by graph components.
- **Graph Connector** consumes normalized events emitted by **Agent Context Graph**.
- **Agent Development SDK** integration + **Runtime Hook** integration both use a **Runtime Adapter** to emit same event protocol.
- **Runtime Plugin** = deployment surface for runtime hooks/setup helpers; not a graph component.
- **Codex Plugin** installs Codex hook wiring invoking the Codex runtime adapter command.
- **Claude Code Plugin** installs Claude Code hook wiring invoking the Claude Code runtime adapter command.
- **Event Protocol** carries agent activity, not graph semantics.
- **Agent Session** may be persisted as a session node by a graph component; **Event Protocol** only carries the session identifier.
- Tool/message events carry `agent_name` when runtime identifies a subagent. Agent Context Graph only transports this id; Graph Connectors decide how to use it. Codex adapter doesn't yet handle subagent lifecycle/ids ([#275](https://github.com/memgraph/ai-toolkit/issues/275)).
- Runtime Plugin's generated command calls `hook run <name>`. CLI resolves name via Runtime Registration; plugin never names an adapter class directly.
- Built-in Runtime Registrations: `codex`, `claude-code`. Other packages may publish more.

## Example dialogue

> **Dev:** "Should Agent Context Graph write the skill usage edge?"
> **Domain expert:** "No. Agent Context Graph emits the tool event; Skills Graph decides whether that event represents skill usage."

> **Dev:** "Are OpenAI Agents SDK and Codex hooks the same kind of integration?"
> **Domain expert:** "No. OpenAI Agents SDK is for agent development; Codex hooks are runtime hooks. Both become normalized events before graph connectors see them."

> **Dev:** "Should the Codex plugin know how to write skill usage?"
> **Domain expert:** "No. Plugin installs hook wiring. Codex runtime adapter emits events. Skills Graph decides if those events mean skill usage."

## Flagged ambiguities

- "graph" = umbrella **Context Graph** family or specific component? Resolved: **Agent Context Graph** = adapter layer only, never persistence.
- "connector" can sound like generic transport plumbing. Resolved: **Graph Connector** owns graph-specific event interpretation.
- "adapter" can mean SDK integrations or runtime hooks. Resolved: **Runtime Adapter** = anything translating agent activity into shared event protocol.
- "SDK adapter" too narrow: Codex command hooks are runtime hooks, not SDK callbacks. Resolved: code/docs use **Runtime Adapter**.
- "event" shouldn't carry graph nomenclature. Resolved: graph meaning assigned by **Graph Connectors**, not **Event Protocol**.
- "Session" = activity or graph state? Use **Agent Session** for activity, "session node" for persisted state.
- "Plugin" can sound like a graph extension. Use **Runtime Plugin** for host-specific distribution; Graph Connectors own graph interpretation.
- Entry-point discovery = **Runtime Registration**, not a runtime/CLI plugin ([#269](https://github.com/memgraph/ai-toolkit/issues/269)).
- Code hasn't fully caught up to **Runtime Adapter** terminology: Event Protocol's `source_sdk` field (`events.py`, set by all 4 adapters, read by Actions Graph's connector) still names the rejected term (ADR 0001). Rename to e.g. `source_adapter` tracked, not done.
