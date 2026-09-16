# Agent Context Graph

Agent Context Graph is the adapter layer for Context Graph components. It normalizes agent SDK and runtime activity into events and routes those events to graph connectors.

## Language

**Agent Context Graph**:
The adapter layer that normalizes agent SDK and runtime activity into events and routes those events to graph connectors.
_Avoid_: Graph storage, skills graph, memory store

**Graph Connector**:
The integration point owned by a graph component that decides which normalized events matter to that component and persists their meaning.
_Avoid_: Adapter, hook, writer

**Agent Development SDK**:
A framework used to build agent applications, where integrations usually attach through in-process callbacks or hook objects.
_Avoid_: Runtime hook, command hook

**Runtime Hook**:
A hook emitted by an agent runtime around an already-running agent session, often by invoking an external command with a JSON payload.
_Avoid_: Agent development SDK

**Runtime Adapter**:
The integration point that translates one agent development SDK or runtime hook shape into the shared event protocol.
_Avoid_: Graph connector, storage adapter

**Runtime Plugin**:
A host-specific distribution package that installs runtime hooks, skills, commands, setup helpers, or other host-native integration files.
_Avoid_: Graph component, graph connector, storage layer

**Codex Plugin**:
A Runtime Plugin for OpenAI Codex that wires Codex lifecycle hooks to an Agent Context Graph runtime adapter entrypoint.
_Avoid_: Codex graph, skills graph plugin

**Claude Code Plugin**:
A Runtime Plugin for Claude Code that wires Claude Code lifecycle hooks to an Agent Context Graph runtime adapter entrypoint.
_Avoid_: Claude graph, skills graph plugin

**Event Protocol**:
The runtime-agnostic set of agent activity events emitted by runtime adapters and consumed by graph connectors.
_Avoid_: Graph event, graph protocol, hook payload, callback data

**Agent Session**:
The runtime-side unit of agent activity that groups related events under a shared session identifier.
_Avoid_: Session node

**Hook Configuration**:
A persistent TOML file that supplies identity and connection settings to hook
subprocesses. The default path is `~/.config/context-graph/config.toml`;
`CONTEXT_GRAPH_CONFIG` may select another file. At hook runtime, config values
come only from the selected file. The environment variable selects a file; it
never supplies a value from that file.
_Avoid_: env config, runtime config, shell config

**Runtime Registration**:
A Python object implementing `RuntimeCLIPlugin`. A package publishes it through
the `agent_context_graph.runtimes` entry-point group. The CLI resolves a runtime
name through this object to find its adapter, hook response, hook config, and
optional initializer.

This is not a **Runtime Plugin**. The plugin installs host-facing files; the
registration lets Agent Context Graph discover runtime support through
`importlib.metadata.entry_points()`. Adding one needs no central registry change.
_Avoid_: Runtime Plugin (already means the distribution package — see above), Runtime Adapter (a Runtime Registration *references* one via `adapter_class`, it isn't one)

## Relationships

- **Agent Context Graph** belongs to the broader **Context Graph** family.
- **Agent Context Graph** routes events to graph connectors owned by graph components.
- A **Graph Connector** consumes normalized events emitted by **Agent Context Graph**.
- An **Agent Development SDK** integration and a **Runtime Hook** integration both use a **Runtime Adapter** to emit the same event protocol.
- A **Runtime Plugin** is a deployment surface for runtime hooks and setup helpers; it is not a graph component.
- A **Codex Plugin** installs Codex hook wiring that invokes the Codex runtime adapter command.
- A **Claude Code Plugin** installs Claude Code hook wiring that invokes the Claude Code runtime adapter command.
- The **Event Protocol** carries agent activity, not graph semantics.
- An **Agent Session** may be persisted as a session node by a graph component, but the **Event Protocol** only carries the session identifier.
- Tool and message events carry `agent_name` when the runtime identifies a
  subagent. Agent Context Graph only transports this id; Graph Connectors decide
  how to use it. The Codex adapter does not yet handle subagent lifecycle or ids
  ([#275](https://github.com/memgraph/ai-toolkit/issues/275)).
- A Runtime Plugin's generated command calls `hook run <name>`. The CLI resolves
  that name through a Runtime Registration; the plugin does not name an adapter
  class directly.
- Built-in Runtime Registrations are `codex` and `claude-code`. Other packages
  may publish more.

## Example dialogue

> **Dev:** "Should Agent Context Graph write the skill usage edge?"
> **Domain expert:** "No. Agent Context Graph emits the tool event; Skills Graph decides whether that event represents skill usage."

> **Dev:** "Are OpenAI Agents SDK and Codex hooks the same kind of integration?"
> **Domain expert:** "No. OpenAI Agents SDK is for agent development; Codex hooks are runtime hooks. They both become normalized events before graph connectors see them."

> **Dev:** "Should the Codex plugin know how to write skill usage?"
> **Domain expert:** "No. The plugin installs hook wiring. The Codex runtime adapter emits events. Skills Graph decides whether those events mean skill usage."

## Flagged ambiguities

- "graph" can mean the umbrella **Context Graph** family or a specific graph component. Resolved: use **Agent Context Graph** only for the adapter layer, not for persistence.
- "connector" can sound like generic transport plumbing. Resolved: a **Graph Connector** owns graph-specific event interpretation.
- "adapter" can refer to SDK integrations or runtime hooks. Resolved: use **Runtime Adapter** for anything that translates agent activity into the shared event protocol.
- "SDK adapter" is too narrow because Codex command hooks are runtime hooks rather than agent development SDK callbacks. Resolved: the code and docs use **Runtime Adapter**.
- "event" should not use graph nomenclature. Resolved: graph meaning is assigned by **Graph Connectors**, not by the **Event Protocol**.
- "Session" can mean activity or graph state. Use **Agent Session** for activity
  and "session node" for persisted state.
- "Plugin" can sound like a graph extension. Use **Runtime Plugin** for a
  host-specific distribution; Graph Connectors own graph interpretation.
- Entry-point discovery is **Runtime Registration**, not a runtime or CLI plugin
  ([#269](https://github.com/memgraph/ai-toolkit/issues/269)).
