# The model may write explicit Memory files over MCP; everything derived stays background-only

Map #259 decided the MCP surface is read-only: all memory is written by hooks and background reconciliation, never by a model calling a tool. This ADR narrows that rule for one node type. It doesn't reverse it.

## Context

Context Graph should be able to replace a harness's built-in memory (map #484). Neither Claude Code nor Codex has a pluggable memory backend. Claude Code's auto memory is plain Markdown files the model writes with its ordinary file tools. Codex only exposes on/off memory controls. Replacing either means the model needs somewhere to write the memories it chooses to keep.

The Claude API memory tool (`memory_20250818`) defines that contract: `view`/`create`/`str_replace`/`insert`/`delete`/`rename` on a `/memories` file tree, with fixed result and error strings. `sessions-graph`'s CONTEXT had deferred MCP memory writes "until the write contract is clear". This is that contract.

## Decision

- The MCP `memory` tool may write `(:Memory)` nodes, one per file, and nothing else.
- Everything derived stays background-only: chunks, entities, Episodes, Procedures. Only reconciliation writes them, and the tool never does.
- `recall` stays read-only.
- The tool exists only for a user whose config says `[memory] backend = "context-graph"`, because it replaces the harness's memory rather than sitting beside it.

## Why the line falls here

#259 guarded against an LLM hand-authoring the graph's extracted structure. That structure must come from a repeatable pipeline over captured activity. A Memory was always different: it was defined as "an assertion an agent deliberately saves for later sessions." Writing it is a deliberate act by the model in any design. Before this ADR it went through the Python API; now the model can also do it over MCP.

The two write paths don't compete. Reconciliation processes everything that happened after the session ends. The tool records what the model judged worth keeping, at the moment it judged so.

## Consequences

- `(:Memory)` gains `path` (unique per user) and `updated_at`. Every Memory has a path; `save_memory()` without one files under `/memories/notes/<id>.md`.
- Path checks live in one place, `sessions_graph.memory_store.MemoryStore`, shared by every entry point.
- Identity never comes from the call. The MCP tool reads the user from the config file. An API wrapper takes it from the application's own code.
