# Context Map

## Language

**Context Graph**:
Graph-backed component family. Records agent activity (**Collection Tier**), keeps what agents save on purpose (**Memory**), and lets an agent recall both through an **Index**. Built for agents/runtimes, not mainly humans reviewing sessions.
_Avoid_: Knowledge graph (names tech underneath, not goal)

**Collection Tier**:
Agent activity captured at hook time, e.g. Actions Graph's Action and Agent nodes. Its messages are the user's **History**. Per-component data + planned retention rules: each component's own `CONTEXT.md` ([#267](https://github.com/memgraph/ai-toolkit/issues/267)).
_Avoid_: Memory, raw memory

**Memory**:
Curated text an agent saves on purpose, as a file under `/memories` through the `memory` tool (Sessions Graph's `(:Memory)`). The only thing called memory, and the only thing a Shared Space holds.
_Avoid_: Semantic memory, extracted insight, session summary

**History**:
A user's past turns. Recall answers from them directly: they are the evidence, not raw input waiting to be distilled.
_Avoid_: Episode, session summary

**Index**:
Chunks, entities and typed facts extracted from History and Memory with local GLiNER2, no LLM. Recall uses it to find the turns and files that answer a question; it is not a kind of memory.
_Avoid_: Semantic memory, knowledge graph (as if it replaced its sources)

**Procedure** (designed, not implemented):
Reusable procedure mined from repeated work; see Skills Graph's `CONTEXT.md`.

**LLM key**:
Optional. Nothing in recording, extraction, embedding or recall calls an LLM; the harness's own model answers from what recall returns. A key only enables ontology derivation, which learns the user's own relation types.
_Avoid_: required credential

## Contexts

- [Memgraph Context Graph](./memgraph-context-graph/CONTEXT.md) — translates agent SDK/runtime hook activity into shared event stream, routes to graph connectors. Also serves components' read tools (recall, resource) to the model over stdio MCP.
- [Skills Graph](./skills-graph/CONTEXT.md) — stores reusable agent skills in Memgraph, records session skill usage.
- [Actions Graph](./actions-graph/CONTEXT.md) — records agent session activity as action nodes for observability + tool usage analytics.
- [Resources Graph](./resources-graph/CONTEXT.md) — remembers the public GitHub issues, PRs and repositories an agent explored as shared Resources, and serves them back through the `resource` tool.
- [Sessions Graph](./sessions-graph/CONTEXT.md) — stores Memory files, indexes History and Memory into extracted entities, serves recall. Named `sessions-graph`: `memory-graph` taken on PyPI. Domain = memory recall.
- [Context Graph Spaces](./spaces/CONTEXT.md) — Enterprise edition: personal + shared spaces for many users on one licensed Memgraph, isolated by Memgraph access control.
- [Context Graph Viewer](./viewer/README.md) — read-only terminal views of capture health, pipeline state, sessions and the knowledge graph. Reads what the others write; never writes.
- [Context Graph Eval](./eval/CONTEXT.md) — measures whether recall finds what History and Memory hold. Not part of the pipeline it measures.

## Relationships

- **Memgraph Context Graph -> graph components**: emits normalized events. Each component-owned connector decides what to persist. `ActionsGraphConnector` records observability data, `SkillGraphConnector` records skill activity, `SessionsGraphConnector` records session/user lifecycle data.
- **Session Node (shared)**: `(:Session {session_id})` joins component data. Any component may `MERGE` it; none owns it. Adapter sets `session_id`.
- **Actions Graph -> Skills Graph**: both consume tool activity. Actions Graph records it; Skills Graph decides if it proves skill usage.
- **Actions Graph -> Sessions Graph**: reconciliation reads top-level + subagent actions via `get_session_actions()`. Never modifies Actions Graph data.
- **Actions Graph -> Skills Graph (`Agent` nodes)**: Actions Graph owns each subagent's `Agent` node. Skills Graph may attach `USED_SKILL` to it by `agent_id` — mirrors `HAS_ACTION`/`USED_SKILL` on a top-level Session.
- **Sessions Graph -> [unstructured2graph](../unstructured2graph/CONTEXT.md)**: `SESSION_END`/`TURN_END` only mark Session `reconciliation_status = 'pending'`. Later, out-of-band reconciliation run sends Action + Memory content to `unstructured2graph`: links source nodes to `Chunk` nodes via `HAS_CHUNK`, builds entity graph. No LLM. `unstructured2graph` is outside the Context Graph family.
- **Actions/Sessions Graph -> Context Graph Eval**: Eval Batch stages Action fixtures + runs Session Reconciliation in dedicated Memgraph instance. Retrieval + scoring then read resulting Action/Chunk/entity data, no modification.

## Cross-component graph patterns

`session_id` joins all component data via same `Session` node.

| Query goal | Graph path |
|---|---|
| All actions triggered by a user | `User -> Session -> Action`, plus `User -> Session -> Agent -> Action` for subagents |
| Which skills a user has used | `User -> Session -> Skill`, plus `User -> Session -> Agent -> Skill` for subagents |
| Memories produced during a session | `(:Session)-[:PRODUCED_MEMORY]->(:Memory)` |
| All memories owned by a user | `(:User)-[:HAS_MEMORY]->(:Memory)` |
| Sessions where a specific tool was called | `Session -> Action`, or `Session -> Agent -> Action`, filtered by `Action.tool_name` |
| Entities surfaced in a user's sessions | `User -> Session -> Action|Memory -> Chunk <- Entity`; use `Session -> Agent -> Action` for subagent sources |

Sessions Graph alone owns `User` nodes + `HAD_SESSION`. Other user-scoped queries start there.

## Testing

One rule for Context Graph packages + `unstructured2graph`: test graph persistence against real Memgraph. Mocks OK for pure event translation, model validation, unreachable-DB error branches, LLM boundaries. Never assert on Cypher strings sent to a mock. Full rationale + commands: [AGENTS.md](../AGENTS.md#testing-policy-prefer-real-memgraph-over-mocks).
