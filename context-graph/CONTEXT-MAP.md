# Context Map

## Language

**Context Graph**:
Graph-backed component family. Raw agent activity (**Collection Tier**) -> durable reusable knowledge (**Memory Tier**). Built for agents/runtimes, not mainly humans reviewing sessions.
_Avoid_: Knowledge graph (names tech underneath, not goal)

**Collection Tier**:
Raw, short-lived input captured at hook time or read by a memory builder. E.g. Actions Graph's Action Nodes, source text for reconciliation/procedure mining. Not memory until distilled. Per-component data + planned retention rules: each component's own `CONTEXT.md` ([#267](https://github.com/memgraph/ai-toolkit/issues/267)).
_Avoid_: Memory, raw memory

**Memory Tier**:
Durable knowledge, derived from Collection-Tier Data or saved explicitly. Three forms:

- **Semantic**: extracted entities/relationships, or explicit `Memory` assertion. `Chunk` nodes preserve source text + provenance; not memory themselves.
- **Episodic**: an `Episode` summarizing one session.
- **Procedural**: reusable `Procedure` mined from repeated work. Designed, not implemented; see Skills Graph's `CONTEXT.md`.

_Avoid_: Collection tier, raw data

## Contexts

- [Agent Context Graph](./agent-context-graph/CONTEXT.md) — translates agent SDK/runtime hook activity into shared event stream, routes to graph connectors.
- [Skills Graph](./skills-graph/CONTEXT.md) — stores reusable agent skills in Memgraph, records session skill usage.
- [Actions Graph](./actions-graph/CONTEXT.md) — records agent session activity as action nodes for observability + tool usage analytics.
- [Sessions Graph](./sessions-graph/CONTEXT.md) — stores explicit Memories, turns completed sessions into Episodes + extracted entities. Named `sessions-graph`: `memory-graph` taken on PyPI. Domain = memory recall.
- [Context Graph Eval](./eval/CONTEXT.md) — measures whether Memory-Tier output is recallable. Not part of the pipeline it measures.

## Relationships

- **Agent Context Graph -> graph components**: emits normalized events. Each component-owned connector decides what to persist. `ActionsGraphConnector` records observability data, `SkillGraphConnector` records skill activity, `SessionsGraphConnector` records session/user lifecycle data.
- **Session Node (shared)**: `(:Session {session_id})` joins component data. Any component may `MERGE` it; none owns it. Adapter sets `session_id`.
- **Actions Graph -> Skills Graph**: both consume tool activity. Actions Graph records it; Skills Graph decides if it proves skill usage.
- **Actions Graph -> Sessions Graph**: reconciliation reads top-level + subagent actions via `get_session_actions()`. Never modifies Actions Graph data ([#275](https://github.com/memgraph/ai-toolkit/issues/275), ticket #281).
- **Actions Graph -> Skills Graph (`Agent` nodes)**: Actions Graph owns each subagent's `Agent` node. Skills Graph may attach `USED_SKILL` to it by `agent_id` — mirrors `HAS_ACTION`/`USED_SKILL` on a top-level Session ([#275](https://github.com/memgraph/ai-toolkit/issues/275)).
- **Sessions Graph -> [unstructured2graph](../unstructured2graph/CONTEXT.md)**: `SESSION_END` only marks Session `reconciliation_status = 'pending'`. Later, out-of-band reconciliation run sends Action + Memory content to `unstructured2graph`: links source nodes to `Chunk` nodes via `HAS_CHUNK`, builds entity graph, makes separate LLM call for Episode. `unstructured2graph` is outside the Context Graph family.
- **Actions/Sessions Graph -> Context Graph Eval**: Eval Batch stages Action fixtures + runs Session Reconciliation in dedicated Memgraph instance. Retrieval + scoring then read resulting Action/Chunk/entity/Episode data, no modification.

## Cross-component graph patterns

`session_id` joins all component data via same `Session` node.

| Query goal | Graph path |
|---|---|
| All actions triggered by a user | `User -> Session -> Action`, plus `User -> Session -> Agent -> Action` for subagents |
| Which skills a user has used | `User -> Session -> Skill`, plus `User -> Session -> Agent -> Skill` for subagents |
| Memories produced during a session | `(:Session)-[:PRODUCED_MEMORY]->(:Memory)` |
| A session's episodic memory (what happened, distilled) | `(:Session)-[:HAS_EPISODE]->(:Episode)` |
| All memories owned by a user | `(:User)-[:HAS_MEMORY]->(:Memory)` |
| Sessions where a specific tool was called | `Session -> Action`, or `Session -> Agent -> Action`, filtered by `Action.tool_name` |
| Entities surfaced in a user's sessions | `User -> Session -> Action|Memory -> Chunk <- Entity`; use `Session -> Agent -> Action` for subagent sources |

Sessions Graph alone owns `User` nodes + `HAD_SESSION`. Other user-scoped queries start there.

## Testing

One rule for Context Graph packages + `unstructured2graph`: test graph persistence against real Memgraph. Mocks OK for pure event translation, model validation, unreachable-DB error branches, LLM boundaries. Never assert on Cypher strings sent to a mock. Full rationale + commands: [AGENTS.md](../AGENTS.md#testing-policy-prefer-real-memgraph-over-mocks).
