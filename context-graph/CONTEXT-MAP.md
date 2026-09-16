# Context Map

## Language

**Context Graph**:
A family of graph-backed components that turns raw agent activity (**Collection Tier**)
into durable knowledge that agents can reuse (**Memory Tier**). It serves agents
and their runtimes, not mainly people reviewing session history.
_Avoid_: Knowledge graph (names the underlying technology, not the goal)

**Collection Tier**:
Raw, short-lived input captured at hook time or read by a memory builder. Examples
include Actions Graph's Action Nodes and source text used for reconciliation or
procedure mining. Collection-Tier Data is not memory until it is distilled. See
each component's `CONTEXT.md` for its data and planned retention rules
([#267](https://github.com/memgraph/ai-toolkit/issues/267)).
_Avoid_: Memory, raw memory

**Memory Tier**:
Durable knowledge derived from Collection-Tier Data or saved explicitly. It has
three forms:

- **Semantic**: extracted entities and relationships, or an explicit `Memory`
  assertion. `Chunk` nodes preserve source text and provenance; they are not
  themselves memory.
- **Episodic**: an `Episode` summarizing one session.
- **Procedural**: a reusable `Procedure` mined from repeated work. Designed, not
  yet implemented; see Skills Graph's `CONTEXT.md`.

_Avoid_: Collection tier, raw data

## Contexts

- [Agent Context Graph](./agent-context-graph/CONTEXT.md) — translates agent SDK and runtime hook activity into a shared event stream and routes it to graph connectors.
- [Skills Graph](./skills-graph/CONTEXT.md) — stores reusable agent skills in Memgraph and records when sessions use those skills.
- [Actions Graph](./actions-graph/CONTEXT.md) — records agent session activity as action nodes for observability and tool usage analytics.
- [Sessions Graph](./sessions-graph/CONTEXT.md) — stores explicit Memories and
  turns completed sessions into Episodes and extracted entities. The package is
  named `sessions-graph` because `memory-graph` was unavailable on PyPI; its
  domain is memory recall.
- [Context Graph Eval](./eval/CONTEXT.md) — measures whether Memory-Tier output
  can be recalled usefully. It is not part of the pipeline it measures.

## Relationships

- **Agent Context Graph -> graph components**: emits normalized events. Each
  component-owned connector decides what to persist. `ActionsGraphConnector`
  records observability data, `SkillGraphConnector` records skill activity, and
  `SessionsGraphConnector` records session/user lifecycle data.
- **Session Node (shared)**: `(:Session {session_id})` joins component data. Any
  component may `MERGE` it; no component owns it. The adapter sets `session_id`.
- **Actions Graph -> Skills Graph**: both consume tool activity. Actions Graph
  records it; Skills Graph decides whether it proves skill usage.
- **Actions Graph -> Sessions Graph**: reconciliation reads top-level and
  subagent actions through `get_session_actions()`. It does not modify Actions
  Graph data ([#275](https://github.com/memgraph/ai-toolkit/issues/275), ticket
  #281).
- **Actions Graph -> Skills Graph (`Agent` nodes)**: Actions Graph owns each
  subagent's `Agent` node. Skills Graph may attach `USED_SKILL` to that node by
  `agent_id`. This mirrors `HAS_ACTION` and `USED_SKILL` on a top-level Session
  ([#275](https://github.com/memgraph/ai-toolkit/issues/275)).
- **Sessions Graph -> [unstructured2graph](../unstructured2graph/CONTEXT.md)**:
  `SESSION_END` only marks the Session
  `reconciliation_status = 'pending'`. Later, an out-of-band reconciliation run
  sends Action and Memory content to `unstructured2graph`. It links source nodes
  to `Chunk` nodes with `HAS_CHUNK`, builds the entity graph, and makes a separate
  LLM call for the Episode. `unstructured2graph` is outside the Context Graph
  family.
- **Actions/Sessions Graph -> Context Graph Eval**: An Eval Batch stages Action
  fixtures and runs Session Reconciliation in a dedicated Memgraph instance.
  Retrieval and scoring then read the resulting Action, Chunk, entity, and
  Episode data without modifying it.

## Cross-component graph patterns

`session_id` joins all component data through the same `Session` node.

| Query goal | Graph path |
|---|---|
| All actions triggered by a user | `User -> Session -> Action`, plus `User -> Session -> Agent -> Action` for subagents |
| Which skills a user has used | `User -> Session -> Skill`, plus `User -> Session -> Agent -> Skill` for subagents |
| Memories produced during a session | `(:Session)-[:PRODUCED_MEMORY]->(:Memory)` |
| A session's episodic memory (what happened, distilled) | `(:Session)-[:HAS_EPISODE]->(:Episode)` |
| All memories owned by a user | `(:User)-[:HAS_MEMORY]->(:Memory)` |
| Sessions where a specific tool was called | `Session -> Action`, or `Session -> Agent -> Action`, filtered by `Action.tool_name` |
| Entities surfaced in a user's sessions | `User -> Session -> Action|Memory -> Chunk <- Entity`; use `Session -> Agent -> Action` for subagent sources |

Sessions Graph alone owns `User` nodes and `HAD_SESSION`. Other user-scoped
queries start there.

## Testing

Context Graph packages and `unstructured2graph` share one rule: test graph
persistence against real Memgraph. Keep mocks for pure event translation, model
validation, unreachable DB error branches, and LLM boundaries. Do not test
Cypher by asserting on strings sent to a mock. Full rationale and commands:
[AGENTS.md](../AGENTS.md#testing-policy-prefer-real-memgraph-over-mocks).
