# Actions Graph

Actions Graph records agent-session activity for observability. Tool usage is its
main analytics surface.

## Language

**Actions Graph**:
The graph component that records agent session activity as action nodes for observability and analytics.
_Avoid_: Action tracker, Claude hook storage, telemetry sink

**Action Node**:
The persisted graph representation of one meaningful event in an agent session.
_Avoid_: Event, log line, trace span

**Tool Call Action**:
An **Action Node** representing the start of a tool invocation.
_Avoid_: Tool definition, tool result

**Tool Result Action**:
An **Action Node** representing the completion of a tool invocation.
_Avoid_: Tool call

**Tool Usage Analytics**:
Aggregated insight over tool call actions, tool result actions, errors, and sequence relationships.
_Avoid_: Skill usage, raw hook logging

**Collection-Tier Data**:
Raw, short-lived observability data captured at hook time. Other components may
distill it into an Episode or, in future, a Procedure. Actions Graph itself does
not produce memory.
_Avoid_: Memory, durable record (an Action Node is neither — see [context-graph#261](https://github.com/memgraph/ai-toolkit/issues/261))

**Confirmation-Gated Deletion** (designed, not yet implemented — see [context-graph#267](https://github.com/memgraph/ai-toolkit/issues/267)):
Planned retention rule for Collection-Tier Data. An Action becomes deletable only
after every memory builder confirms it is done. The design has two consumers:
Session Reconciliation and a future procedure miner. Enforcement
will use an allowlist of collection-tier types (`Action`, `Agent`), not a
denylist ([#267](https://github.com/memgraph/ai-toolkit/issues/267),
[#275](https://github.com/memgraph/ai-toolkit/issues/275)).
_Avoid_: TTL, expiry (implies a fixed window; the real gate is per-consumer confirmation, see the ticket)

**Agent Node** (implemented — map [context-graph#275](https://github.com/memgraph/ai-toolkit/issues/275), tickets #277/#278):
A `(:Agent {...})` node representing one subagent's full lifecycle. Start and stop
events update the same node through `start_agent` and `end_agent`; they do not
create Action nodes. Agent nodes are Collection-Tier Data.
_Avoid_: `SubagentEvent`/`ActionType.SUBAGENT_START`/`SUBAGENT_STOP` (deleted outright, not deprecated, when Agent replaced them)

## Relationships

- **Actions Graph** belongs to the broader **Context Graph** family.
- **Actions Graph** consumes relevant **Event Protocol** events through `ActionsGraphConnector`.
- A **Tool Call Action** is created from a tool-start event.
- A **Tool Result Action** is created from a tool-end event.
- **Tool Usage Analytics** is derived from action nodes and their relationships.
- **Actions Graph** observes tool usage; it does not decide whether a tool call represents **Skill Usage**.
- Sessions Graph reads Action Nodes to produce an Episode. `get_session_actions()`
  includes both `Session -> Action` and `Session -> Agent -> Action`. A future
  procedure miner will read the same `FOLLOWED_BY` sequences.
- Every Agent has `(:Session)-[:HAS_AGENT]->(:Agent)`, even when its spawning
  Action cannot be inferred. Agent nodes are never orphaned.
- `(:Action)-[:SPAWNED]->(:Agent)` links a spawning tool call to its subagent when
  temporal inference finds one unambiguous match. It first considers open spawn
  calls, then matches `agent_type` to `tool_input.subagent_type`. Ambiguity leaves
  the link unset ([#277](https://github.com/memgraph/ai-toolkit/issues/277)).
  Claude Code reports the tool as `Agent`, while some surfaces call it `Task`, so
  `agent_spawning_tool_names` defaults to both. Codex support and concurrent-agent
  inference remain unverified ([#286](https://github.com/memgraph/ai-toolkit/issues/286),
  [#287](https://github.com/memgraph/ai-toolkit/issues/287)). Adapters with
  subagent support use this common inference rule
  ([#294](https://github.com/memgraph/ai-toolkit/issues/294)).
- `(:Agent)-[:HAS_ACTION]->(:Action)` holds a subagent's actions. Top-level
  actions use `(:Session)-[:HAS_ACTION]->(:Action)`. Each container has its own
  `FOLLOWED_BY` chain.
- `(:Action)-[:PARENT_OF]->(:Action)` links a ToolCall parent to its ToolResult or
  error child. Subagent nesting instead uses `Agent`, `HAS_ACTION`, and `SPAWNED`.
- `Session.parent_session_id` and `FORKED_FROM` were removed because no runtime
  payload supplied a parent session id (#279).

## Example dialogue

> **Dev:** "Is Actions Graph only for Claude Agent SDK hooks?"
> **Domain expert:** "No. Runtime adapters emit Event Protocol events. Actions Graph consumes those events through its graph connector."

> **Dev:** "Does Actions Graph replace Skills Graph?"
> **Domain expert:** "No. Actions Graph records tool activity for observability. Skills Graph interprets some activity as skill usage."

## Flagged ambiguities

- "action" can mean any agent activity or only tool calls. Resolved: **Action Node** is the general persisted unit; tool calls are the primary analytics surface.
- "tool usage" can sound like skill usage. Resolved: **Tool Usage Analytics** is observability over tools; **Skill Usage** remains owned by Skills Graph.
- Whether Actions Graph produces memory. Resolved: no. It stores Collection-Tier
  Data for observability. Sessions Graph and a future procedure miner perform
  distillation.
