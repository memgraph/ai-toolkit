# Actions Graph

Records agent-session activity for observability. Tool usage = main analytics surface.

## Language

**Actions Graph**:
Graph component recording agent session activity as action nodes, for observability + analytics.
_Avoid_: Action tracker, Claude hook storage, telemetry sink

**Action Node**:
Persisted graph rep of one meaningful event in an agent session.
_Avoid_: Event, log line, trace span

**Tool Call Action**:
**Action Node** = start of tool invocation.
_Avoid_: Tool definition, tool result

**Tool Result Action**:
**Action Node** = completion of tool invocation.
_Avoid_: Tool call

**Tool Usage Analytics**:
Aggregated insight over tool call actions, tool result actions, errors, sequence relationships.
_Avoid_: Skill usage, raw hook logging

**Collection-Tier Data**:
Raw, short-lived observability data captured at hook time. Other components may distill it into an Episode or (future) Procedure. Actions Graph itself doesn't produce memory.
_Avoid_: Memory, durable record (Action Node is neither — see [context-graph#261](https://github.com/memgraph/ai-toolkit/issues/261))

**Confirmation-Gated Deletion** (designed, not implemented — [context-graph#267](https://github.com/memgraph/ai-toolkit/issues/267)):
Planned retention rule for Collection-Tier Data. Action deletable only after every memory builder confirms done. Two consumers: Session Reconciliation + future procedure miner. Enforcement = allowlist of collection-tier types (`Action`, `Agent`), not denylist ([#267](https://github.com/memgraph/ai-toolkit/issues/267), [#275](https://github.com/memgraph/ai-toolkit/issues/275)).
_Avoid_: TTL, expiry (implies fixed window; real gate = per-consumer confirmation)

**Agent Node** (implemented — map [context-graph#275](https://github.com/memgraph/ai-toolkit/issues/275), tickets #277/#278):
`(:Agent {...})` node = one subagent's full lifecycle. Start/stop events update same node via `start_agent`/`end_agent`; no Action nodes created. Agent nodes are Collection-Tier Data.
_Avoid_: `SubagentEvent`/`ActionType.SUBAGENT_START`/`SUBAGENT_STOP` (deleted outright, not deprecated, when Agent replaced them)

## Relationships

- **Actions Graph** belongs to broader **Context Graph** family.
- **Actions Graph** consumes relevant **Event Protocol** events via `ActionsGraphConnector`.
- **Tool Call Action** created from tool-start event.
- **Tool Result Action** created from tool-end event.
- **Tool Usage Analytics** derived from action nodes + relationships.
- **Actions Graph** observes tool usage; doesn't decide if a tool call = **Skill Usage**.
- Sessions Graph reads Action Nodes to produce an Episode. `get_session_actions()` covers both `Session -> Action` and `Session -> Agent -> Action`. Future procedure miner reads same `FOLLOWED_BY` sequences.
- Every Agent has `(:Session)-[:HAS_AGENT]->(:Agent)`, even when spawning Action can't be inferred. Agent nodes never orphaned.
- `(:Action)-[:SPAWNED]->(:Agent)` links spawning tool call to its subagent when temporal inference finds one unambiguous match: first considers open spawn calls, then matches `agent_type` to `tool_input.subagent_type`. Ambiguity -> link unset ([#277](https://github.com/memgraph/ai-toolkit/issues/277)). Claude Code reports tool as `Agent`, some surfaces call it `Task` -> `agent_spawning_tool_names` defaults to both. Codex support + concurrent-agent inference unverified ([#286](https://github.com/memgraph/ai-toolkit/issues/286), [#287](https://github.com/memgraph/ai-toolkit/issues/287)). Adapters with subagent support use this common inference rule — even where a better signal exists: Claude Agent SDK's message-stream API exposes real `parent_tool_use_id` that'd link a spawn exactly, but `claude.py` deliberately stays on the inferred rule so all four runtime adapters behave the same ([#294](https://github.com/memgraph/ai-toolkit/issues/294)). Revisit only if that asymmetry becomes worth taking on.
- `(:Agent)-[:HAS_ACTION]->(:Action)` = subagent's actions. Top-level actions use `(:Session)-[:HAS_ACTION]->(:Action)`. Each container own `FOLLOWED_BY` chain.
- `(:Action)-[:PARENT_OF]->(:Action)` links ToolCall parent to its ToolResult/error child. Subagent nesting uses `Agent`/`HAS_ACTION`/`SPAWNED` instead.
- `Session.parent_session_id` + `FORKED_FROM` removed: no runtime payload ever supplied a parent session id (#279).

## Example dialogue

> **Dev:** "Is Actions Graph only for Claude Agent SDK hooks?"
> **Domain expert:** "No. Runtime adapters emit Event Protocol events. Actions Graph consumes those via its graph connector."

> **Dev:** "Does Actions Graph replace Skills Graph?"
> **Domain expert:** "No. Actions Graph records tool activity for observability. Skills Graph interprets some activity as skill usage."

## Flagged ambiguities

- "action" = any agent activity or only tool calls? Resolved: **Action Node** = general persisted unit; tool calls = primary analytics surface.
- "tool usage" can sound like skill usage. Resolved: **Tool Usage Analytics** = observability over tools; **Skill Usage** owned by Skills Graph.
- Does Actions Graph produce memory? Resolved: no. Stores Collection-Tier Data for observability. Sessions Graph + future procedure miner do distillation.
