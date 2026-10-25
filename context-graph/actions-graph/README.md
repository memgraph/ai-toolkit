# Actions Graph

Record agent tool calls, results, messages, errors, and subagents in Memgraph. Inspect a session's timeline or query tool usage across sessions.

Part of [Context Graph](../README.md). The [installer](../README.md#getting-started) enables automatic capture for coding agents; this README shows direct Python use.

## Installation

```bash
pip install actions-graph
```

For the runtime connector, install `actions-graph[memgraph-context-graph]`. A running Memgraph is required; the default connection is `bolt://localhost:7687`.

## Quick start

```python
from actions_graph import ActionsGraph, Session

graph = ActionsGraph()  # reads MEMGRAPH_URL / MEMGRAPH_USER / MEMGRAPH_PASSWORD
graph.setup()  # Create indexes and constraints

session = Session(
    session_id="session-123",
    working_directory="/path/to/project",
)
graph.create_session(session)

tool_call = graph.record_tool_call(
    session_id="session-123",
    tool_name="Read",
    tool_input={"file_path": "/path/to/file.py"},
    tool_use_id="tool-use-001",
)

tool_result = graph.record_tool_result(
    session_id="session-123",
    tool_use_id="tool-use-001",
    tool_name="Read",
    content="def hello():\n    print('Hello, World!')",
)

summary = graph.get_session_summary("session-123")
print(f"Actions: {summary['action_count']}, Tools: {summary['tool_call_count']}")
```

`ActionsGraph()` reads `MEMGRAPH_URL`, `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD`, and `MEMGRAPH_DATABASE`, or accepts connection keyword arguments. Hook integrations use the Context Graph config file instead.

## What is stored

### Nodes

- **Session**: `session_id`, `started_at`, `ended_at`, `status`, `model`, `total_cost_usd`, token totals, `working_directory`, `git_branch`, `metadata`.
- **Agent**: one subagent's lifecycle (`agent_id`, `agent_type`, `started_at`, `ended_at`, `status`, `description`, `last_assistant_message`).
- **Action**: one event, with a type label: `ToolCall`, `ToolResult`, `Message` (plus `UserMessage`/`AssistantMessage`/`SystemMessage`), `StructuredOutput`, `PermissionRequest`, `ErrorEvent`, `RateLimitEvent`.
  Properties: `action_id`, `action_type`, `timestamp`, `status`, `duration_ms`, `parent_action_id`, `tool_name`, `tool_use_id` (the harness's id for the call), `is_error`, `is_mcp`, `mcp_server` (MCP tool calls), `text` (user and assistant messages only), `properties` (type-specific JSON), `metadata`.
  The session link is the `HAS_ACTION` edge; there is no `session_id` on `(:Action)`.

### Relationships

```
(:Session)-[:HAS_ACTION]->(:Action)   # top-level actions
(:Session)-[:HAS_AGENT]->(:Agent)     # every agent, even when its spawner is unknown
(:Agent)-[:HAS_ACTION]->(:Action)     # actions inside a subagent
(:Action)-[:SPAWNED]->(:Agent)        # spawning tool call, when unambiguous
(:Action)-[:FOLLOWED_BY]->(:Action)   # temporal sequence, one chain per Session/Agent
(:Action)-[:PARENT_OF]->(:Action)     # ToolCall -> its ToolResult
```

## Use with an agent runtime

Runtime adapters translate activity into shared events. `ActionsGraphConnector` records the events as session, action, and subagent nodes.

```python
from actions_graph import ActionsGraph
from actions_graph.connector import ActionsGraphConnector
from memgraph_context_graph import AgentLink

graph = ActionsGraph()
graph.setup()

link = AgentLink()
link.add_connector(ActionsGraphConnector(graph))
```

Actions Graph should consume runtime activity through Memgraph Context Graph when possible. Runtime adapters normalize SDK callbacks and command hooks into the shared Event Protocol; `ActionsGraphConnector` decides which events become session and action nodes.

For a complete SDK application, see the [SDK quick starts](../memgraph-context-graph/docs/sdk-memory.md).

## Query and API reference

The [Python API guide](docs/python-api.md) covers recording events, session analytics, action types, and example queries.

## Development

From the repository root:

```bash
./scripts/dev-memgraph.sh test actions-graph
```

## License

MIT
