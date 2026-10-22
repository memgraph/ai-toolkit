# Actions Graph

Store and track LLM actions, tool calls, and sessions in Memgraph.

> Part of the [Context Graph](../README.md) family — usually wired into `memgraph-context-graph` to capture Claude Code / Codex sessions automatically. This README covers using it directly.

Records each agent session's tool calls, tool results, messages, errors, and subagents as `(:Action)`/`(:Agent)` nodes for observability and tool-usage analytics. Collection-tier data, not memory.

## Installation

```bash
pip install actions-graph
```

For Memgraph Context Graph integration:
```bash
pip install "actions-graph[memgraph-context-graph]"
```

Needs a running Memgraph (default `bolt://localhost:7687`); `ActionsGraph()` reads `MEMGRAPH_URL`, `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD`, `MEMGRAPH_DATABASE` (or takes them as kwargs):

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
```

## Quick Start

### Basic Usage

```python
from actions_graph import ActionsGraph, Session

# Initialize the graph
graph = ActionsGraph()  # reads MEMGRAPH_URL / MEMGRAPH_USER / MEMGRAPH_PASSWORD
graph.setup()  # Create indexes and constraints

# Create a session
session = Session(
    session_id="session-123",
    model="claude-sonnet-4-20250514",
    working_directory="/path/to/project",
)
graph.create_session(session)

# Record a tool call
tool_call = graph.record_tool_call(
    session_id="session-123",
    tool_name="Read",
    tool_input={"file_path": "/path/to/file.py"},
    tool_use_id="tool-use-001",
)

# Record the result
tool_result = graph.record_tool_result(
    session_id="session-123",
    tool_use_id="tool-use-001",
    tool_name="Read",
    content="def hello():\n    print('Hello, World!')",
)

# Get session summary
summary = graph.get_session_summary("session-123")
print(f"Actions: {summary['action_count']}, Tools: {summary['tool_call_count']}")
```

### Memgraph Context Graph Integration

```python
from actions_graph import ActionsGraph
from actions_graph.connector import ActionsGraphConnector
from memgraph_context_graph import AgentLink
from memgraph_context_graph.sdk import ContextGraphMemory
from memgraph_context_graph.sdk.claude import ClaudeAgentMemory

graph = ActionsGraph()
graph.setup()

link = AgentLink()
link.add_connector(ActionsGraphConnector(graph))

# A Claude Agent SDK app; OpenAIAgentsMemory(memory) wires an OpenAI Agents SDK app the same way.
options = ClaudeAgentMemory(ContextGraphMemory(link=link)).options()
```

Actions Graph should consume runtime activity through Memgraph Context Graph when possible. Runtime adapters normalize SDK callbacks and command hooks into the shared Event Protocol; `ActionsGraphConnector` decides which events become session and action nodes.

## Graph Schema

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

## API Reference

### ActionsGraph

Main class for interacting with the graph.

```python
graph = ActionsGraph()

# Setup
graph.setup()  # Create indexes
graph.drop()  # Remove indexes
graph.clear()  # Clear all data

# Sessions
graph.create_session(session)
graph.get_session(session_id)
graph.end_session(session_id, status=ActionStatus.COMPLETED)
graph.list_sessions(limit=100, status=None)  # keyword-only args

# Actions
graph.record_action(action)
graph.record_tool_call(session_id, tool_name, tool_input, ...)
graph.record_tool_result(session_id, tool_use_id, tool_name, content, ...)
graph.record_message(session_id, role, content, ...)
graph.get_action(action_id)
graph.get_session_actions(session_id, action_type=None, limit=1000)

# Subagents
graph.start_agent(agent)
graph.end_agent(agent_id, status=ActionStatus.COMPLETED)
graph.get_agent(agent_id)
graph.get_session_agents(session_id)

# Analytics
graph.get_tool_usage_stats(session_id=None)
graph.get_action_sequence(session_id, include_content=False)
graph.get_session_summary(session_id)
```

### Action Types

| Type | Model Class | Description |
|------|-------------|-------------|
| `tool_call` | `ToolCall` | Tool/function invocation |
| `tool_result` | `ToolResult` | Tool execution result |
| `user_message` | `Message` | User input |
| `assistant_message` | `Message` | LLM response |
| `system_message` | `Message` | System messages |
| `structured_output` | `StructuredOutput` | Validated JSON output |
| `error` | `ErrorEvent` | Error occurred |
| `permission_request` | `PermissionRequest` | Permission requested |
| `rate_limit` | `RateLimitEvent` | Rate limit event |

### Memgraph Context Graph Connector

```python
from actions_graph.connector import ActionsGraphConnector

connector = ActionsGraphConnector(graph)
link.add_connector(connector)
```

The connector records:

- `SessionStartEvent` and `SessionEndEvent` as session lifecycle data
- `ToolStartEvent`/`ToolEndEvent` as `ToolCall`/`ToolResult` actions
- `MessageEvent` and `ErrorOccurredEvent` as `Message`/`ErrorEvent` actions
- `AgentStartEvent`/`AgentEndEvent` as one `Agent` node

## Example Queries

### Find sessions with errors

```python
sessions = graph.list_sessions(status=ActionStatus.FAILED)
```

### Get all tool calls in a session

```python
from actions_graph import ActionType

tool_calls = graph.get_session_actions(
    session_id,
    action_type=ActionType.TOOL_CALL,
)
```

### Custom Cypher queries

```python
rows = graph.db.query(
    """
    MATCH (s:Session {session_id: $session_id})-[:HAS_ACTION]->(a:ToolCall)
    RETURN a.tool_name AS tool, count(*) AS count
    ORDER BY count DESC
""",
    params={"session_id": "my-session"},
)
```

## License

MIT
