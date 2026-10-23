# Actions Graph Python API

[Back to Actions Graph](../README.md)

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
from actions_graph import ActionStatus

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
