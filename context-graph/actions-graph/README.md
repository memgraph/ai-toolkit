# Actions Graph

Store and track LLM actions, tool calls, and sessions in Memgraph.

Actions Graph provides a graph-based storage system for tracking all LLM interactions, including:
- **Tool Calls**: Function/tool invocations by the LLM
- **Tool Results**: Outputs from tool executions
- **Messages**: User, assistant, and system messages
- **Structured Outputs**: Validated JSON outputs from the LLM
- **Subagent Events**: Subagent lifecycle tracking
- **Sessions**: Conversation session management

## Features

- 📊 **Graph-based Storage**: Store actions as nodes with relationships in Memgraph
- 🔗 **Temporal Sequences**: Track action order with `FOLLOWED_BY` relationships
- 🌳 **Nested Actions**: Support for parent-child action relationships (e.g., subagents)
- 🏷️ **Session Management**: Create, track, and query sessions with tags
- 📈 **Analytics**: Built-in queries for tool usage stats and session summaries
- 🤖 **Claude Agent SDK Integration**: Ready-to-use hooks for automatic tracking

## Installation

```bash
pip install actions-graph
```

For Claude Agent SDK integration:
```bash
pip install actions-graph[claude-agent]
```

## Quick Start

### Basic Usage

```python
from actions_graph import ActionsGraph, Session, ToolCall

# Initialize the graph
graph = ActionsGraph()
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

### Claude Agent SDK Integration

```python
import asyncio
from actions_graph import ActionsGraph
from actions_graph.hooks import create_tracking_hooks
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    # Initialize graph
    graph = ActionsGraph()
    graph.setup()

    # Create tracking hooks
    hooks = create_tracking_hooks(
        graph,
        session_id="my-session-123",
        session_kwargs={
            "model": "claude-sonnet-4-20250514",
            "working_directory": "/path/to/project",
            "tags": ["code-review", "python"],
        },
    )

    # Run with automatic tracking
    async for message in query(
        prompt="Review the code in src/main.py for potential bugs",
        options=ClaudeAgentOptions(
            hooks=hooks,
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="acceptEdits",
        ),
    ):
        if hasattr(message, "result"):
            print(message.result)

    # Query the recorded actions
    actions = graph.get_session_actions("my-session-123")
    print(f"Recorded {len(actions)} actions")

    # Get tool usage stats
    stats = graph.get_tool_usage_stats("my-session-123")
    for stat in stats:
        print(f"{stat['tool_name']}: {stat['call_count']} calls")

asyncio.run(main())
```

## Graph Schema

### Nodes

- **Session**: LLM conversation sessions
  - Properties: `session_id`, `started_at`, `ended_at`, `status`, `model`, `total_cost_usd`, etc.

- **Action**: Individual actions with type-specific labels
  - Labels: `ToolCall`, `ToolResult`, `Message`, `StructuredOutput`, `SubagentEvent`, etc.
  - Properties: `action_id`, `session_id`, `action_type`, `timestamp`, `status`, etc.

- **Tool**: Tool definitions
  - Properties: `name`, `is_mcp`, `mcp_server`

- **Tag**: Session/action tags
  - Properties: `name`

### Relationships

```
(:Session)-[:HAS_ACTION]->(:Action)
(:Action)-[:FOLLOWED_BY]->(:Action)
(:Action)-[:PARENT_OF]->(:Action)
(:Session)-[:FORKED_FROM]->(:Session)
(:Action)-[:USED_TOOL]->(:Tool)
(:Session)-[:HAS_TAG]->(:Tag)
```

## API Reference

### ActionsGraph

Main class for interacting with the graph.

```python
graph = ActionsGraph()

# Setup
graph.setup()      # Create indexes
graph.drop()       # Remove indexes
graph.clear()      # Clear all data

# Sessions
graph.create_session(session)
graph.get_session(session_id)
graph.end_session(session_id, status=ActionStatus.COMPLETED)
graph.list_sessions(limit=100, status=None, tag=None)

# Actions
graph.record_action(action)
graph.record_tool_call(session_id, tool_name, tool_input, ...)
graph.record_tool_result(session_id, tool_use_id, tool_name, content, ...)
graph.record_message(session_id, role, content, ...)
graph.get_action(action_id)
graph.get_session_actions(session_id, action_type=None, limit=1000)

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
| `subagent_start` | `SubagentEvent` | Subagent started |
| `subagent_stop` | `SubagentEvent` | Subagent completed |
| `error` | `ErrorEvent` | Error occurred |
| `permission_request` | `PermissionRequest` | Permission requested |
| `rate_limit` | `RateLimitEvent` | Rate limit event |

### Hooks

For Claude Agent SDK integration:

```python
from actions_graph.hooks import create_tracking_hooks, ActionTracker

# Simple usage
hooks = create_tracking_hooks(graph, session_id)

# Advanced usage with custom tracker
tracker = ActionTracker(
    graph,
    session_id,
    track_tool_calls=True,
    track_tool_results=True,
    track_messages=True,
    track_subagents=True,
    track_permissions=True,
    track_errors=True,
)
```

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
rows = graph._db.query("""
    MATCH (s:Session {session_id: $session_id})-[:HAS_ACTION]->(a:ToolCall)
    RETURN a.tool_name AS tool, count(*) AS count
    ORDER BY count DESC
""", params={"session_id": "my-session"})
```

## License

MIT
