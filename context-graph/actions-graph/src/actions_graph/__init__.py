"""Actions Graph: Store and track LLM actions, tool calls, and sessions in Memgraph.

This package provides a graph-based storage system for tracking all LLM interactions,
including tool calls, messages, structured outputs, and session management.

Quick Start:
    from actions_graph import ActionsGraph, Session, ToolCall

    # Initialize the graph
    graph = ActionsGraph()
    graph.setup()

    # Create a session
    session = Session(session_id="my-session-123")
    graph.create_session(session)

    # Record tool calls
    graph.record_tool_call(
        session_id="my-session-123",
        tool_name="Read",
        tool_input={"file_path": "/path/to/file"},
    )

Integration with Claude Agent SDK:
    from actions_graph import ActionsGraph
    from actions_graph.hooks import create_tracking_hooks
    from claude_agent_sdk import query, ClaudeAgentOptions

    graph = ActionsGraph()
    graph.setup()

    hooks = create_tracking_hooks(graph, session_id="my-session-123")

    async for message in query(
        prompt="Analyze this codebase",
        options=ClaudeAgentOptions(
            hooks=hooks,
            allowed_tools=["Read", "Glob", "Grep"],
        ),
    ):
        print(message)
"""

from .core import ActionsGraph
from .models import (
    Action,
    ActionStatus,
    ActionType,
    ActionValidationError,
    ErrorEvent,
    Message,
    MessageRole,
    PermissionRequest,
    RateLimitEvent,
    Session,
    StructuredOutput,
    SubagentEvent,
    ToolCall,
    ToolResult,
)

__all__ = [
    "Action",
    "ActionStatus",
    "ActionType",
    "ActionValidationError",
    "ActionsGraph",
    "ErrorEvent",
    "Message",
    "MessageRole",
    "PermissionRequest",
    "RateLimitEvent",
    "Session",
    "StructuredOutput",
    "SubagentEvent",
    "ToolCall",
    "ToolResult",
]

__version__ = "0.1.0"
