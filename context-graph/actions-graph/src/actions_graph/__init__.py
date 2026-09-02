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

Integration with Agent Context Graph:
    from actions_graph import ActionsGraph
    from actions_graph.connector import ActionsGraphConnector
    from agent_context_graph import AgentLink

    graph = ActionsGraph()
    graph.setup()

    link = AgentLink()
    link.add_connector(ActionsGraphConnector(graph))
"""

from importlib import metadata

from .core import ActionsGraph
from .models import (
    Action,
    ActionStatus,
    ActionType,
    ActionValidationError,
    Agent,
    ErrorEvent,
    Message,
    MessageRole,
    PermissionRequest,
    RateLimitEvent,
    Session,
    StructuredOutput,
    ToolCall,
    ToolResult,
)

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "Action",
    "ActionStatus",
    "ActionType",
    "ActionValidationError",
    "ActionsGraph",
    "Agent",
    "ErrorEvent",
    "Message",
    "MessageRole",
    "PermissionRequest",
    "RateLimitEvent",
    "Session",
    "StructuredOutput",
    "ToolCall",
    "ToolResult",
    "__version__",
]
