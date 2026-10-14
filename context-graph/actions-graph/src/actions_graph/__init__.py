"""Actions Graph: record agent-session tool calls, messages, and subagents in Memgraph.

Usually fed through the Memgraph Context Graph hub:
    from actions_graph import ActionsGraph
    from actions_graph.connector import ActionsGraphConnector
    from memgraph_context_graph import AgentLink

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
    __version__ = ""
del metadata

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
