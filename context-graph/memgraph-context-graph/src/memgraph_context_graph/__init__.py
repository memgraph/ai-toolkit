"""Memgraph Context Graph: Connect agent runtimes to context-graph components.

Memgraph Context Graph provides a generic adapter layer that bridges any agent
development SDK or runtime hook source to any context-graph component
(actions-graph, skills-graph, etc.).

Architecture:
    Runtime Adapter (Claude Code, Codex, SDK hooks, ...) → Event Protocol → Graph Connector(s)

    Adapters live here.  Connectors live in each graph library.

An app built on the Claude Agent SDK or the OpenAI Agents SDK gets capture,
session context and the memory tools from :mod:`memgraph_context_graph.sdk`::

    from memgraph_context_graph.sdk import ContextGraphMemory
    from memgraph_context_graph.sdk.claude import ClaudeAgentMemory

    options = ClaudeAgentMemory(ContextGraphMemory(user_id="alice")).options(model="claude-sonnet-5-5")
"""

from importlib import metadata

from .capture_policy import CapturePolicy
from .events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorOccurredEvent,
    Event,
    EventType,
    HandoffEvent,
    LLMEndEvent,
    LLMStartEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from .link import AgentLink
from .protocols import GraphConnector, RuntimeAdapter

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "AgentEndEvent",
    "AgentLink",
    "AgentStartEvent",
    "CapturePolicy",
    "ErrorOccurredEvent",
    "Event",
    "EventType",
    "GraphConnector",
    "HandoffEvent",
    "LLMEndEvent",
    "LLMStartEvent",
    "MessageEvent",
    "RuntimeAdapter",
    "SessionEndEvent",
    "SessionStartEvent",
    "ToolEndEvent",
    "ToolStartEvent",
    "TurnEndEvent",
    "__version__",
]
