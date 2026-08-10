"""Agent Context Graph: Connect agent runtimes to context-graph components.

Agent Context Graph provides a generic adapter layer that bridges any agent
development SDK or runtime hook source to any context-graph component
(actions-graph, skills-graph, etc.).

Architecture:
    Runtime Adapter (Claude, OpenAI, Codex, ...) → Event Protocol → Graph Connector(s)

    Adapters live here.  Connectors live in each graph library.

Quick Start::

    from agent_context_graph import AgentLink
    from agent_context_graph.adapters.claude import ClaudeAdapter
    from skills_graph.connector import SkillGraphConnector

    link = AgentLink()
    link.add_connector(SkillGraphConnector(skill_graph))
    adapter = ClaudeAdapter(link, session_id="s-1")
    hooks = adapter.get_runtime_hooks()
"""

from importlib import metadata

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
)
from .link import AgentLink
from .protocols import GraphConnector, RuntimeAdapter

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "AgentEndEvent",
    "AgentLink",
    "AgentStartEvent",
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
    "__version__",
]
