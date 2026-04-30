"""Agent Context Graph: Connect agent SDKs to context-graph components.

Agent Context Graph provides a generic adapter layer that bridges any agent SDK
(Claude Agent SDK, OpenAI Agents SDK, etc.) to any context-graph
component (actions-graph, skills-graph, etc.).

Architecture:
    SDK Adapter (Claude, OpenAI, ...) → Event Protocol → Graph Connector(s)

    Adapters live here.  Connectors live in each graph library.

Quick Start::

    from agent_context_graph import AgentLink
    from agent_context_graph.adapters.claude import ClaudeAdapter
    from skills_graph.connector import SkillGraphConnector

    link = AgentLink()
    link.add_connector(SkillGraphConnector(skill_graph))
    adapter = ClaudeAdapter(link, session_id="s-1")
    hooks = adapter.get_sdk_hooks()
"""

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
from .protocols import GraphConnector, SDKAdapter

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
    "SDKAdapter",
    "SessionEndEvent",
    "SessionStartEvent",
    "ToolEndEvent",
    "ToolStartEvent",
]

__version__ = "0.1.0"
