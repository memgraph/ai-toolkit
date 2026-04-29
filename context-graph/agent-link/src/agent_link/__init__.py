"""Agent Link: Connect agent SDKs to context-graph components.

Agent Link provides a generic adapter layer that bridges any agent SDK
(Claude Agent SDK, OpenAI Agents SDK, etc.) to any context-graph
component (actions-graph, skills-graph, etc.).

Architecture:
    SDK Adapter (Claude, OpenAI, ...) → Event Protocol → Graph Connector(s)

Quick Start:
    from agent_link import AgentLink
    from agent_link.adapters.claude import ClaudeAdapter
    from agent_link.connectors.actions import ActionsConnector

    link = AgentLink()
    link.add_connector(ActionsConnector(actions_graph))
    adapter = ClaudeAdapter(link)

    # Use adapter.hooks() with Claude Agent SDK
    options = ClaudeAgentOptions(hooks=adapter.hooks())
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
