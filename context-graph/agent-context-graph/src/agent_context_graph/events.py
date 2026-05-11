"""Common event protocol for agent runtime actions.

This module defines runtime-agnostic event types that any adapter can emit
and any graph connector can consume. Events capture the essential
information about agent lifecycle moments without coupling to a specific runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Types of events that can occur during an agent run."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    HANDOFF = "handoff"
    MESSAGE = "message"
    ERROR = "error"


@dataclass
class Event:
    """Base event emitted by runtime adapters.

    Every concrete event subclass carries a fixed ``event_type`` and
    runtime-agnostic payload fields. Adapters populate these from their
    respective runtime callbacks or hook payloads, and connectors decide which fields are
    relevant to them.
    """

    event_type: EventType
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    #: The SDK that produced this event (e.g. "claude", "openai").
    source_sdk: str = ""


# ------------------------------------------------------------------
# Session events
# ------------------------------------------------------------------


@dataclass
class SessionStartEvent(Event):
    """Fired when an agent session begins."""

    event_type: EventType = field(default=EventType.SESSION_START, init=False)
    model: str | None = None
    working_directory: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionEndEvent(Event):
    """Fired when an agent session ends."""

    event_type: EventType = field(default=EventType.SESSION_END, init=False)
    status: str = "completed"
    total_cost_usd: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None


# ------------------------------------------------------------------
# Tool events
# ------------------------------------------------------------------


@dataclass
class ToolStartEvent(Event):
    """Fired before a tool/function is invoked."""

    event_type: EventType = field(default=EventType.TOOL_START, init=False)
    tool_name: str = ""
    tool_input: Any = None
    tool_use_id: str | None = None
    agent_name: str | None = None


@dataclass
class ToolEndEvent(Event):
    """Fired after a tool/function returns."""

    event_type: EventType = field(default=EventType.TOOL_END, init=False)
    tool_name: str = ""
    tool_use_id: str | None = None
    result: Any = None
    is_error: bool = False
    error_message: str | None = None
    agent_name: str | None = None


# ------------------------------------------------------------------
# Agent events
# ------------------------------------------------------------------


@dataclass
class AgentStartEvent(Event):
    """Fired when an agent (or subagent) begins execution."""

    event_type: EventType = field(default=EventType.AGENT_START, init=False)
    agent_name: str = ""
    agent_type: str = ""
    parent_agent_name: str | None = None


@dataclass
class AgentEndEvent(Event):
    """Fired when an agent (or subagent) finishes execution."""

    event_type: EventType = field(default=EventType.AGENT_END, init=False)
    agent_name: str = ""
    agent_type: str = ""
    output: Any = None


# ------------------------------------------------------------------
# LLM events
# ------------------------------------------------------------------


@dataclass
class LLMStartEvent(Event):
    """Fired before an LLM call is made."""

    event_type: EventType = field(default=EventType.LLM_START, init=False)
    agent_name: str | None = None
    system_prompt: str | None = None
    input_items: list[Any] = field(default_factory=list)


@dataclass
class LLMEndEvent(Event):
    """Fired after an LLM call returns."""

    event_type: EventType = field(default=EventType.LLM_END, init=False)
    agent_name: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    response: Any = None


# ------------------------------------------------------------------
# Communication events
# ------------------------------------------------------------------


@dataclass
class HandoffEvent(Event):
    """Fired when one agent hands off to another."""

    event_type: EventType = field(default=EventType.HANDOFF, init=False)
    from_agent: str = ""
    to_agent: str = ""


@dataclass
class MessageEvent(Event):
    """Fired when a message is sent (user, assistant, system)."""

    event_type: EventType = field(default=EventType.MESSAGE, init=False)
    role: str = ""
    content: Any = None
    model: str | None = None


# ------------------------------------------------------------------
# Error events
# ------------------------------------------------------------------


@dataclass
class ErrorOccurredEvent(Event):
    """Fired when an error occurs during agent execution."""

    event_type: EventType = field(default=EventType.ERROR, init=False)
    error_type: str = ""
    error_message: str = ""
    error_details: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
