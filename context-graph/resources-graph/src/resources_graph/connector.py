"""Agent-link connector: hook events become address-only Touches.

Makes no network call — a hook must never wait on GitHub. Three events matter:

- a tool start that fetches GitHub (WebFetch, ``gh``, ``curl``, a GitHub MCP
  tool) becomes a FETCHED Touch;
- a user prompt that mentions GitHub links becomes PROMPTED Touches;
- a finished call of this component's own ``resource`` tool becomes a Cache
  Read, with the outcome the tool printed. The tool runs over MCP and can't
  see the session, so its Touch is written here, where the session is known.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agent_context_graph.events import Event, EventType, MessageEvent, ToolEndEvent, ToolStartEvent
from agent_context_graph.protocols import GraphConnector

from .address import Address, addresses_from_text, addresses_from_tool
from .models import FETCHED, PROMPTED

if TYPE_CHECKING:
    from .core import ResourcesGraph

_SUPPORTED_EVENTS = {EventType.TOOL_START, EventType.TOOL_END, EventType.MESSAGE}
#: The first line of every ``resource`` tool answer; see :mod:`resources_graph.tool`.
OUTCOME_LINE = re.compile(r"\[resource (hit|subsumed|miss)\] (\S+)(?: from (\S+))?")


def is_resource_tool(tool_name: str) -> bool:
    """Whether ``tool_name`` is this component's ``resource`` tool, however the harness prefixes it.

    Harnesses namespace MCP tools (``mcp__plugin_context-graph_context-graph__resource``).
    """
    name = tool_name.strip().lower()
    return name == "resource" or (name.endswith("__resource") and "context" in name)


class ResourcesGraphConnector(GraphConnector):
    """Records Touches and Cache Reads in a :class:`ResourcesGraph`."""

    def __init__(self, graph: ResourcesGraph) -> None:
        self._graph = graph

    def supports(self, event: Event) -> bool:
        return event.event_type in _SUPPORTED_EVENTS

    def on_event(self, event: Event) -> None:
        if isinstance(event, ToolStartEvent):
            self._on_tool_start(event)
        elif isinstance(event, ToolEndEvent):
            self._on_tool_end(event)
        elif isinstance(event, MessageEvent) and event.role == "user":
            self._on_prompt(event)

    def context_before_tool(self, event: ToolStartEvent) -> str | None:
        """The Nudge: one line when the tool is about to fetch something memory already holds.

        Facts, not a verdict — when it was fetched and when GitHub last changed
        it — so the model decides; the fetch proceeds either way. No GitHub call.
        """
        if is_resource_tool(event.tool_name):
            return None
        lines = []
        for address in addresses_from_tool(event.tool_name, event.tool_input):
            facts = self._graph.in_memory(address)
            if facts is None:
                continue
            ages = f"fetched {_age(facts['fetched_at'])}"
            if facts.get("updated_at"):
                ages += f", GitHub updated_at {_age(facts['updated_at'])}"
            how = " (filtered from a broader stored listing)" if facts["outcome"] == "subsumed" else ""
            lines.append(
                f"Context Graph memory already has {address.key}{how} ({ages}); "
                "the `resource` tool returns it without fetching."
            )
        return "\n".join(lines) or None

    def _on_tool_start(self, event: ToolStartEvent) -> None:
        if is_resource_tool(event.tool_name):
            return
        for address in addresses_from_tool(event.tool_name, event.tool_input):
            self._graph.record_touch(
                event.session_id,
                address,
                FETCHED,
                discriminator=event.tool_use_id or event.timestamp,
                tool_use_id=event.tool_use_id,
                agent_name=event.agent_name,
                at=event.timestamp,
            )

    def _on_tool_end(self, event: ToolEndEvent) -> None:
        if event.is_error or not is_resource_tool(event.tool_name):
            return
        match = OUTCOME_LINE.search(_text(event.result))
        if not match:
            return
        try:
            address = Address.from_key(match[2])
        except ValueError:
            return
        self._graph.record_cache_read(
            event.session_id,
            address,
            match[1],
            discriminator=event.tool_use_id or event.timestamp,
            served_from=match[3] or match[2],
            tool_use_id=event.tool_use_id,
            agent_name=event.agent_name,
            at=event.timestamp,
        )

    def _on_prompt(self, event: MessageEvent) -> None:
        text = _text(event.content)
        for address in addresses_from_text(text):
            self._graph.record_touch(
                event.session_id,
                address,
                PROMPTED,
                discriminator=event.timestamp,
                agent_name=event.agent_name,
                at=event.timestamp,
            )


def _age(timestamp: str | None) -> str:
    """``2h ago``-style age of an ISO timestamp, for a line the model reads at a glance."""
    if not timestamp:
        return "at an unknown time"
    try:
        then = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp
    seconds = max(0, int((datetime.now(timezone.utc) - then).total_seconds()))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return "just now"


def _text(value: Any) -> str:
    """Flatten a hook payload value (string, MCP content blocks, nested dicts) to searchable text."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_text(item) for item in value)
    return "" if value is None else str(value)
