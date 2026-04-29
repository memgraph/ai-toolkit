"""Agent-link connector for SkillGraph.

This module implements the :class:`agent_graph.GraphConnector` interface
so that a ``SkillGraph`` can receive events directly from an ``AgentLink``
hub.  The connector lives *inside* skills-graph because **the library
itself** knows which events are relevant and how to persist them — the
agent-graph package only provides the plumbing.

Usage::

    from skills_graph import SkillGraph
    from skills_graph.connector import SkillGraphConnector
    from agent_graph import AgentLink
    from agent_graph.adapters.claude import ClaudeAdapter

    link = AgentLink()
    link.add_connector(SkillGraphConnector(SkillGraph()))
    adapter = ClaudeAdapter(link, session_id="s-1")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_graph.events import Event, EventType, ToolEndEvent, ToolStartEvent
from agent_graph.protocols import GraphConnector

if TYPE_CHECKING:
    from .core import SkillGraph


_SUPPORTED_EVENTS = {
    EventType.TOOL_START,
    EventType.TOOL_END,
}


class SkillGraphConnector(GraphConnector):
    """Receives agent-graph events and records skill usage in a SkillGraph.

    Watches for tool calls whose name matches a skill operation
    (``get_skill``, ``add_skill``, etc.) and creates
    ``(:Session)-[:USED_SKILL]->(:Skill)`` relationships.

    Args:
        graph: An initialised SkillGraph instance.
        skill_tool_names: Override the set of tool names that indicate
            skill access.  Defaults to the built-in CRUD + search names.
    """

    DEFAULT_SKILL_TOOLS: frozenset[str] = frozenset(
        {
            "get_skill",
            "add_skill",
            "update_skill",
            "delete_skill",
            "list_skills",
            "search_skills",
            "search_by_tags",
            "search_by_name",
        }
    )

    def __init__(
        self,
        graph: SkillGraph,
        *,
        skill_tool_names: set[str] | None = None,
    ) -> None:
        self._graph = graph
        self._skill_tool_names = skill_tool_names or set(self.DEFAULT_SKILL_TOOLS)

    # ------------------------------------------------------------------
    # GraphConnector interface
    # ------------------------------------------------------------------

    def supports(self, event: Event) -> bool:
        if event.event_type not in _SUPPORTED_EVENTS:
            return False
        if isinstance(event, ToolStartEvent | ToolEndEvent):
            return event.tool_name in self._skill_tool_names
        return False

    def on_event(self, event: Event) -> None:
        if isinstance(event, ToolStartEvent):
            self._on_tool_start(event)
        elif isinstance(event, ToolEndEvent):
            self._on_tool_end(event)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_tool_start(self, event: ToolStartEvent) -> None:
        """A skill-related tool was invoked — record the access."""
        skill_name = self._extract_skill_name(event.tool_input)
        if skill_name:
            self._record_skill_access(
                session_id=event.session_id,
                skill_name=skill_name,
                action=event.tool_name,
                timestamp=event.timestamp,
            )

    def _on_tool_end(self, event: ToolEndEvent) -> None:
        """A search/list tool returned — record which skills appeared."""
        if event.tool_name not in {"list_skills", "search_skills", "search_by_tags", "search_by_name"}:
            return
        if not isinstance(event.result, list):
            return
        for item in event.result:
            name = item.get("name") if isinstance(item, dict) else None
            if name:
                self._record_skill_access(
                    session_id=event.session_id,
                    skill_name=name,
                    action=f"{event.tool_name}_result",
                    timestamp=event.timestamp,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_skill_name(tool_input: Any) -> str | None:
        """Pull the skill name out of the tool's input dict."""
        if not isinstance(tool_input, dict):
            return None
        return tool_input.get("name") or tool_input.get("skill_name") or tool_input.get("pattern")

    def _record_skill_access(
        self,
        *,
        session_id: str,
        skill_name: str,
        action: str,
        timestamp: str,
    ) -> None:
        """Persist a ``(:Session)-[:USED_SKILL]->(:Skill)`` edge."""
        self._graph._db.query(
            """
            MERGE (sess:Session {session_id: $session_id})
            WITH sess
            MATCH (sk:Skill {name: $skill_name})
            MERGE (sess)-[r:USED_SKILL]->(sk)
            ON CREATE SET r.first_access = $timestamp,
                          r.access_count = 1,
                          r.actions = [$action]
            ON MATCH SET r.last_access = $timestamp,
                         r.access_count = r.access_count + 1,
                         r.actions = r.actions + $action
            """,
            params={
                "session_id": session_id,
                "skill_name": skill_name,
                "timestamp": timestamp,
                "action": action,
            },
        )
