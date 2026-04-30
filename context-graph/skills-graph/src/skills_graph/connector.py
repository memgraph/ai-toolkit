"""Agent-link connector for SkillGraph.

This module implements the :class:`agent_context_graph.GraphConnector` interface
so that a ``SkillGraph`` can receive events directly from an ``AgentLink``
hub.  The connector lives *inside* skills-graph because **the library
itself** knows which events are relevant and how to persist them — the
agent-context-graph package only provides the plumbing.

Usage::

    from skills_graph import SkillGraph
    from skills_graph.connector import SkillGraphConnector
    from agent_context_graph import AgentLink
    from agent_context_graph.adapters.claude import ClaudeAdapter

    link = AgentLink()
    link.add_connector(SkillGraphConnector(SkillGraph()))
    adapter = ClaudeAdapter(link, session_id="s-1")
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent_context_graph.events import Event, EventType, ToolEndEvent, ToolStartEvent
from agent_context_graph.protocols import GraphConnector

if TYPE_CHECKING:
    from .core import SkillGraph


_SUPPORTED_EVENTS = {
    EventType.TOOL_START,
    EventType.TOOL_END,
}


class SkillGraphConnector(GraphConnector):
    """Receives agent-context-graph events and records skill usage in a SkillGraph.

    Watches for tool calls whose name matches a skill access operation
    (``get_skill``, ``update_skill``, search/list tools, etc.) and creates
    ``(:Session)-[:USED_SKILL]->(:Skill)`` relationships.

    Args:
        graph: An initialised SkillGraph instance.
        skill_tool_names: Override the set of tool names that indicate
            skill access. Defaults to built-in read/update/search names.
    """

    DEFAULT_SKILL_TOOLS: frozenset[str] = frozenset(
        {
            "get_skill",
            "update_skill",
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
            return self._operation_name(event.tool_name) in self._skill_tool_names
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
                action=self._operation_name(event.tool_name),
                timestamp=event.timestamp,
            )

    def _on_tool_end(self, event: ToolEndEvent) -> None:
        """A search/list tool returned — record which skills appeared."""
        operation_name = self._operation_name(event.tool_name)
        if operation_name not in {"list_skills", "search_skills", "search_by_tags", "search_by_name"}:
            return
        for name in self._extract_result_skill_names(event.result):
            self._record_skill_access(
                session_id=event.session_id,
                skill_name=name,
                action=f"{operation_name}_result",
                timestamp=event.timestamp,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _operation_name(tool_name: str) -> str:
        """Return the local tool name from direct or Codex MCP-style names."""
        if tool_name.startswith("mcp__"):
            return tool_name.rsplit("__", maxsplit=1)[-1]
        return tool_name

    @classmethod
    def _extract_skill_name(cls, tool_input: Any) -> str | None:
        """Pull the skill name out of the tool's input dict."""
        if not isinstance(tool_input, dict):
            return None
        for key in ("name", "skill_name", "skill", "pattern"):
            value = tool_input.get(key)
            if isinstance(value, str) and value:
                return value
        for nested_key in ("arguments", "params", "input"):
            nested = tool_input.get(nested_key)
            nested_name = cls._extract_skill_name(nested)
            if nested_name:
                return nested_name
        return None

    @classmethod
    def _extract_result_skill_names(cls, result: Any) -> list[str]:
        """Pull skill names out of direct Python results or JSON tool content."""
        if isinstance(result, str):
            parsed = cls._parse_json_result(result)
            if parsed is not None:
                return cls._extract_result_skill_names(parsed)
            return []

        if isinstance(result, list):
            names: list[str] = []
            for item in result:
                names.extend(cls._extract_result_skill_names(item))
            return names

        if isinstance(result, dict):
            name = result.get("name")
            if isinstance(name, str) and name:
                return [name]
            names: list[str] = []
            for key in ("skills", "results", "items", "content"):
                if key in result:
                    names.extend(cls._extract_result_skill_names(result[key]))
            text = result.get("text")
            if isinstance(text, str):
                names.extend(cls._extract_result_skill_names(text))
            return names

        name = getattr(result, "name", None)
        if isinstance(name, str) and name:
            return [name]
        return []

    @staticmethod
    def _parse_json_result(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

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
