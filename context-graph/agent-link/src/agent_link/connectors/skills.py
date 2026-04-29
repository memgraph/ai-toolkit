"""Connector that routes agent-link events to a SkillGraph.

Tracks which skills are accessed (read) or modified (write) during
an agent session.  Listens for ToolStartEvent/ToolEndEvent whose
tool_name matches skill-related operations and records the
interactions in the SkillGraph.

Relationships created:
    (:Session)-[:USED_SKILL]->(:Skill)   — with properties: timestamp, action
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_link.events import Event, EventType, ToolEndEvent, ToolStartEvent
from agent_link.protocols import GraphConnector

if TYPE_CHECKING:
    from skills_graph import SkillGraph


_SUPPORTED_EVENTS = {
    EventType.TOOL_START,
    EventType.TOOL_END,
}


class SkillsConnector(GraphConnector):
    """Writes skill-usage events to a :class:`SkillGraph`.

    The connector watches for tool calls that interact with skills
    (e.g. tools named ``get_skill``, ``add_skill``, ``update_skill``,
    ``search_skills``).  You can customise the tool names via
    ``skill_tool_names``.

    Args:
        graph: An initialised SkillGraph instance.
        skill_tool_names: Set of tool names that indicate skill access.
    """

    DEFAULT_SKILL_TOOLS = frozenset(
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
        # Track per-session skill accesses: {session_id: {skill_name: count}}
        self._session_skills: dict[str, dict[str, int]] = {}

    def supports(self, event: Event) -> bool:
        if event.event_type not in _SUPPORTED_EVENTS:
            return False
        if isinstance(event, ToolStartEvent | ToolEndEvent):
            return event.tool_name in self._skill_tool_names
        return False

    def on_event(self, event: Event) -> None:
        if isinstance(event, ToolStartEvent):
            self._handle_tool_start(event)
        elif isinstance(event, ToolEndEvent):
            self._handle_tool_end(event)

    def _handle_tool_start(self, event: ToolStartEvent) -> None:
        """Record that a skill-related tool was invoked."""
        skill_name = self._extract_skill_name(event.tool_name, event.tool_input)
        if not skill_name:
            return

        session_skills = self._session_skills.setdefault(event.session_id, {})
        session_skills[skill_name] = session_skills.get(skill_name, 0) + 1

        # Record the access relationship in the graph
        self._record_skill_access(
            session_id=event.session_id,
            skill_name=skill_name,
            action=event.tool_name,
            timestamp=event.timestamp,
        )

    def _handle_tool_end(self, event: ToolEndEvent) -> None:
        """Track skill results (e.g. which skills were returned by search)."""
        if event.tool_name in {"list_skills", "search_skills", "search_by_tags", "search_by_name"} and isinstance(
            event.result, list
        ):
            for item in event.result:
                name = item.get("name") if isinstance(item, dict) else None
                if name:
                    self._record_skill_access(
                        session_id=event.session_id,
                        skill_name=name,
                        action=f"{event.tool_name}_result",
                        timestamp=event.timestamp,
                    )

    def get_session_skills(self, session_id: str) -> dict[str, int]:
        """Return skill-name → access-count for a session."""
        return dict(self._session_skills.get(session_id, {}))

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_skill_name(tool_name: str, tool_input: object) -> str | None:
        """Extract the skill name from tool input parameters."""
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
        """Create a (:Session)-[:USED_SKILL]->(:Skill) relationship."""
        # Use the underlying Memgraph client via the SkillGraph's internal _db
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
