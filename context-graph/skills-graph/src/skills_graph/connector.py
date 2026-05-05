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
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_context_graph.events import Event, EventType, ToolEndEvent, ToolStartEvent
from agent_context_graph.protocols import GraphConnector

if TYPE_CHECKING:
    from .core import SkillGraph


_SUPPORTED_EVENTS = {
    EventType.TOOL_START,
    EventType.TOOL_END,
}
_MAX_RESULT_DEPTH = 10
_SKILL_FILE_NAME = "SKILL.md"


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
        if isinstance(event, ToolStartEvent):
            if self._operation_name(event.tool_name) in self._skill_tool_names:
                return True
            return self._extract_skill_file_read(event.tool_input) is not None
        if isinstance(event, ToolEndEvent):
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
        skill_file = self._extract_skill_file_read(event.tool_input)
        if skill_file is not None:
            metadata = self._metadata_from_skill_file(skill_file)
            skill_name = metadata.get("name") or skill_file.parent.name
            self._record_skill_access(
                session_id=event.session_id,
                skill_name=skill_name,
                action="read_skill_file",
                timestamp=event.timestamp,
                create_missing=True,
                description=metadata.get("description", ""),
                content=metadata.get("content", ""),
                source_path=str(skill_file),
                tags=metadata.get("tags", []),
                metadata=metadata.get("metadata", {}),
            )
            return

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
        """Return the operation from direct or MCP ``mcp__<server>__<operation>`` names."""
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
    def _extract_result_skill_names(cls, result: Any, *, _depth: int = 0) -> list[str]:
        """Pull skill names out of direct Python results or JSON tool content."""
        if _depth > _MAX_RESULT_DEPTH:
            return []

        if isinstance(result, str):
            parsed = cls._parse_json_result(result)
            if parsed is not None:
                return cls._extract_result_skill_names(parsed, _depth=_depth + 1)
            return []

        extracted_names: list[str] = []
        if isinstance(result, list):
            for item in result:
                extracted_names.extend(cls._extract_result_skill_names(item, _depth=_depth + 1))
            return extracted_names

        if isinstance(result, dict):
            name = result.get("name")
            if isinstance(name, str) and name:
                return [name]
            for key in ("skills", "results", "items", "content"):
                if key in result:
                    extracted_names.extend(cls._extract_result_skill_names(result[key], _depth=_depth + 1))
            text = result.get("text")
            if isinstance(text, str):
                extracted_names.extend(cls._extract_result_skill_names(text, _depth=_depth + 1))
            return extracted_names

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

    @classmethod
    def _extract_skill_file_read(cls, tool_input: Any) -> Path | None:
        for value in cls._iter_string_values(tool_input):
            for candidate in cls._candidate_paths(value):
                skill_file = cls._skill_file_from_candidate(candidate)
                if skill_file is not None:
                    return skill_file
        return None

    @classmethod
    def _iter_string_values(cls, value: Any, *, _depth: int = 0) -> list[str]:
        if _depth > _MAX_RESULT_DEPTH:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            values: list[str] = []
            for item in value:
                values.extend(cls._iter_string_values(item, _depth=_depth + 1))
            return values
        if isinstance(value, dict):
            values = []
            for item in value.values():
                values.extend(cls._iter_string_values(item, _depth=_depth + 1))
            return values
        return []

    @staticmethod
    def _candidate_paths(value: str) -> list[str]:
        try:
            candidates = shlex.split(value)
        except ValueError:
            candidates = value.split()
        return candidates or [value]

    @staticmethod
    def _skill_file_from_candidate(candidate: str) -> Path | None:
        if _SKILL_FILE_NAME not in candidate:
            return None

        path = Path(candidate.strip())
        if path.name != _SKILL_FILE_NAME:
            return None
        if path.parent.name == "":
            return None
        return path

    @staticmethod
    def _metadata_from_skill_file(path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "content": "",
            "metadata": {"source": "local_skill_file", "source_path": str(path)},
            "tags": [],
        }
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            metadata["name"] = path.parent.name
            return metadata

        metadata["content"] = content
        frontmatter = _parse_frontmatter(content)
        name = frontmatter.get("name")
        description = frontmatter.get("description")
        tags = frontmatter.get("tags")
        if isinstance(name, str) and name:
            metadata["name"] = name
        else:
            metadata["name"] = path.parent.name
        if isinstance(description, str):
            metadata["description"] = description
        if isinstance(tags, list):
            metadata["tags"] = [tag for tag in tags if isinstance(tag, str) and tag]
        metadata["metadata"].update(
            {key: str(value) for key, value in frontmatter.items() if key not in {"name", "description", "tags"}}
        )
        return metadata

    def _record_skill_access(
        self,
        *,
        session_id: str,
        skill_name: str,
        action: str,
        timestamp: str,
        create_missing: bool = False,
        description: str = "",
        content: str = "",
        source_path: str | None = None,
        metadata: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Persist a ``(:Session)-[:USED_SKILL]->(:Skill)`` edge."""
        self._graph.record_skill_usage(
            session_id=session_id,
            skill_name=skill_name,
            action=action,
            timestamp=timestamp,
            create_missing=create_missing,
            description=description,
            content=content,
            source_path=source_path,
            metadata=metadata,
            tags=tags,
        )


def _parse_frontmatter(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    data: dict[str, Any] = {}
    key_for_list: str | None = None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            return data
        if not stripped:
            continue
        if stripped.startswith("- ") and key_for_list:
            data.setdefault(key_for_list, []).append(stripped[2:].strip().strip("\"'"))
            continue
        if ":" not in line:
            key_for_list = None
            continue
        key, value = line.split(":", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if value:
            data[key] = value.strip("\"'")
            key_for_list = None
        else:
            data[key] = []
            key_for_list = key
    return {}
