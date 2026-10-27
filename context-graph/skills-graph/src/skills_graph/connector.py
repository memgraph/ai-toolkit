"""Memgraph Context Graph connector for SkillGraph.

Lives in skills-graph because this package decides which events prove skill
usage; memgraph-context-graph only routes them.

Usage::

    from skills_graph import SkillGraph
    from skills_graph.connector import SkillGraphConnector
    from memgraph_context_graph import AgentLink
    from memgraph_context_graph.sdk import ContextGraphMemory

    link = AgentLink()
    link.add_connector(SkillGraphConnector(SkillGraph()))
    memory = ContextGraphMemory(link=link)  # then ClaudeAgentMemory(memory) or OpenAIAgentsMemory(memory)
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path, PurePath
from typing import Any

from memgraph_context_graph.connectors import ConnectorSpec
from memgraph_context_graph.events import Event, EventType, MessageEvent, ToolEndEvent, ToolStartEvent
from memgraph_context_graph.protocols import GraphConnector

from .core import SkillGraph

_SUPPORTED_EVENTS = {
    EventType.TOOL_START,
    EventType.TOOL_END,
    EventType.MESSAGE,
}
_MAX_RESULT_DEPTH = 10
_SKILL_FILE_NAME = "SKILL.md"
# Claude Code's own skill tool: ``tool_input = {"skill": "<name>" | "<plugin>:<name>", "args": ...}``.
_CLAUDE_CODE_SKILL_TOOL = "Skill"
# Claude Code prefixes a skill's injected content with this line.
_SKILL_EXPANSION_MARKER = "Base directory for this skill:"
# Dedicated file-read tools across the supported runtimes, lowercased.
_FILE_READ_TOOLS = frozenset({"read", "read_file", "readfile", "view", "view_file"})
# Programs whose file arguments are read, not written, searched or staged.
_READ_PROGRAMS = frozenset({"cat", "head", "tail", "sed", "less", "more", "bat", "nl"})
# Unquoted newlines separate commands like ``;`` does; quoted ones stay inside their word.
_SHELL_PUNCTUATION = "();<>|&\n"
_HEREDOC_START = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
# Characters that mean the shell had not expanded the path yet, so it names no one file.
_UNEXPANDED = frozenset("$*?[{`")
_SKILL_TOOLS = frozenset(
    {
        "get_skill",
        "add_skill",
        "update_skill",
        "delete_skill",
        "list_skills",
        "search_skills",
        "search_by_name",
    }
)


class SkillGraphConnector(GraphConnector):
    """Records ``(:Session|:Agent)-[:USED_SKILL]->(:Skill)`` from skill tool calls, ``SKILL.md`` reads, and
    skill content injected into prompts.

    Args:
        graph: An initialised SkillGraph instance.
    """

    def __init__(self, graph: SkillGraph) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # GraphConnector interface
    # ------------------------------------------------------------------

    def supports(self, event: Event) -> bool:
        if event.event_type not in _SUPPORTED_EVENTS:
            return False
        if isinstance(event, ToolStartEvent):
            if event.tool_name == _CLAUDE_CODE_SKILL_TOOL or self._operation_name(event.tool_name) in _SKILL_TOOLS:
                return True
            return bool(self._skill_files_read(event.tool_name, event.tool_input, _event_cwd(event)))
        if isinstance(event, ToolEndEvent):
            return self._operation_name(event.tool_name) in _SKILL_TOOLS
        if isinstance(event, MessageEvent):
            return self._extract_skill_content(event.content) is not None
        return False

    def on_event(self, event: Event) -> None:
        if isinstance(event, ToolStartEvent):
            self._on_tool_start(event)
        elif isinstance(event, ToolEndEvent):
            self._on_tool_end(event)
        elif isinstance(event, MessageEvent):
            self._on_message(event)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_tool_start(self, event: ToolStartEvent) -> None:
        """A skill was invoked, or its ``SKILL.md`` read — record the access."""
        if event.tool_name == _CLAUDE_CODE_SKILL_TOOL:
            self._on_claude_code_skill(event)
            return

        skill_files = self._skill_files_read(event.tool_name, event.tool_input, _event_cwd(event))
        for skill_file in skill_files:
            metadata = self._metadata_from_skill_file(skill_file)
            self._record_local_skill(event, metadata, fallback_name=skill_file.parent.name, action="read_skill_file")
        if skill_files:
            return

        skill_name = self._extract_skill_name(event.tool_input)
        if skill_name:
            self._graph.record_skill_usage(
                session_id=event.session_id,
                skill_name=skill_name,
                action=self._operation_name(event.tool_name),
                timestamp=event.timestamp,
                container_agent_id=event.agent_name,
            )

    def _on_claude_code_skill(self, event: ToolStartEvent) -> None:
        """Claude Code's ``Skill`` tool: the name is the evidence, the ``SKILL.md`` only enriches the node."""
        invoked = event.tool_input.get("skill") if isinstance(event.tool_input, dict) else None
        if not isinstance(invoked, str) or not invoked.strip():
            return
        plugin, _, name = invoked.strip().rpartition(":")
        skill_file = _resolve_installed_skill(name, plugin=plugin or None, cwd=_event_cwd(event))
        if skill_file is not None:
            metadata = self._metadata_from_skill_file(skill_file)
        else:
            metadata = {"content": "", "metadata": {"source": "claude_code_skill_tool"}}
        self._record_local_skill(event, {**metadata, "name": name}, fallback_name=name, action="invoke_skill")

    def _record_local_skill(self, event: Event, metadata: dict[str, Any], *, fallback_name: str, action: str) -> None:
        self._graph.record_skill_usage(
            session_id=event.session_id,
            skill_name=metadata.get("name") or fallback_name,
            action=action,
            timestamp=event.timestamp,
            create_missing=True,
            description=metadata.get("description", ""),
            content=metadata.get("content", ""),
            source_path=metadata.get("metadata", {}).get("source_path"),
            metadata=metadata.get("metadata", {}),
            container_agent_id=getattr(event, "agent_name", None),
        )

    def _on_tool_end(self, event: ToolEndEvent) -> None:
        """A search/list tool returned — record which skills appeared."""
        operation_name = self._operation_name(event.tool_name)
        if operation_name not in {"list_skills", "search_skills", "search_by_name"}:
            return
        for name in self._extract_result_skill_names(event.result):
            self._graph.record_skill_usage(
                session_id=event.session_id,
                skill_name=name,
                action=f"{operation_name}_result",
                timestamp=event.timestamp,
                container_agent_id=event.agent_name,
            )

    def _on_message(self, event: MessageEvent) -> None:
        """Skill content Claude Code injected into the conversation — record the skill."""
        skill_content = self._extract_skill_content(event.content)
        if skill_content is None:
            return
        base_directory = _expansion_base_directory(skill_content)
        metadata = self._metadata_from_skill_content(
            skill_content,
            source_path=str(base_directory / _SKILL_FILE_NAME) if base_directory is not None else None,
        )
        skill_name = metadata.get("name")
        if isinstance(skill_name, str) and skill_name:
            self._record_local_skill(event, metadata, fallback_name=skill_name, action="prompt_expansion")

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
    def _skill_files_read(cls, tool_name: str, tool_input: Any, cwd: Path | None) -> list[Path]:
        """``SKILL.md`` files this tool call reads, from its own input only.

        A path counts only when the call reads it: a file-read tool's arguments,
        or a read program's arguments in a shell command. A path that is staged,
        searched for, written, or merely mentioned (a subagent prompt, say) is not
        a read, so it is not Skill Usage.
        """
        if not isinstance(tool_input, dict):
            return []
        workdir = tool_input.get("workdir")
        if isinstance(workdir, str) and workdir:
            cwd = _resolve_path(workdir, cwd)
        if cls._operation_name(tool_name).lower() in _FILE_READ_TOOLS:
            reads = [(value, cwd) for value in tool_input.values() if isinstance(value, str)]
        else:
            command = _command_text(tool_input.get("command", tool_input.get("cmd")))
            reads = _shell_reads(command, cwd) if command is not None else []
        skill_files: list[Path] = []
        for candidate, directory in reads:
            skill_file = _skill_file_from_candidate(candidate, directory)
            if skill_file is not None and skill_file not in skill_files:
                skill_files.append(skill_file)
        return skill_files

    @staticmethod
    def _metadata_from_skill_file(path: Path) -> dict[str, Any]:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return {
                "name": path.parent.name,
                "content": "",
                "metadata": {"source": "local_skill_file", "source_path": str(path)},
            }
        return SkillGraphConnector._metadata_from_skill_content(
            content, source_path=str(path), fallback_name=path.parent.name
        )

    @staticmethod
    def _metadata_from_skill_content(
        content: str,
        *,
        source_path: str | None = None,
        fallback_name: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "content": content,
            "metadata": {"source": "local_skill_file"},
        }
        if source_path is not None:
            metadata["metadata"]["source_path"] = source_path

        frontmatter = _parse_frontmatter(content)
        name = frontmatter.get("name")
        description = frontmatter.get("description")
        if isinstance(name, str) and name:
            metadata["name"] = name
        elif fallback_name:
            metadata["name"] = fallback_name
        if isinstance(description, str):
            metadata["description"] = description
        metadata_keys = {"version", "category", "author", "last_updated", "license", "compatibility"}
        metadata["metadata"].update({key: str(value) for key, value in frontmatter.items() if key in metadata_keys})
        return metadata

    @classmethod
    def _extract_skill_content(cls, result: Any, *, _depth: int = 0) -> str | None:
        if _depth > _MAX_RESULT_DEPTH:
            return None
        if isinstance(result, str):
            return result if _is_skill_expansion(result) else None
        if isinstance(result, list):
            for item in result:
                content = cls._extract_skill_content(item, _depth=_depth + 1)
                if content is not None:
                    return content
            return None
        if isinstance(result, dict):
            for key in ("content", "text"):
                if key in result:
                    content = cls._extract_skill_content(result[key], _depth=_depth + 1)
                    if content is not None:
                        return content
        return None


def _event_cwd(event: Event) -> Path | None:
    cwd = event.metadata.get("cwd") if isinstance(event.metadata, dict) else None
    return Path(cwd) if isinstance(cwd, str) and Path(cwd).is_absolute() else None


def _command_text(command: Any) -> str | None:
    """A shell tool's command as one string; argv lists (``["bash", "-lc", "<script>"]``) yield the script."""
    if isinstance(command, str):
        return command
    if isinstance(command, list) and command and all(isinstance(word, str) for word in command):
        if len(command) >= 3 and command[-2] in ("-c", "-lc"):
            return command[-1]
        return shlex.join(command)
    return None


def _shell_reads(command: str, cwd: Path | None) -> list[tuple[str, Path | None]]:
    """File arguments of the read programs in *command*, each with the directory the shell resolves it in.

    ``cd`` moves that directory for the commands after it; a ``cd`` the connector
    can't follow leaves it unknown (None), and later relative paths are kept as written.
    """
    reads: list[tuple[str, Path | None]] = []
    directory = cwd
    for segment in _shell_segments(_without_heredoc_bodies(command)):
        while segment and "=" in segment[0] and not segment[0].startswith("-"):
            segment = segment[1:]  # leading VAR=value assignments
        if not segment:
            continue
        program, *rest = segment
        program = PurePath(program).name
        if program == "cd":
            targets = [word for word in rest if not word.startswith("-")]
            target = targets[0] if targets else "~"
            directory = None if _UNEXPANDED.intersection(target) else _resolve_path(target, directory)
            continue
        if program not in _READ_PROGRAMS:
            continue
        if program == "sed" and any(word.startswith(("-i", "--in-place")) for word in rest):
            continue
        reads.extend((word, directory) for word in rest if not word.startswith("-"))
    return reads


def _without_heredoc_bodies(command: str) -> str:
    """*command* with each heredoc's body removed: it is input text, not commands."""
    kept: list[str] = []
    delimiters: list[str] = []
    for line in command.splitlines():
        if delimiters:
            if line.strip() == delimiters[0]:
                delimiters.pop(0)
            continue
        kept.append(line)
        delimiters = [match.group(2) for match in _HEREDOC_START.finditer(line.replace("<<<", ""))]
    return "\n".join(kept)


def _shell_segments(command: str) -> list[list[str]]:
    """Split *command* into simple commands, dropping redirection targets."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    segments: list[list[str]] = [[]]
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
        elif set(token) <= set(_SHELL_PUNCTUATION):
            if "<" in token or ">" in token:
                skip_next = True  # the redirection's target is not an argument
            else:
                segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _resolve_path(path: str, base: Path | None) -> Path | None:
    """*path* as the shell would resolve it from *base*; None when it is relative to an unknown directory."""
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        if base is None:
            return None
        expanded = base / expanded
    return Path(os.path.normpath(expanded))


def _skill_file_from_candidate(candidate: str, cwd: Path | None) -> Path | None:
    candidate = candidate.strip()
    if not candidate or _UNEXPANDED.intersection(candidate) or PurePath(candidate).name != _SKILL_FILE_NAME:
        return None
    path = _resolve_path(candidate, cwd)
    if path is None and PurePath(candidate).parent.name:
        path = Path(candidate)  # relative, with no known directory: keep the path as given
    if path is None or path.parent.name == "":
        return None
    return path


def _resolve_installed_skill(name: str, *, plugin: str | None, cwd: Path | None) -> Path | None:
    """The ``SKILL.md`` Claude Code would load for *name*, from its standard skill locations."""
    if any(_UNEXPANDED.intersection(part) or "/" in part for part in (name, plugin or "")) or not name:
        return None
    home = Path.home()
    if plugin:
        cached: list[Path] = list(home.glob(f".claude/plugins/cache/*/{plugin}/*/skills/{name}/{_SKILL_FILE_NAME}"))
        return max(cached, key=lambda path: path.stat().st_mtime) if cached else None
    roots = [cwd / ".claude" / "skills"] if cwd is not None else []
    roots += [home / ".claude" / "skills", home / ".agents" / "skills"]
    return next((root / name / _SKILL_FILE_NAME for root in roots if (root / name / _SKILL_FILE_NAME).is_file()), None)


def _expansion_base_directory(content: str) -> Path | None:
    first_line = content.lstrip().splitlines()[0]
    directory = first_line.removeprefix(_SKILL_EXPANSION_MARKER).strip()
    return Path(directory) if directory else None


def _parse_frontmatter(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    start_index = _frontmatter_start_index(lines)
    if start_index is None:
        return {}

    data: dict[str, Any] = {}
    key_for_list: str | None = None
    for line in lines[start_index + 1 :]:
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


def _frontmatter_start_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            return index
        if stripped and not stripped.startswith("Base directory for this skill:"):
            return None
    return None


def _is_skill_expansion(content: str) -> bool:
    """Skill content as Claude Code injects it: the base-directory marker, then name and description."""
    if not content.lstrip().startswith(_SKILL_EXPANSION_MARKER):
        return False
    frontmatter = _parse_frontmatter(content)
    name, description = frontmatter.get("name"), frontmatter.get("description")
    return isinstance(name, str) and bool(name) and isinstance(description, str) and bool(description)


CONNECTOR = ConnectorSpec(
    name="skills-graph",
    graph=SkillGraph,
    build=lambda graph, config: SkillGraphConnector(graph),
)
