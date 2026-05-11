"""Claude Code hooks adapter for agent-context-graph.

Claude Code hooks are command-based: Claude Code invokes a configured command
with the hook payload on stdin. This adapter translates those JSON payloads into
the common Event protocol used by AgentLink.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Any

from agent_context_graph.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorOccurredEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_context_graph.link import AgentLink
from agent_context_graph.protocols import RuntimeAdapter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from agent_context_graph.events import Event

_SOURCE = "claude-code"
_DEFAULT_COMMAND = "agent-context-graph hook run claude-code"
_SUPPORTED_HOOKS = (
    "SessionStart",
    "UserPromptSubmit",
    "UserPromptExpansion",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionRequest",
    "PermissionDenied",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "StopFailure",
)


class ClaudeCodeHooksAdapter(RuntimeAdapter):
    """Adapter that converts Claude Code hook payloads into graph events."""

    def __init__(self, link: AgentLink, session_id: str | None = None) -> None:
        self._link = link
        self._session_id = session_id

    def get_runtime_hooks(self) -> dict[str, list[dict[str, Any]]]:
        """Return a hooks.json-compatible config skeleton."""
        return build_hooks_config(_DEFAULT_COMMAND)

    def handle_payload(self, payload: dict[str, Any]) -> list[Event]:
        """Translate and emit a Claude Code hook payload."""
        hook_event_name = payload.get("hook_event_name")
        events = self._events_from_payload(hook_event_name, payload)
        for event in events:
            self._link.emit(event)
        return events

    def _events_from_payload(self, hook_event_name: Any, payload: dict[str, Any]) -> list[Event]:
        session_id = self._session_id or str(payload.get("session_id") or "")
        metadata = _metadata_from_payload(payload)

        if hook_event_name == "SessionStart":
            return [
                SessionStartEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    working_directory=_string_or_none(payload.get("cwd")),
                    metadata=metadata,
                )
            ]

        if hook_event_name in {"UserPromptSubmit", "UserPromptExpansion"}:
            return [
                MessageEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    role="user",
                    content=payload.get("prompt", ""),
                    metadata=metadata,
                )
            ]

        if hook_event_name == "PreToolUse":
            return [
                ToolStartEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    tool_name=str(payload.get("tool_name") or ""),
                    tool_input=payload.get("tool_input"),
                    tool_use_id=_string_or_none(payload.get("tool_use_id")),
                    metadata=metadata,
                )
            ]

        if hook_event_name == "PostToolUse":
            tool_response = payload.get("tool_response")
            if "tool_input" in payload:
                metadata["tool_input"] = payload.get("tool_input")
            return [
                ToolEndEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    tool_name=str(payload.get("tool_name") or ""),
                    tool_use_id=_string_or_none(payload.get("tool_use_id")),
                    result=tool_response,
                    metadata=metadata,
                )
            ]

        if hook_event_name == "PostToolUseFailure":
            if "tool_input" in payload:
                metadata["tool_input"] = payload.get("tool_input")
            return [
                ToolEndEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    tool_name=str(payload.get("tool_name") or ""),
                    tool_use_id=_string_or_none(payload.get("tool_use_id")),
                    is_error=True,
                    error_message=_string_or_none(payload.get("error")),
                    metadata=metadata,
                )
            ]

        if hook_event_name == "PermissionRequest":
            return [
                MessageEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    role="system",
                    content=str(payload.get("tool_name") or "permission_request"),
                    metadata=metadata,
                )
            ]

        if hook_event_name == "PermissionDenied":
            return [
                ErrorOccurredEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    error_type="permission_denied",
                    error_message=str(payload.get("reason") or "Permission denied"),
                    metadata=metadata,
                    recoverable=True,
                )
            ]

        if hook_event_name == "SubagentStart":
            agent_name = str(payload.get("agent_id") or payload.get("agent_type") or "")
            return [
                AgentStartEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    agent_name=agent_name,
                    agent_type=str(payload.get("agent_type") or ""),
                    metadata=metadata,
                )
            ]

        if hook_event_name == "SubagentStop":
            agent_name = str(payload.get("agent_id") or payload.get("agent_type") or "")
            return [
                AgentEndEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    agent_name=agent_name,
                    agent_type=str(payload.get("agent_type") or ""),
                    output=payload.get("last_assistant_message"),
                    metadata=metadata,
                )
            ]

        if hook_event_name == "Stop":
            return [
                SessionEndEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    status="completed",
                    metadata=metadata,
                )
            ]

        if hook_event_name == "StopFailure":
            return [
                ErrorOccurredEvent(
                    session_id=session_id,
                    source_sdk=_SOURCE,
                    error_type=str(payload.get("error") or "unknown"),
                    error_message=str(
                        payload.get("error_details") or payload.get("error") or "Claude Code stop failure"
                    ),
                    metadata=metadata,
                    recoverable=True,
                )
            ]

        return []


ClaudeCodeAdapter = ClaudeCodeHooksAdapter


def build_hooks_config(command: str, *, timeout: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Build a Claude Code hooks config using *command* for every supported hook."""
    config: dict[str, list[dict[str, Any]]] = {}
    for hook_name in _SUPPORTED_HOOKS:
        entry: dict[str, Any] = {
            "hooks": [
                {
                    "type": "command",
                    "command": command,
                    "timeout": timeout,
                }
            ]
        }
        if hook_name == "SessionStart":
            entry["matcher"] = "startup|resume|clear"
        elif hook_name in {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "PermissionRequest",
            "PermissionDenied",
        }:
            entry["matcher"] = "*"
        config[hook_name] = [entry]
    return config


def load_payload(stream: Any | None = None) -> dict[str, Any]:
    """Read one Claude Code hook payload from a text stream."""
    if stream is None:
        stream = sys.stdin
    raw = stream.read()
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        msg = "Claude Code hook payload must be a JSON object"
        raise TypeError(msg)
    return payload


def create_link(connector_names: Iterable[str] = ()) -> AgentLink:
    """Create an AgentLink with optional connectors named by CLI/config."""
    link = AgentLink()
    for connector_name in connector_names:
        normalized = connector_name.strip().replace("-", "_")
        if not normalized:
            continue
        if normalized == "skills_graph":
            _add_skills_graph_connector(link)
        else:
            msg = f"Unsupported connector: {connector_name}"
            raise ValueError(msg)
    return link


def response_for_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return hook JSON response, when Claude Code benefits from one."""
    hook_event_name = payload.get("hook_event_name")
    if hook_event_name in {"Stop", "SubagentStop"}:
        return {"continue": True}
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge Claude Code hooks to agent-context-graph.")
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to enable. Currently supported: skills-graph.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Override the session id from the Claude Code hook payload.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero status if the hook payload cannot be recorded.",
    )
    args = parser.parse_args(argv)

    connector_names = args.connector
    if connector_names is None:
        connector_names = _connectors_from_env()

    payload: dict[str, Any] = {}
    try:
        payload = load_payload()
        link = create_link(connector_names)
        adapter = ClaudeCodeHooksAdapter(link, session_id=args.session_id)
        adapter.handle_payload(payload)
        response = response_for_payload(payload)
        if response is not None:
            print(json.dumps(response))
    except Exception as exc:
        if args.strict or os.environ.get("AGENT_CONTEXT_GRAPH_CLAUDE_CODE_STRICT") == "1":
            raise
        response = response_for_payload(payload)
        if response is not None:
            print(json.dumps(response))
        _debug_log(f"agent-context-graph Claude Code hook skipped: {exc}")
    return 0


def _add_skills_graph_connector(link: AgentLink) -> None:
    try:
        from skills_graph import SkillGraph
        from skills_graph.connector import SkillGraphConnector
    except ImportError as exc:
        msg = "skills-graph is required for the skills-graph Claude Code connector"
        raise ImportError(msg) from exc

    graph = SkillGraph()
    link.add_connector(SkillGraphConnector(graph))


def _connectors_from_env() -> list[str]:
    value = os.environ.get("AGENT_CONTEXT_GRAPH_CLAUDE_CODE_CONNECTORS", "")
    return [part.strip() for part in value.split(",") if part.strip()]


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "cwd",
        "transcript_path",
        "permission_mode",
        "tool_name",
        "tool_input",
        "tool_use_id",
        "tool_response",
        "duration_ms",
        "error",
        "error_details",
        "is_interrupt",
        "reason",
        "stop_hook_active",
        "last_assistant_message",
        "agent_id",
        "agent_type",
        "agent_transcript_path",
        "command_name",
        "command_args",
        "command_source",
        "expansion_type",
    ):
        if key in payload and payload.get(key) is not None:
            metadata[key] = payload.get(key)
    return metadata


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _debug_log(message: str) -> None:
    if os.environ.get("AGENT_CONTEXT_GRAPH_CLAUDE_CODE_DEBUG") == "1":
        print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
