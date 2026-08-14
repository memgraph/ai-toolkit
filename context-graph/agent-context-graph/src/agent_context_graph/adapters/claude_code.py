"""Claude Code hooks adapter for agent-context-graph.

Claude Code hooks are command-based: Claude Code invokes a configured command
with the hook payload on stdin. This adapter translates those JSON payloads into
the common Event protocol used by AgentLink.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from agent_context_graph.hooks.runner import create_link, load_payload  # noqa: F401 (re-exported for callers)
from agent_context_graph.protocols import RuntimeAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agent_context_graph.events import Event
    from agent_context_graph.link import AgentLink

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
                    user_id=_resolve_user_id(payload),
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
                    agent_name=_string_or_none(payload.get("agent_id")),
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
                    agent_name=_string_or_none(payload.get("agent_id")),
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
                    agent_name=_string_or_none(payload.get("agent_id")),
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
                    agent_name=_string_or_none(payload.get("agent_id")),
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
                    agent_name=_string_or_none(payload.get("agent_id")),
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
        if hook_name in {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "PermissionRequest",
            "PermissionDenied",
        }:
            entry["matcher"] = "*"
        config[hook_name] = [entry]
    return config


def response_for_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return hook JSON response, when Claude Code benefits from one."""
    hook_event_name = payload.get("hook_event_name")
    if hook_event_name in {"Stop", "SubagentStop"}:
        return {"continue": True}
    return None


@dataclass(frozen=True)
class _ClaudeCodePlugin:
    """Registered under the ``agent_context_graph.runtimes`` entry point as ``PLUGIN``.

    No ``init`` -- Claude Code project-local hook setup isn't implemented yet
    (see ``hooks/cli.py``'s generic ``_init`` dispatch, which reports that
    clearly rather than assuming every runtime supports it).
    """

    name: str = "claude-code"
    adapter_class: type[RuntimeAdapter] = ClaudeCodeHooksAdapter

    def response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        return build_hooks_config(command, timeout=timeout)


PLUGIN = _ClaudeCodePlugin()


def main(argv: Sequence[str] | None = None) -> int:
    from agent_context_graph.hooks.runner import run_hook

    return run_hook(PLUGIN, argv)


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


def _resolve_user_id(payload: dict[str, Any]) -> str | None:
    """Resolve a stable user identity for SessionStartEvent.

    Resolution order:
    1. ``user_id`` field in the hook payload (forward-compat).
    2. Config file ``[identity] user_id`` at ``~/.config/context-graph/config.toml``.
    """
    from agent_context_graph.adapters._identity import resolve_user_id

    return resolve_user_id(payload)


if __name__ == "__main__":
    raise SystemExit(main())
