"""OpenAI Codex hooks adapter for agent-context-graph.

Codex hooks are command-based: Codex invokes a configured command with the
hook payload on stdin.  This adapter translates those JSON payloads into the
common Event protocol used by AgentLink.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Any

from agent_context_graph.events import (
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_context_graph.link import AgentLink
from agent_context_graph.protocols import SDKAdapter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from agent_context_graph.events import Event

_SOURCE = "codex"
_DEFAULT_COMMAND = "agent-context-graph-hook codex"
_SUPPORTED_HOOKS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "Stop",
)

# TODO: When adding Claude Code command hooks, extract the shared stdin
# loading, connector construction, and CLI runner into a command-hook helper
# module. Keep product-specific payload mapping and stdout response semantics
# in each runtime adapter.


class CodexHooksAdapter(SDKAdapter):
    """Adapter that converts OpenAI Codex hook payloads into graph events.

    Args:
        link: The AgentLink hub to emit events to.
        session_id: Optional override for all emitted event session ids.
    """

    def __init__(self, link: AgentLink, session_id: str | None = None) -> None:
        self._link = link
        self._session_id = session_id

    def get_sdk_hooks(self) -> dict[str, list[dict[str, Any]]]:
        """Return a hooks.json-compatible config skeleton.

        Command paths are deployment-specific, so callers that need a custom
        command should use :func:`build_hooks_config`.
        """
        return build_hooks_config(_DEFAULT_COMMAND)

    def handle_payload(self, payload: dict[str, Any]) -> list[Event]:
        """Translate and emit a Codex hook payload.

        Returns the emitted events, which is mostly useful for tests and custom
        command runners.
        """
        hook_event_name = payload.get("hook_event_name")
        event = self._event_from_payload(hook_event_name, payload)
        if event is None:
            return []
        self._link.emit(event)
        return [event]

    def _event_from_payload(self, hook_event_name: Any, payload: dict[str, Any]) -> Event | None:
        session_id = self._session_id or str(payload.get("session_id") or "")
        metadata = _metadata_from_payload(payload)

        if hook_event_name == "SessionStart":
            return SessionStartEvent(
                session_id=session_id,
                source_sdk=_SOURCE,
                model=_string_or_none(payload.get("model")),
                working_directory=_string_or_none(payload.get("cwd")),
                metadata=metadata,
            )

        if hook_event_name == "UserPromptSubmit":
            return MessageEvent(
                session_id=session_id,
                source_sdk=_SOURCE,
                role="user",
                content=payload.get("prompt", ""),
                model=_string_or_none(payload.get("model")),
                metadata=metadata,
            )

        if hook_event_name == "PreToolUse":
            return ToolStartEvent(
                session_id=session_id,
                source_sdk=_SOURCE,
                tool_name=str(payload.get("tool_name") or ""),
                tool_input=payload.get("tool_input"),
                tool_use_id=_string_or_none(payload.get("tool_use_id")),
                metadata=metadata,
            )

        if hook_event_name == "PostToolUse":
            tool_response = payload.get("tool_response")
            result, is_error, error_message = _extract_tool_result(tool_response)
            if "tool_input" in payload:
                metadata["tool_input"] = payload.get("tool_input")
            return ToolEndEvent(
                session_id=session_id,
                source_sdk=_SOURCE,
                tool_name=str(payload.get("tool_name") or ""),
                tool_use_id=_string_or_none(payload.get("tool_use_id")),
                result=result,
                is_error=is_error,
                error_message=error_message,
                metadata=metadata,
            )

        if hook_event_name == "PermissionRequest":
            content = str(payload.get("tool_name") or "permission_request")
            return MessageEvent(
                session_id=session_id,
                source_sdk=_SOURCE,
                role="system",
                content=content,
                metadata=metadata,
            )

        if hook_event_name == "Stop":
            return SessionEndEvent(
                session_id=session_id,
                source_sdk=_SOURCE,
                status="completed",
                metadata=metadata,
            )

        return None


CodexAdapter = CodexHooksAdapter


def build_hooks_config(command: str, *, timeout: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Build a Codex hooks config using *command* for every supported hook."""
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
        elif hook_name in {"PreToolUse", "PostToolUse"}:
            entry["matcher"] = "*"
        config[hook_name] = [entry]
    return config


def load_payload(stream: Any | None = None) -> dict[str, Any]:
    """Read one Codex hook payload from a text stream."""
    if stream is None:
        stream = sys.stdin
    raw = stream.read()
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        msg = "Codex hook payload must be a JSON object"
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
    """Return hook JSON response, when Codex expects one."""
    if payload.get("hook_event_name") == "Stop":
        return {"continue": True}
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge OpenAI Codex hooks to agent-context-graph.")
    parser.add_argument(
        "--connector",
        action="append",
        default=None,
        help="Graph connector to enable. Currently supported: skills-graph.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Override the session id from the Codex hook payload.",
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
        adapter = CodexHooksAdapter(link, session_id=args.session_id)
        adapter.handle_payload(payload)
        response = response_for_payload(payload)
        if response is not None:
            print(json.dumps(response))
    except Exception as exc:
        if args.strict or os.environ.get("AGENT_CONTEXT_GRAPH_CODEX_STRICT") == "1":
            raise
        response = response_for_payload(payload)
        if response is not None:
            print(json.dumps(response))
        _debug_log(f"agent-context-graph Codex hook skipped: {exc}")
    return 0


def _add_skills_graph_connector(link: AgentLink) -> None:
    try:
        from skills_graph import SkillGraph
        from skills_graph.connector import SkillGraphConnector
    except ImportError as exc:
        msg = "skills-graph is required for the skills-graph Codex connector"
        raise ImportError(msg) from exc

    graph = SkillGraph()
    link.add_connector(SkillGraphConnector(graph))


def _connectors_from_env() -> list[str]:
    value = os.environ.get("AGENT_CONTEXT_GRAPH_CODEX_CONNECTORS", "")
    return [part.strip() for part in value.split(",") if part.strip()]


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "cwd",
        "source",
        "transcript_path",
        "turn_id",
        "permission_mode",
        "tool_name",
        "tool_input",
        "tool_use_id",
        "reason",
        "decision",
        "stop_hook_active",
    ):
        if key in payload and payload.get(key) is not None:
            metadata[key] = payload.get(key)
    return metadata


def _extract_tool_result(tool_response: Any) -> tuple[Any, bool, str | None]:
    if not isinstance(tool_response, dict):
        return tool_response, False, None

    is_error = bool(
        tool_response.get("is_error", False)
        or tool_response.get("error")
        or tool_response.get("exit_code") not in (None, 0)
    )
    error_message = tool_response.get("error") or tool_response.get("stderr")
    result = tool_response.get("content", tool_response)
    return result, is_error, _string_or_none(error_message)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _debug_log(message: str) -> None:
    if os.environ.get("AGENT_CONTEXT_GRAPH_CODEX_DEBUG") == "1":
        print(message, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
