"""OpenAI Codex hooks adapter for agent-context-graph.

Codex hooks are command-based: Codex invokes a configured command with the
hook payload on stdin.  This adapter translates those JSON payloads into the
common Event protocol used by AgentLink.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_context_graph.events import (
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_context_graph.hooks.runner import create_link, load_payload  # noqa: F401 (re-exported for callers)
from agent_context_graph.protocols import RuntimeAdapter
from memgraph_toolbox.api.memgraph import MEMGRAPH_ENV_KEYS, memgraph_env

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from agent_context_graph.events import Event
    from agent_context_graph.link import AgentLink

_SOURCE = "codex"
_DEFAULT_COMMAND = "agent-context-graph hook run codex"
_SUPPORTED_HOOKS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "Stop",
)


class CodexHooksAdapter(RuntimeAdapter):
    """Adapter that converts OpenAI Codex hook payloads into graph events.

    Args:
        link: The AgentLink hub to emit events to.
        session_id: Optional override for all emitted event session ids.
    """

    def __init__(self, link: AgentLink, session_id: str | None = None) -> None:
        self._link = link
        self._session_id = session_id

    def get_runtime_hooks(self) -> dict[str, list[dict[str, Any]]]:
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
                user_id=_resolve_user_id(payload),
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


def response_for_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return hook JSON response, when Codex expects one."""
    if payload.get("hook_event_name") == "Stop":
        return {"continue": True}
    return None


def init(project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
    """Generate a private Codex hook config (``.codex/config.toml`` + ``.codex/hooks.json``).

    Extracted from what was ``hooks/cli.py``'s ``_init_codex`` -- kwargs mirror
    its former CLI flags: hook_command, memgraph_url/user/password/database,
    setup_schema, timeout, force.
    """
    hook_command = kwargs.get("hook_command")
    timeout = kwargs.get("timeout", 30)
    force = kwargs.get("force", False)
    setup_schema = kwargs.get("setup_schema", False)

    codex_dir = project_dir / ".codex"
    config_path = codex_dir / "config.toml"
    hooks_path = codex_dir / "hooks.json"

    existing = [path for path in (config_path, hooks_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        msg = f"Refusing to overwrite existing Codex config: {names} (pass force=True to replace)"
        raise FileExistsError(msg)

    resolved_memgraph_env = memgraph_env(
        url=kwargs.get("memgraph_url"),
        username=kwargs.get("memgraph_user"),
        password=kwargs.get("memgraph_password"),
        database=kwargs.get("memgraph_database"),
    )

    if hook_command is None:
        executable = shutil.which("agent-context-graph")
        base = [executable] if executable else [sys.executable, "-m", "agent_context_graph.cli"]
        command_parts = [*base, "hook", "run", "codex"]
        for connector in connectors:
            command_parts.extend(["--connector", connector])
        hook_command = shlex.join(command_parts)

    codex_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("[features]\nhooks = true\n", encoding="utf-8")
    hooks_path.write_text(
        json.dumps({"hooks": build_hooks_config(hook_command, timeout=timeout)}, indent=2) + "\n",
        encoding="utf-8",
    )

    if setup_schema:
        previous = {key: os.environ.get(key) for key in MEMGRAPH_ENV_KEYS}
        os.environ.update(resolved_memgraph_env)
        try:
            for connector in connectors:
                normalized = connector.strip().replace("-", "_")
                if normalized == "skills_graph":
                    from skills_graph import SkillGraph

                    SkillGraph().setup()
                elif normalized == "actions_graph":
                    from actions_graph import ActionsGraph

                    ActionsGraph().setup()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    print(f"Wrote {config_path}")
    print(f"Wrote {hooks_path}")
    print(f"Memgraph URL: {resolved_memgraph_env['MEMGRAPH_URL']}")
    print(f"Memgraph database: {resolved_memgraph_env['MEMGRAPH_DATABASE']}")
    secret = resolved_memgraph_env["MEMGRAPH_PASSWORD"]
    masked = hook_command.replace(shlex.quote(secret), "'****'").replace(secret, "****") if secret else hook_command
    print(f"Hook command: {masked}")


@dataclass(frozen=True)
class _CodexPlugin:
    """Registered under the ``agent_context_graph.runtimes`` entry point as ``PLUGIN``."""

    name: str = "codex"
    adapter_class: type[RuntimeAdapter] = CodexHooksAdapter

    def response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        return build_hooks_config(command, timeout=timeout)

    def init(self, project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
        init(project_dir, connectors, **kwargs)


PLUGIN = _CodexPlugin()


def main(argv: Sequence[str] | None = None) -> int:
    from agent_context_graph.hooks.runner import run_hook

    return run_hook(PLUGIN, argv)


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


def _resolve_user_id(payload: dict[str, Any]) -> str | None:
    """Resolve a stable user identity for SessionStartEvent.

    Resolution order:
    1. ``user_id`` field in the hook payload (forward-compat).
    2. ``AGENT_CONTEXT_GRAPH_USER_ID`` environment variable.
    3. Config file at ``~/.config/agent-context-graph/config.toml``.
    """
    from agent_context_graph.adapters._identity import resolve_user_id

    return resolve_user_id(payload)


if __name__ == "__main__":
    raise SystemExit(main())
