"""OpenCode V2 plugin runtime adapter and installer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    event_user_id,
    hook_command_for,
)
from agent_context_graph.events import (
    ErrorOccurredEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from agent_context_graph.protocols import RuntimeAdapter


def _session_start(context: EventContext) -> SessionStartEvent:
    return SessionStartEvent(
        **context.base(),
        model=context.optional_text("model"),
        working_directory=context.optional_text("cwd"),
        user_id=event_user_id(context),
    )


def _session_end(context: EventContext) -> SessionEndEvent:
    return SessionEndEvent(**context.base(), status="completed")


def _prompt(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="user", content=context.value("prompt", ""))


def _message(context: EventContext) -> MessageEvent | None:
    role = context.payload.get("role")
    content = context.payload.get("content")
    # Prompt admission already records user messages, so only observe the
    # assistant side of message updates to avoid duplicate actions.
    if role != "assistant" or content is None:
        return None
    return MessageEvent(
        **context.base(),
        role=str(role),
        content=content,
        model=context.optional_text("model"),
    )


def _tool_start(context: EventContext) -> ToolStartEvent:
    return ToolStartEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_input=context.value("tool_input"),
        tool_use_id=context.optional_text("tool_use_id"),
    )


def _tool_end(context: EventContext) -> ToolEndEvent:
    error = context.payload.get("error")
    error_dict = error if isinstance(error, dict) else {}
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=context.value("tool_result"),
        is_error=bool(context.payload.get("is_error")),
        error_message=str(error_dict.get("message", error)) if error else None,
    )


def _error(context: EventContext) -> ErrorOccurredEvent:
    error = context.payload.get("error")
    error_dict = error if isinstance(error, dict) else {}
    return ErrorOccurredEvent(
        **context.base(),
        error_type=str(error_dict.get("name") or "opencode_error"),
        error_message=str(error_dict.get("message") or error or "OpenCode session error"),
        error_details={"event": context.payload.get("event")},
        recoverable=True,
    )


def _permission(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="system",
        content=context.payload.get("permission") or "permission_requested",
    )


SPEC = RuntimeSpec(
    name="opencode",
    source_sdk="opencode",
    event_key="hook_event_name",
    hooks=(),
    rules={
        "session.created": _session_start,
        "session.deleted": _session_end,
        "session.prompt": _prompt,
        "message.updated": _message,
        "tool.execute.before": _tool_start,
        "tool.execute.after": _tool_end,
        "session.error": _error,
        "permission.asked": _permission,
    },
    metadata_keys=("cwd", "model", "metadata"),
    config=HookConfig(path=".opencode/plugins/agent-context-graph/index.js", layout="flat"),
    probe_payload={"hook_event_name": "session.deleted", "session_id": "doctor"},
)


class OpenCodeHooksAdapter(SpecAdapter):
    """Convert normalized OpenCode V2 plugin callbacks into events."""

    SPEC = SPEC


OpenCodeAdapter = OpenCodeHooksAdapter


def build_hooks_config(command: str, *, timeout: int = 30) -> dict[str, Any]:
    """Return an empty map because OpenCode uses a V2 plugin, not JSON hooks."""
    return {}


def response_for_payload(payload: dict[str, Any]) -> None:
    """OpenCode callbacks are observational and need no hook response."""
    return None


def init(project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
    """Install the dependency-free OpenCode V2 capture plugin."""
    target = project_dir / SPEC.config.path
    if target.exists() and not kwargs.get("force", False):
        raise FileExistsError(f"Refusing to overwrite existing OpenCode plugin: {target} (pass force=True)")

    command = kwargs.get("hook_command") or hook_command_for("opencode", connectors)
    template = (
        resources.files("agent_context_graph.adapters").joinpath("_opencode_plugin.js").read_text(encoding="utf-8")
    )
    plugin = template.replace("__AGENT_CONTEXT_GRAPH_COMMAND__", json.dumps(command))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plugin, encoding="utf-8")
    print(f"Wrote {target}")


@dataclass(frozen=True)
class _OpenCodePlugin:
    """OpenCode runtime registration."""

    name: str = "opencode"
    adapter_class: type[RuntimeAdapter] = OpenCodeHooksAdapter
    probe_payload: Mapping[str, Any] = field(default_factory=lambda: SPEC.probe_payload)

    def response_for_payload(self, payload: dict[str, Any]) -> None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        return build_hooks_config(command, timeout=timeout)

    def init(self, project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
        init(project_dir, connectors, **kwargs)


PLUGIN = _OpenCodePlugin()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the OpenCode capture hook command."""
    from agent_context_graph.hooks.runner import run_hook

    return run_hook(PLUGIN, argv)


if __name__ == "__main__":
    raise SystemExit(main())
