"""Cursor command-hook runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    FieldMap,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    event_user_id,
    write_hook_config,
)
from agent_context_graph.adapters._spec import (
    build_hooks_config as build_spec_hooks_config,
)
from agent_context_graph.events import (
    AgentEndEvent,
    AgentStartEvent,
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
    workspace_roots = context.payload.get("workspace_roots")
    cwd = workspace_roots[0] if isinstance(workspace_roots, list) and workspace_roots else context.value("cwd")
    return SessionStartEvent(
        **context.base(),
        model=context.optional_text("model"),
        working_directory=None if cwd is None else str(cwd),
        user_id=event_user_id(context),
    )


def _session_end(context: EventContext) -> SessionEndEvent:
    return SessionEndEvent(**context.base(), status=str(context.payload.get("reason") or "completed"))


def _user_message(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="user",
        content=context.value("prompt", ""),
        model=context.optional_text("model"),
    )


def _assistant_message(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="assistant",
        content=context.payload.get("text", ""),
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
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=context.value("tool_result"),
    )


def _tool_failure(context: EventContext) -> ToolEndEvent:
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        is_error=True,
        error_message=str(context.payload.get("error_message") or "Cursor tool failed"),
    )


def _agent_start(context: EventContext) -> AgentStartEvent:
    agent_type = context.text("agent_type")
    # Cursor's stop payload omits subagent_id. Use subagent_type as the common
    # lifecycle key and preserve the start-only ID in metadata.
    return AgentStartEvent(
        **context.base(),
        agent_name=agent_type,
        agent_type=agent_type,
    )


def _agent_end(context: EventContext) -> AgentEndEvent:
    agent_type = context.text("agent_type")
    return AgentEndEvent(
        **context.base(),
        agent_name=agent_type,
        agent_type=agent_type,
        output=str(context.payload.get("summary")) if context.payload.get("summary") is not None else None,
    )


SPEC = RuntimeSpec(
    name="cursor",
    source_sdk="cursor",
    event_key="hook_event_name",
    hooks=(
        "sessionStart",
        "sessionEnd",
        "beforeSubmitPrompt",
        "preToolUse",
        "postToolUse",
        "postToolUseFailure",
        "subagentStart",
        "subagentStop",
        "afterAgentResponse",
    ),
    rules={
        "sessionStart": _session_start,
        "sessionEnd": _session_end,
        "beforeSubmitPrompt": _user_message,
        "preToolUse": _tool_start,
        "postToolUse": _tool_end,
        "postToolUseFailure": _tool_failure,
        "subagentStart": _agent_start,
        "subagentStop": _agent_end,
        "afterAgentResponse": _assistant_message,
    },
    metadata_keys=(
        "generation_id",
        "model_id",
        "model_params",
        "cursor_version",
        "workspace_roots",
        "user_email",
        "transcript_path",
        "tool_name",
        "tool_input",
        "tool_use_id",
        "duration",
        "failure_type",
        "is_interrupt",
        "task",
        "subagent_id",
        "subagent_type",
        "description",
        "status",
        "duration_ms",
        "agent_transcript_path",
    ),
    config=HookConfig(
        path=".cursor/hooks.json",
        layout="flat",
        version=1,
        merge=True,
        timeout_key="timeout",
        matchers={
            "preToolUse": "*",
            "postToolUse": "*",
            "postToolUseFailure": "*",
            "subagentStart": "*",
            "subagentStop": "*",
        },
    ),
    fields=FieldMap(session_id=("conversation_id", "session_id")),
    probe_payload={"hook_event_name": "sessionEnd", "session_id": "doctor", "reason": "completed"},
)


class CursorHooksAdapter(SpecAdapter):
    """Convert Cursor hook payloads into Event Protocol events."""

    SPEC = SPEC


CursorAdapter = CursorHooksAdapter


def build_hooks_config(command: str, *, timeout: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Build a Cursor hooks map using *command*."""
    return build_spec_hooks_config(SPEC, command, timeout=timeout)


def response_for_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return non-blocking responses for Cursor's permission hooks."""
    event_name = payload.get("hook_event_name")
    if event_name in {"preToolUse", "subagentStart"}:
        return {"permission": "allow"}
    if event_name == "beforeSubmitPrompt":
        return {"continue": True}
    return None


def init(project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
    """Write ``.cursor/hooks.json``."""
    path = write_hook_config(
        SPEC,
        project_dir,
        connectors,
        hook_command=kwargs.get("hook_command"),
        timeout=kwargs.get("timeout", 30),
        force=kwargs.get("force", False),
    )
    print(f"Wrote {path}")


@dataclass(frozen=True)
class _CursorPlugin:
    """Cursor runtime registration."""

    name: str = "cursor"
    adapter_class: type[RuntimeAdapter] = CursorHooksAdapter
    probe_payload: Mapping[str, Any] = field(default_factory=lambda: SPEC.probe_payload)

    def response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        return build_hooks_config(command, timeout=timeout)

    def init(self, project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
        init(project_dir, connectors, **kwargs)


PLUGIN = _CursorPlugin()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Cursor hook command."""
    from agent_context_graph.hooks.runner import run_hook

    return run_hook(PLUGIN, argv)


if __name__ == "__main__":
    raise SystemExit(main())
