"""Claude Code command-hook runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    event_user_id,
)
from agent_context_graph.adapters._spec import (
    build_hooks_config as build_spec_hooks_config,
)
from agent_context_graph.adapters._spec import (
    response_for_payload as spec_response_for_payload,
)
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
from agent_context_graph.hooks.runner import create_link, load_payload  # noqa: F401 -- public compatibility exports

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agent_context_graph.protocols import RuntimeAdapter


def _session_start(context: EventContext) -> SessionStartEvent:
    return SessionStartEvent(
        **context.base(),
        working_directory=context.optional_text("cwd"),
        user_id=event_user_id(context),
    )


def _user_prompt(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="user",
        content=context.value("prompt", ""),
        agent_name=context.optional_text("agent_id"),
    )


def _tool_start(context: EventContext) -> ToolStartEvent:
    return ToolStartEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_input=context.value("tool_input"),
        tool_use_id=context.optional_text("tool_use_id"),
        agent_name=context.optional_text("agent_id"),
    )


def _tool_end(context: EventContext) -> ToolEndEvent:
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=context.value("tool_result"),
        agent_name=context.optional_text("agent_id"),
    )


def _tool_failure(context: EventContext) -> ToolEndEvent:
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        is_error=True,
        error_message=context.optional_text("error"),
        agent_name=context.optional_text("agent_id"),
    )


def _permission(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="system",
        content=context.text("tool_name", "permission_request"),
        agent_name=context.optional_text("agent_id"),
    )


def _permission_denied(context: EventContext) -> ErrorOccurredEvent:
    return ErrorOccurredEvent(
        **context.base(),
        error_type="permission_denied",
        error_message=str(context.payload.get("reason") or "Permission denied"),
        recoverable=True,
    )


def _agent_start(context: EventContext) -> AgentStartEvent:
    agent_type = context.text("agent_type")
    return AgentStartEvent(
        **context.base(),
        agent_name=context.text("agent_id", agent_type),
        agent_type=agent_type,
    )


def _agent_end(context: EventContext) -> AgentEndEvent:
    agent_type = context.text("agent_type")
    return AgentEndEvent(
        **context.base(),
        agent_name=context.text("agent_id", agent_type),
        agent_type=agent_type,
        output=context.payload.get("last_assistant_message"),
    )


def _session_end(context: EventContext) -> SessionEndEvent:
    return SessionEndEvent(**context.base(), status="completed")


def _stop_failure(context: EventContext) -> ErrorOccurredEvent:
    error = str(context.payload.get("error") or "unknown")
    return ErrorOccurredEvent(
        **context.base(),
        error_type=error,
        error_message=str(context.payload.get("error_details") or error or "Claude Code stop failure"),
        recoverable=True,
    )


_HOOKS = (
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
SPEC = RuntimeSpec(
    name="claude-code",
    source_sdk="claude-code",
    event_key="hook_event_name",
    hooks=_HOOKS,
    rules={
        "SessionStart": _session_start,
        "UserPromptSubmit": _user_prompt,
        "UserPromptExpansion": _user_prompt,
        "PreToolUse": _tool_start,
        "PostToolUse": _tool_end,
        "PostToolUseFailure": _tool_failure,
        "PermissionRequest": _permission,
        "PermissionDenied": _permission_denied,
        "SubagentStart": _agent_start,
        "SubagentStop": _agent_end,
        "Stop": _session_end,
        "StopFailure": _stop_failure,
    },
    metadata_keys=(
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
    ),
    config=HookConfig(
        path=".claude/settings.json",
        layout="nested",
        merge=True,
        matchers={
            hook: "*"
            for hook in ("PreToolUse", "PostToolUse", "PostToolUseFailure", "PermissionRequest", "PermissionDenied")
        },
    ),
    response_events=frozenset({"Stop", "SubagentStop"}),
    probe_payload={"hook_event_name": "Stop", "session_id": "doctor"},
)


class ClaudeCodeHooksAdapter(SpecAdapter):
    """Convert Claude Code command-hook payloads into Event Protocol events."""

    SPEC = SPEC


ClaudeCodeAdapter = ClaudeCodeHooksAdapter


def build_hooks_config(command: str, *, timeout: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Build a Claude Code hooks configuration using *command*."""
    return build_spec_hooks_config(SPEC, command, timeout=timeout)


def response_for_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return hook JSON when Claude Code expects a response."""
    return spec_response_for_payload(SPEC, payload)


@dataclass(frozen=True)
class _ClaudeCodePlugin:
    """Claude Code runtime registration."""

    name: str = "claude-code"
    adapter_class: type[RuntimeAdapter] = ClaudeCodeHooksAdapter
    probe_payload: Mapping[str, Any] = field(default_factory=lambda: SPEC.probe_payload)

    def response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        return build_hooks_config(command, timeout=timeout)


PLUGIN = _ClaudeCodePlugin()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Claude Code hook CLI."""
    from agent_context_graph.hooks.runner import run_hook

    return run_hook(PLUGIN, argv)


if __name__ == "__main__":
    raise SystemExit(main())
