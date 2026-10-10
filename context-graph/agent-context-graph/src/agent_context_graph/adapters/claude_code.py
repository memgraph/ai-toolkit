"""Claude Code command-hook runtime adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    SpecPlugin,
    pre_tool_use_context,
    session_start,
    tool_start,
    turn_end,
)
from agent_context_graph.adapters.claude_code_memory import CLAUDE_CODE_MEMORY
from agent_context_graph.events import (
    AgentEndEvent,
    AgentStartEvent,
    ErrorOccurredEvent,
    MessageEvent,
    SessionEndEvent,
    ToolEndEvent,
)

if TYPE_CHECKING:
    from agent_context_graph.events import Event


def _user_prompt(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="user",
        content=context.value("prompt", ""),
        agent_name=context.optional_text("agent_id"),
    )


def _tool_result_text(tool_response: Any) -> Any:
    """Return the text of a Claude Code tool response, or the response unchanged.

    Bash responses carry ``stdout``/``stderr``, and Read responses carry
    ``file.content``; other tools' structured responses are kept whole.
    """
    if not isinstance(tool_response, dict):
        return tool_response
    if isinstance(tool_response.get("stdout"), str):
        stderr = tool_response.get("stderr")
        return f"{tool_response['stdout']}\n{stderr}" if stderr else tool_response["stdout"]
    file = tool_response.get("file")
    if isinstance(file, dict) and isinstance(file.get("content"), str):
        return file["content"]
    return tool_response


def _tool_end(context: EventContext) -> ToolEndEvent:
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=_tool_result_text(context.value("tool_result")),
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


def _turn_end(context: EventContext) -> list[Event]:
    # Stop ends one turn; SessionEnd, registered separately, ends the session.
    return turn_end(context, reply=context.payload.get("last_assistant_message"))


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
    "SessionEnd",
)
SPEC = RuntimeSpec(
    name="claude-code",
    source_sdk="claude-code",
    event_key="hook_event_name",
    hooks=_HOOKS,
    rules={
        "SessionStart": session_start,
        "UserPromptSubmit": _user_prompt,
        "UserPromptExpansion": _user_prompt,
        "PreToolUse": tool_start,
        "PostToolUse": _tool_end,
        "PostToolUseFailure": _tool_failure,
        "PermissionRequest": _permission,
        "PermissionDenied": _permission_denied,
        "SubagentStart": _agent_start,
        "SubagentStop": _agent_end,
        "Stop": _turn_end,
        "StopFailure": _stop_failure,
        "SessionEnd": _session_end,
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
    # Installed by the Claude Code plugin's hooks.json, not a project-local file.
    config=HookConfig(
        layout="nested",
        matchers={
            hook: "*"
            for hook in ("PreToolUse", "PostToolUse", "PostToolUseFailure", "PermissionRequest", "PermissionDenied")
        },
    ),
    responses={"Stop": {"continue": True}, "SubagentStop": {"continue": True}},
    context_before_tool=pre_tool_use_context,
    # Grok Build also runs .claude/settings.json hooks, with its own payloads
    # (camelCase duplicates of every field); its grok runtime records those.
    foreign_payload_keys=frozenset({"hookEventName"}),
    probe_payload={"hook_event_name": "Stop", "session_id": "doctor"},
)


class ClaudeCodeHooksAdapter(SpecAdapter):
    """Convert Claude Code command-hook payloads into Event Protocol events."""

    SPEC = SPEC


PLUGIN = SpecPlugin(SPEC, ClaudeCodeHooksAdapter, native_memory=CLAUDE_CODE_MEMORY)
