"""Grok Build (``grok``) command-hook runtime adapter.

Grok sends each payload field in both camelCase and Claude Code's snake_case,
so the shared field aliases resolve most of it; only its typed tool results
and per-turn ``Stop`` need Grok-specific rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    FieldMap,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    SpecPlugin,
    json_hook_installer,
    session_start,
    tool_start,
    turn_end,
)
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
    return MessageEvent(**context.base(), role="user", content=context.value("prompt", ""))


def _tool_result(tool_response: Any) -> tuple[Any, bool]:
    """Return a typed Grok tool result's model-facing text and whether it failed.

    Results are tagged by ``type``: shell results carry ``output_for_prompt``
    and ``exit_code``, file reads carry ``FileContent.content``. Other types
    are kept whole.
    """
    if not isinstance(tool_response, dict):
        return tool_response, False
    exit_code = tool_response.get("exit_code")
    text = tool_response.get("output_for_prompt")
    if text is None:
        file_content = tool_response.get("FileContent")
        text = file_content.get("content") if isinstance(file_content, dict) else None
    return (tool_response if text is None else text), exit_code not in (None, 0)


def _tool_end(context: EventContext) -> ToolEndEvent:
    result, is_error = _tool_result(context.value("tool_result"))
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=result,
        is_error=is_error,
        agent_name=context.optional_text("agent_id"),
    )


def _tool_failure(context: EventContext) -> ToolEndEvent:
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        is_error=True,
        error_message=str(context.payload.get("error") or "Grok tool failed"),
        agent_name=context.optional_text("agent_id"),
    )


def _permission_denied(context: EventContext) -> ErrorOccurredEvent:
    return ErrorOccurredEvent(
        **context.base(),
        error_type="permission_denied",
        error_message=str(context.payload.get("reason") or context.text("tool_name", "Permission denied")),
        recoverable=True,
    )


def _turn_end(context: EventContext) -> list[Event]:
    # Stop fires at the end of every turn (reason "end_turn") and once more at
    # shutdown, right after SessionEnd; only the former ends a turn.
    reason = context.payload.get("reason")
    if reason == "shutdown":
        return []
    reply = context.payload.get("lastAssistantMessage", context.payload.get("last_assistant_message"))
    return turn_end(context, reply=reply, reason=reason)


def _stop_failure(context: EventContext) -> ErrorOccurredEvent:
    error = str(context.payload.get("error") or "unknown")
    return ErrorOccurredEvent(
        **context.base(),
        error_type=error,
        error_message=str(context.payload.get("reason") or error),
        recoverable=True,
    )


def _session_end(context: EventContext) -> SessionEndEvent:
    return SessionEndEvent(**context.base(), status="completed")


def _agent_start(context: EventContext) -> AgentStartEvent:
    agent_type = context.text("agent_type")
    return AgentStartEvent(**context.base(), agent_name=context.text("agent_id", agent_type), agent_type=agent_type)


def _agent_end(context: EventContext) -> AgentEndEvent:
    agent_type = context.text("agent_type")
    return AgentEndEvent(
        **context.base(),
        agent_name=context.text("agent_id", agent_type),
        agent_type=agent_type,
        output=context.payload.get("lastAssistantMessage", context.payload.get("last_assistant_message")),
    )


_HOOKS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "PermissionDenied",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "StopFailure",
    "SessionEnd",
)
SPEC = RuntimeSpec(
    name="grok",
    source_sdk="grok",
    event_key="hook_event_name",
    hooks=_HOOKS,
    rules={
        "SessionStart": session_start,
        "UserPromptSubmit": _user_prompt,
        "PreToolUse": tool_start,
        "PostToolUse": _tool_end,
        "PostToolUseFailure": _tool_failure,
        "PermissionDenied": _permission_denied,
        "SubagentStart": _agent_start,
        "SubagentStop": _agent_end,
        "Stop": _turn_end,
        "StopFailure": _stop_failure,
        "SessionEnd": _session_end,
    },
    metadata_keys=(
        "cwd",
        "workspaceRoot",
        "permissionMode",
        "source",
        "reason",
        "promptId",
        "transcriptPath",
        "durationMs",
        "isBackgrounded",
        "toolInputTruncated",
        "toolResultTruncated",
        "stopHookActive",
    ),
    config=HookConfig(path=".grok/hooks/agent-context-graph.json", layout="nested", merge=True),
    fields=FieldMap(tool_result=("tool_response", "toolResult")),
    probe_payload={"hook_event_name": "SessionEnd", "session_id": "doctor", "reason": "shutdown"},
)


class GrokHooksAdapter(SpecAdapter):
    """Convert Grok Build hook payloads into Event Protocol events."""

    SPEC = SPEC


init = json_hook_installer(SPEC)
PLUGIN = SpecPlugin(SPEC, GrokHooksAdapter, init=init)
