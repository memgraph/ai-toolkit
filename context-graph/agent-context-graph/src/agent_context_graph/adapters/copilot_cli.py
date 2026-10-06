"""GitHub Copilot CLI command-hook runtime adapter."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    SpecPlugin,
    extract_tool_result,
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
    ToolStartEvent,
)

if TYPE_CHECKING:
    from agent_context_graph.events import Event


def _session_end(context: EventContext) -> SessionEndEvent:
    reason = str(
        context.payload.get("reason")
        or context.payload.get("stop_reason")
        or context.payload.get("stopReason")
        or "completed"
    )
    return SessionEndEvent(**context.base(), status="completed" if reason in {"complete", "end_turn"} else reason)


def _message(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="user", content=context.value("prompt", ""))


def _tool_input(context: EventContext) -> Any:
    # Native Copilot payloads may carry toolArgs as a JSON-encoded string.
    tool_input = context.value("tool_input")
    if isinstance(tool_input, str):
        try:
            return json.loads(tool_input)
        except ValueError:
            return tool_input
    return tool_input


def _tool_start(context: EventContext) -> ToolStartEvent:
    return dataclasses.replace(tool_start(context), tool_input=_tool_input(context))


def _tool_end(context: EventContext) -> ToolEndEvent:
    tool_result = context.value("tool_result")
    result, is_error, error_message = extract_tool_result(tool_result)
    result_type = (
        tool_result.get("resultType", tool_result.get("result_type")) if isinstance(tool_result, dict) else None
    )
    if result_type not in (None, "success"):
        is_error = True
        error_message = error_message or f"Copilot tool result: {result_type}"
    return _with_tool_input(
        context,
        ToolEndEvent(
            **context.base(),
            tool_name=context.text("tool_name"),
            tool_use_id=context.optional_text("tool_use_id"),
            result=result,
            is_error=is_error,
            error_message=error_message,
        ),
    )


def _tool_failure(context: EventContext) -> ToolEndEvent:
    return _with_tool_input(
        context,
        ToolEndEvent(
            **context.base(),
            tool_name=context.text("tool_name"),
            tool_use_id=context.optional_text("tool_use_id"),
            is_error=True,
            error_message=str(context.payload.get("error") or "Tool failed"),
        ),
    )


def _with_tool_input(context: EventContext, event: ToolEndEvent) -> ToolEndEvent:
    # Copilot sends no tool call id, so keep the input on the result to tell
    # repeated calls of the same tool apart.
    tool_input = _tool_input(context)
    if tool_input is not None:
        event.metadata["tool_input"] = tool_input
    return event


def _permission(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="system", content=context.text("tool_name", "permission_request"))


def _agent_name(context: EventContext) -> str:
    # subagentStart carries only agentName; subagentStop adds agentId and
    # agentType. agentName is the one key both share, so it names the Agent.
    return str(context.payload.get("agentName") or context.payload.get("agent_name") or context.text("agent_type"))


def _agent_start(context: EventContext) -> AgentStartEvent:
    agent_name = _agent_name(context)
    return AgentStartEvent(**context.base(), agent_name=agent_name, agent_type=agent_name)


def _agent_end(context: EventContext) -> AgentEndEvent:
    agent_name = _agent_name(context)
    output = context.payload.get("last_assistant_message", context.payload.get("response"))
    return AgentEndEvent(
        **context.base(),
        agent_name=agent_name,
        agent_type=context.text("agent_type", agent_name),
        output=None if output is None else str(output),
    )


def _turn_end(context: EventContext) -> list[Event]:
    # agentStop ends one turn; sessionEnd ends the session. Its payload carries
    # no reply text.
    return turn_end(context, reason=context.payload.get("stopReason", context.payload.get("stop_reason")))


def _error(context: EventContext) -> ErrorOccurredEvent:
    error = context.payload.get("error")
    error_dict = error if isinstance(error, dict) else {}
    message = error_dict.get("message", error)
    details = {"context": context.payload.get("error_context", context.payload.get("errorContext"))}
    if error_dict.get("stack") is not None:
        details["stack"] = error_dict["stack"]
    return ErrorOccurredEvent(
        **context.base(),
        error_type=str(error_dict.get("name") or "copilot_error"),
        error_message=str(message or "Copilot CLI error"),
        error_details=details,
        recoverable=bool(context.payload.get("recoverable", True)),
    )


_HOOKS = (
    "sessionStart",
    "sessionEnd",
    "userPromptSubmitted",
    "preToolUse",
    "postToolUse",
    "postToolUseFailure",
    "permissionRequest",
    "subagentStart",
    "subagentStop",
    "agentStop",
    "errorOccurred",
)
SPEC = RuntimeSpec(
    name="copilot-cli",
    source_sdk="copilot-cli",
    event_key="hook_event_name",
    hooks=_HOOKS,
    rules={
        "sessionStart": session_start,
        "sessionEnd": _session_end,
        "userPromptSubmitted": _message,
        "preToolUse": _tool_start,
        "postToolUse": _tool_end,
        "postToolUseFailure": _tool_failure,
        "permissionRequest": _permission,
        "subagentStart": _agent_start,
        "subagentStop": _agent_end,
        "agentStop": _turn_end,
        "errorOccurred": _error,
    },
    # VS Code-compatible hook files use PascalCase event names.
    event_aliases={
        **{hook[0].upper() + hook[1:]: hook for hook in _HOOKS},
        "UserPromptSubmit": "userPromptSubmitted",
    },
    metadata_keys=(
        "cwd",
        "timestamp",
        "source",
        "transcript_path",
        "reason",
        "stop_reason",
        "tool_name",
        "tool_input",
        "agent_id",
        "agent_type",
        "agent_name",
        "agent_display_name",
        "agent_description",
        "agentName",
        "agentDisplayName",
        "agentDescription",
        "error_context",
    ),
    config=HookConfig(
        path=".github/hooks/agent-context-graph.json",
        layout="flat",
        version=1,
        merge=True,
        timeout_key="timeoutSec",
        inject_event_name=True,
        matchers={
            "preToolUse": ".*",
            "postToolUse": ".*",
            "permissionRequest": ".*",
            "subagentStart": ".*",
        },
    ),
    probe_payload={"hook_event_name": "sessionEnd", "sessionId": "doctor", "reason": "complete"},
)


class CopilotCLIHooksAdapter(SpecAdapter):
    """Convert Copilot CLI hook payloads into Event Protocol events."""

    SPEC = SPEC


init = json_hook_installer(SPEC)
PLUGIN = SpecPlugin(SPEC, CopilotCLIHooksAdapter, init=init)
