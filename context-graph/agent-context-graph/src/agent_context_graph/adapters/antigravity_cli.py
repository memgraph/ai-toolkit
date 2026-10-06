"""Antigravity CLI (``agy``) command-hook runtime adapter.

Antigravity hooks expose less than other runtimes: tool calls and their
errors, model-invocation counters, and execution stops. They carry no tool
results, prompts, assistant text, or token usage, so an Antigravity session
records what ran but not what came back.
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
    event_user_id,
    json_hook_installer,
    string_or_none,
    turn_end,
)
from agent_context_graph.events import ErrorOccurredEvent, SessionStartEvent, ToolEndEvent, ToolStartEvent

if TYPE_CHECKING:
    from agent_context_graph.events import Event


def _workspace(context: EventContext) -> str | None:
    paths = context.payload.get("workspacePaths")
    return str(paths[0]) if isinstance(paths, list) and paths else None


def _tool_call(context: EventContext) -> dict[str, Any]:
    tool_call = context.payload.get("toolCall")
    return tool_call if isinstance(tool_call, dict) else {}


def _step_id(context: EventContext) -> str | None:
    # No tool-call id exists; the trajectory step index is shared by a call's
    # PreToolUse and PostToolUse, so it pairs them within one conversation.
    step = context.payload.get("stepIdx")
    return None if step is None else f"step-{step}"


def _session_start(context: EventContext) -> SessionStartEvent | None:
    # No session-start hook exists. Each turn's first model invocation
    # (invocationNum resets per turn) stands in for it; connectors ignore a
    # start for a session they already hold.
    if context.payload.get("invocationNum") != 0:
        return None
    return SessionStartEvent(
        **context.base(),
        model=string_or_none(context.payload.get("modelName")),
        working_directory=_workspace(context),
        user_id=event_user_id(context),
    )


def _tool_start(context: EventContext) -> ToolStartEvent:
    tool_call = _tool_call(context)
    return ToolStartEvent(
        **context.base(),
        tool_name=str(tool_call.get("name") or ""),
        tool_input=tool_call.get("args"),
        tool_use_id=_step_id(context),
    )


def _tool_end(context: EventContext) -> ToolEndEvent:
    error = string_or_none(context.payload.get("error")) or None
    return ToolEndEvent(
        **context.base(),
        tool_name=str(_tool_call(context).get("name") or ""),
        tool_use_id=_step_id(context),
        is_error=error is not None,
        error_message=error,
    )


def _stop(context: EventContext) -> list[Event]:
    # Stop ends one execution loop. agy has no session-end hook, so a loop that
    # leaves the agent fully idle is the turn end; a failed one is also an error.
    events: list[Event] = []
    error = context.payload.get("error")
    reason = context.payload.get("terminationReason")
    if error:
        events.append(
            ErrorOccurredEvent(
                **context.base(),
                error_type=str(reason or "antigravity_error"),
                error_message=str(error),
                recoverable=True,
            )
        )
    if context.payload.get("fullyIdle"):
        events.extend(turn_end(context, reason=reason))
    return events


_HOOKS = ("PreInvocation", "PreToolUse", "PostToolUse", "Stop")
SPEC = RuntimeSpec(
    name="antigravity-cli",
    source_sdk="antigravity-cli",
    event_key="hook_event_name",
    hooks=_HOOKS,
    rules={
        "PreInvocation": _session_start,
        "PreToolUse": _tool_start,
        "PostToolUse": _tool_end,
        "Stop": _stop,
    },
    metadata_keys=(
        "workspacePaths",
        "transcriptPath",
        "artifactDirectoryPath",
        "modelName",
        "stepIdx",
        "invocationNum",
        "executionNum",
        "terminationReason",
        "fullyIdle",
    ),
    # hooks.json is keyed by named hook groups; this runtime owns one group.
    # Only tool events take matcher groups: agy rejects the whole file if a
    # lifecycle event is nested, so those are written as flat handlers.
    config=HookConfig(
        path=".agents/hooks.json",
        layout="nested",
        root_key="agent-context-graph",
        merge=True,
        inject_event_name=True,
        flat_hooks=frozenset({"PreInvocation", "Stop"}),
    ),
    fields=FieldMap(session_id=("conversationId",)),
    # PostToolUse requires an object; an empty PreToolUse answer leaves agy's
    # own permission prompt in charge, so capture never approves a tool.
    responses={"PostToolUse": {}},
    probe_payload={"hook_event_name": "PreInvocation", "conversationId": "doctor", "invocationNum": 0},
)


class AntigravityCLIHooksAdapter(SpecAdapter):
    """Convert Antigravity CLI hook payloads into Event Protocol events."""

    SPEC = SPEC


init = json_hook_installer(SPEC)
PLUGIN = SpecPlugin(SPEC, AntigravityCLIHooksAdapter, init=init)
