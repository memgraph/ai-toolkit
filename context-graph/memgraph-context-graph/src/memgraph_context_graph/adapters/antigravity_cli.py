"""Antigravity CLI (``agy``) command-hook runtime adapter.

Antigravity hooks expose less than other runtimes: the session's start, tool
calls and their errors, and execution stops. They carry no tool results,
prompts, assistant text, or token usage. The turn's words come from the
transcript the ``Stop`` payload points at instead: when a stop leaves the agent
idle, the last user request and the reply that closed it are read from there.
Its ``SessionStart`` output is the one place a hook can put text in front of
the model for the whole session (``injectSteps``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memgraph_context_graph.adapters._spec import (
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
from memgraph_context_graph.adapters.agent_memory import NO_BUILT_IN_MEMORY
from memgraph_context_graph.events import (
    ErrorOccurredEvent,
    MessageEvent,
    SessionStartEvent,
    ToolEndEvent,
    ToolStartEvent,
)

if TYPE_CHECKING:
    from memgraph_context_graph.events import Event


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


def _session_start(context: EventContext) -> SessionStartEvent:
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


_USER_REQUEST_RE = re.compile(r"<USER_REQUEST>\n?(?P<text>.*?)\n?</USER_REQUEST>", re.DOTALL)


def _last_turn(context: EventContext) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    """The transcript's last user request and the final model reply after it, or None when unreadable."""
    path = context.payload.get("transcriptPath")
    if not isinstance(path, str) or not path:
        return None
    try:
        steps = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return None
    starts = [i for i, step in enumerate(steps) if step.get("type") == "USER_INPUT"]
    if not starts:
        return None
    request = steps[starts[-1]]
    replies = [
        step for step in steps[starts[-1] + 1 :] if step.get("type") == "PLANNER_RESPONSE" and step.get("content")
    ]
    return request, (replies[-1] if replies else None)


def _at(step: dict[str, Any]) -> str | None:
    """The step's time, with its index as microseconds: second-resolution times would collide within a turn,
    and the same step must give the same timestamp (and so the same action id) every time it is read."""
    created = step.get("created_at")
    if not isinstance(created, str):
        return None
    index = int(step.get("step_index") or 0) % 1_000_000
    return created.replace("Z", "") + f".{index:06d}+00:00"


def _turn_messages(context: EventContext) -> tuple[list[Event], str | None]:
    """The turn's user message as an event, and the reply's text for :func:`turn_end`."""
    turn = _last_turn(context)
    if turn is None:
        return [], None
    request, reply = turn
    match = _USER_REQUEST_RE.search(str(request.get("content") or ""))
    events: list[Event] = []
    if match and match["text"].strip():
        base = context.base()
        if at := _at(request):
            base["timestamp"] = at
        events.append(MessageEvent(**base, role="user", content=match["text"].strip()))
    return events, (str(reply["content"]) if reply else None)


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
        messages, reply = _turn_messages(context)
        events.extend(messages)
        events.extend(turn_end(context, reply=reply, reason=reason))
    return events


_HOOKS = ("SessionStart", "PreToolUse", "PostToolUse", "Stop")
SPEC = RuntimeSpec(
    name="antigravity-cli",
    source_sdk="antigravity-cli",
    event_key="hook_event_name",
    hooks=_HOOKS,
    rules={
        "SessionStart": _session_start,
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
        root_key="memgraph-context-graph",
        merge=True,
        inject_event_name=True,
        flat_hooks=frozenset({"SessionStart", "Stop"}),
    ),
    fields=FieldMap(session_id=("conversationId",)),
    # PostToolUse requires an object; an empty PreToolUse answer leaves agy's
    # own permission prompt in charge, so capture never approves a tool.
    responses={"PostToolUse": {}},
    probe_payload={"hook_event_name": "SessionStart", "conversationId": "doctor"},
    # Kept in the conversation as a system message for every later turn.
    session_context=lambda text: {"injectSteps": [{"systemMessage": {"systemMessage": text}}]},
)


class AntigravityCLIHooksAdapter(SpecAdapter):
    """Convert Antigravity CLI hook payloads into Event Protocol events."""

    SPEC = SPEC


init = json_hook_installer(SPEC)
PLUGIN = SpecPlugin(SPEC, AntigravityCLIHooksAdapter, init=init, native_memory=NO_BUILT_IN_MEMORY)
