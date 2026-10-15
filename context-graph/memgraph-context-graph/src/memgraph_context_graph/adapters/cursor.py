"""Cursor command-hook runtime adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    tool_end,
    tool_start,
    turn_end,
)
from memgraph_context_graph.adapters.agent_memory import NO_BUILT_IN_MEMORY
from memgraph_context_graph.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
    ToolEndEvent,
)

if TYPE_CHECKING:
    from memgraph_context_graph.events import Event


def _session_start(context: EventContext) -> SessionStartEvent:
    workspace_roots = context.payload.get("workspace_roots")
    cwd = workspace_roots[0] if isinstance(workspace_roots, list) and workspace_roots else context.value("cwd")
    return SessionStartEvent(
        **context.base(),
        model=context.optional_text("model"),
        working_directory=string_or_none(cwd),
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
    # Cursor's stop payload omits subagent_id, so subagent_type is the key both share;
    # the start-only id stays in metadata.
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
        output=string_or_none(context.payload.get("summary")),
    )


def _turn_end(context: EventContext) -> list[Event]:
    # stop ends one agent loop; sessionEnd ends the conversation.
    return turn_end(context, reason=context.payload.get("status"))


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
        "stop",
    ),
    rules={
        "sessionStart": _session_start,
        "sessionEnd": _session_end,
        "beforeSubmitPrompt": _user_message,
        "preToolUse": tool_start,
        "postToolUse": tool_end,
        "postToolUseFailure": _tool_failure,
        "subagentStart": _agent_start,
        "subagentStop": _agent_end,
        "afterAgentResponse": _assistant_message,
        "stop": _turn_end,
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
        matchers={
            "preToolUse": "*",
            "postToolUse": "*",
            "postToolUseFailure": "*",
            "subagentStart": "*",
            "subagentStop": "*",
        },
    ),
    fields=FieldMap(session_id=("conversation_id", "session_id")),
    # Cursor blocks a permission hook whose output doesn't match its schema, so
    # these hooks must answer. "allow" is the weakest vote: any other hook's
    # "deny" or "ask" still wins, so capture never loosens the user's policy.
    responses={
        "preToolUse": {"permission": "allow"},
        "subagentStart": {"permission": "allow"},
        "beforeSubmitPrompt": {"continue": True},
    },
    # Grok Build also runs .cursor/hooks.json hooks, with its own payloads
    # (camelCase duplicates of every field); its grok runtime records those.
    foreign_payload_keys=frozenset({"hookEventName"}),
    probe_payload={"hook_event_name": "sessionEnd", "session_id": "doctor", "reason": "completed"},
    # Cursor's docs put this in the conversation's initial system context. The IDE's agent window can drop it
    # (the hook runs alongside the composer, not before it); the CLI waits for it.
    session_context=lambda text: {"additional_context": text},
    session_context_event="sessionStart",
)


class CursorHooksAdapter(SpecAdapter):
    """Convert Cursor hook payloads into Event Protocol events."""

    SPEC = SPEC


init = json_hook_installer(SPEC)
PLUGIN = SpecPlugin(SPEC, CursorHooksAdapter, init=init, native_memory=NO_BUILT_IN_MEMORY)
