"""Gemini CLI command-hook runtime adapter."""

from __future__ import annotations

from typing import Any

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    SpecPlugin,
    json_hook_installer,
    session_start,
    string_or_none,
    tool_end,
    tool_start,
)
from agent_context_graph.events import LLMEndEvent, MessageEvent, SessionEndEvent


def _session_end(context: EventContext) -> SessionEndEvent:
    reason = context.payload.get("reason")
    status = "completed" if reason in {None, "exit", "clear", "logout", "prompt_input_exit"} else str(reason)
    return SessionEndEvent(**context.base(), status=status)


def _user_message(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="user", content=context.value("prompt", ""))


def _assistant_message(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="assistant", content=context.payload.get("prompt_response", ""))


def _llm_end(context: EventContext) -> LLMEndEvent | None:
    response = context.payload.get("llm_response")
    response_dict = response if isinstance(response, dict) else {}
    candidates = response_dict.get("candidates")
    # AfterModel fires for every streamed chunk; only the chunk carrying a
    # finishReason ends the model call.
    if not isinstance(candidates, list) or not any(
        isinstance(candidate, dict) and candidate.get("finishReason") for candidate in candidates
    ):
        return None
    request = context.payload.get("llm_request")
    request_dict = request if isinstance(request, dict) else {}
    usage = response_dict.get("usageMetadata")
    usage = usage if isinstance(usage, dict) else {}
    return LLMEndEvent(
        **context.base(),
        model=string_or_none(request_dict.get("model")),
        input_tokens=_integer_or_none(usage.get("promptTokenCount")),
        output_tokens=_integer_or_none(usage.get("candidatesTokenCount")),
        response=response,
    )


def _notification(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="system",
        content=context.payload.get("message", context.payload.get("details", "")),
    )


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


SPEC = RuntimeSpec(
    name="gemini-cli",
    source_sdk="gemini-cli",
    event_key="hook_event_name",
    hooks=(
        "SessionStart",
        "SessionEnd",
        "BeforeAgent",
        "AfterAgent",
        "BeforeTool",
        "AfterTool",
        "AfterModel",
        "Notification",
    ),
    rules={
        "SessionStart": session_start,
        "SessionEnd": _session_end,
        "BeforeAgent": _user_message,
        "AfterAgent": _assistant_message,
        "BeforeTool": tool_start,
        "AfterTool": tool_end,
        "AfterModel": _llm_end,
        "Notification": _notification,
    },
    metadata_keys=(
        "cwd",
        "transcript_path",
        "timestamp",
        "source",
        "reason",
        "tool_name",
        "tool_input",
        "mcp_context",
        "original_request_name",
        "stop_hook_active",
        "notification_type",
    ),
    config=HookConfig(
        path=".gemini/settings.json",
        layout="nested",
        merge=True,
        timeout_multiplier=1000,
        matchers={"BeforeTool": ".*", "AfterTool": ".*"},
    ),
    probe_payload={"hook_event_name": "SessionEnd", "session_id": "doctor", "reason": "exit"},
)


class GeminiCLIHooksAdapter(SpecAdapter):
    """Convert Gemini CLI hook payloads into Event Protocol events."""

    SPEC = SPEC


init = json_hook_installer(SPEC)
PLUGIN = SpecPlugin(SPEC, GeminiCLIHooksAdapter, init=init)
