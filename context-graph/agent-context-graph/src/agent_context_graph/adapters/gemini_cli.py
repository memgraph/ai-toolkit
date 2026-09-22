"""Gemini CLI command-hook runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    event_user_id,
    extract_tool_result,
    write_hook_config,
)
from agent_context_graph.adapters._spec import (
    build_hooks_config as build_spec_hooks_config,
)
from agent_context_graph.events import (
    LLMEndEvent,
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
        working_directory=context.optional_text("cwd"),
        user_id=event_user_id(context),
    )


def _session_end(context: EventContext) -> SessionEndEvent:
    reason = context.payload.get("reason")
    status = "completed" if reason in {None, "exit", "clear", "logout", "prompt_input_exit"} else str(reason)
    return SessionEndEvent(**context.base(), status=status)


def _user_message(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="user", content=context.value("prompt", ""))


def _assistant_message(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="assistant", content=context.payload.get("prompt_response", ""))


def _tool_start(context: EventContext) -> ToolStartEvent:
    return ToolStartEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_input=context.value("tool_input"),
        tool_use_id=context.optional_text("tool_use_id"),
    )


def _tool_end(context: EventContext) -> ToolEndEvent:
    result, is_error, error_message = extract_tool_result(context.value("tool_result"))
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=result,
        is_error=is_error,
        error_message=error_message,
    )


def _llm_end(context: EventContext) -> LLMEndEvent:
    response = context.payload.get("llm_response")
    response_dict = response if isinstance(response, dict) else {}
    usage = response_dict.get("usageMetadata", {})
    usage = usage if isinstance(usage, dict) else {}
    return LLMEndEvent(
        **context.base(),
        model=str(response_dict.get("modelVersion")) if response_dict.get("modelVersion") is not None else None,
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
        "SessionStart": _session_start,
        "SessionEnd": _session_end,
        "BeforeAgent": _user_message,
        "AfterAgent": _assistant_message,
        "BeforeTool": _tool_start,
        "AfterTool": _tool_end,
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


GeminiCLIAdapter = GeminiCLIHooksAdapter


def build_hooks_config(command: str, *, timeout: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Build Gemini CLI hook settings using *command*."""
    return build_spec_hooks_config(SPEC, command, timeout=timeout)


def response_for_payload(payload: dict[str, Any]) -> None:
    """Gemini capture hooks are observational and emit no response."""
    return None


def init(project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
    """Merge Agent Context Graph hooks into ``.gemini/settings.json``."""
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
class _GeminiCLIPlugin:
    """Gemini CLI runtime registration."""

    name: str = "gemini-cli"
    adapter_class: type[RuntimeAdapter] = GeminiCLIHooksAdapter
    probe_payload: Mapping[str, Any] = field(default_factory=lambda: SPEC.probe_payload)

    def response_for_payload(self, payload: dict[str, Any]) -> None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        return build_hooks_config(command, timeout=timeout)

    def init(self, project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
        init(project_dir, connectors, **kwargs)


PLUGIN = _GeminiCLIPlugin()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Gemini CLI hook command."""
    from agent_context_graph.hooks.runner import run_hook

    return run_hook(PLUGIN, argv)


if __name__ == "__main__":
    raise SystemExit(main())
