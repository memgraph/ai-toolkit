"""GitHub Copilot CLI command-hook runtime adapter."""

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
    AgentEndEvent,
    AgentStartEvent,
    ErrorOccurredEvent,
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
    reason = str(
        context.payload.get("reason")
        or context.payload.get("stop_reason")
        or context.payload.get("stopReason")
        or "completed"
    )
    return SessionEndEvent(**context.base(), status="completed" if reason in {"complete", "end_turn"} else reason)


def _message(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="user", content=context.value("prompt", ""))


def _tool_start(context: EventContext) -> ToolStartEvent:
    return ToolStartEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_input=context.value("tool_input"),
        tool_use_id=context.optional_text("tool_use_id"),
    )


def _tool_end(context: EventContext) -> ToolEndEvent:
    result, is_error, error_message = extract_tool_result(context.value("tool_result"))
    tool_input = context.value("tool_input")
    if tool_input is not None:
        context.metadata["tool_input"] = tool_input
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=result,
        is_error=is_error,
        error_message=error_message,
    )


def _tool_failure(context: EventContext) -> ToolEndEvent:
    tool_input = context.value("tool_input")
    if tool_input is not None:
        context.metadata["tool_input"] = tool_input
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        is_error=True,
        error_message=str(context.payload.get("error") or "Tool failed"),
    )


def _permission(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="system", content=context.text("tool_name", "permission_request"))


def _agent_start(context: EventContext) -> AgentStartEvent:
    agent_type = context.text("agent_type")
    # Copilot's start payload has agentName but no agentId. Use the field
    # shared by both lifecycle payloads so start and stop update one Agent.
    return AgentStartEvent(**context.base(), agent_name=agent_type, agent_type=agent_type)


def _agent_end(context: EventContext) -> AgentEndEvent:
    agent_type = context.text("agent_type")
    output = context.payload.get("last_assistant_message", context.payload.get("response"))
    return AgentEndEvent(
        **context.base(),
        agent_name=agent_type,
        agent_type=agent_type,
        output=None if output is None else str(output),
    )


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
    "errorOccurred",
)
SPEC = RuntimeSpec(
    name="copilot-cli",
    source_sdk="copilot-cli",
    event_key="hook_event_name",
    hooks=_HOOKS,
    rules={
        "SessionStart": _session_start,
        "sessionStart": _session_start,
        "SessionEnd": _session_end,
        "sessionEnd": _session_end,
        "UserPromptSubmit": _message,
        "userPromptSubmitted": _message,
        "PreToolUse": _tool_start,
        "preToolUse": _tool_start,
        "PostToolUse": _tool_end,
        "postToolUse": _tool_end,
        "PostToolUseFailure": _tool_failure,
        "postToolUseFailure": _tool_failure,
        "PermissionRequest": _permission,
        "permissionRequest": _permission,
        "SubagentStart": _agent_start,
        "subagentStart": _agent_start,
        "SubagentStop": _agent_end,
        "subagentStop": _agent_end,
        "ErrorOccurred": _error,
        "errorOccurred": _error,
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
            "postToolUseFailure": ".*",
            "permissionRequest": ".*",
            "subagentStart": ".*",
        },
    ),
    probe_payload={"hook_event_name": "sessionEnd", "sessionId": "doctor", "reason": "complete"},
)


class CopilotCLIHooksAdapter(SpecAdapter):
    """Convert Copilot CLI hook payloads into Event Protocol events."""

    SPEC = SPEC


CopilotCLIAdapter = CopilotCLIHooksAdapter


def build_hooks_config(command: str, *, timeout: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Build a Copilot CLI hooks map using VS Code-compatible event names."""
    return build_spec_hooks_config(SPEC, command, timeout=timeout)


def response_for_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Allow completion hooks without changing Copilot's control flow."""
    if payload.get("hook_event_name") in {"Stop", "SubagentStop", "agentStop", "subagentStop"}:
        return {"decision": "allow"}
    return None


def init(project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
    """Write ``.github/hooks/agent-context-graph.json``."""
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
class _CopilotCLIPlugin:
    """Copilot CLI runtime registration."""

    name: str = "copilot-cli"
    adapter_class: type[RuntimeAdapter] = CopilotCLIHooksAdapter
    probe_payload: Mapping[str, Any] = field(default_factory=lambda: SPEC.probe_payload)

    def response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        return build_hooks_config(command, timeout=timeout)

    def init(self, project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
        init(project_dir, connectors, **kwargs)


PLUGIN = _CopilotCLIPlugin()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Copilot CLI hook command."""
    from agent_context_graph.hooks.runner import run_hook

    return run_hook(PLUGIN, argv)


if __name__ == "__main__":
    raise SystemExit(main())
