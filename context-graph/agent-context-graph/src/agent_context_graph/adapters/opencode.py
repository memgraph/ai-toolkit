"""OpenCode V2 plugin runtime adapter and installer."""

from __future__ import annotations

import json
import shlex
from importlib import resources
from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    SpecPlugin,
    hook_command_argv,
    session_start,
    tool_start,
)
from agent_context_graph.events import ErrorOccurredEvent, MessageEvent, SessionEndEvent, ToolEndEvent

if TYPE_CHECKING:
    from pathlib import Path


def _session_end(context: EventContext) -> SessionEndEvent:
    return SessionEndEvent(**context.base(), status="completed")


def _prompt(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="user", content=context.value("prompt", ""))


def _assistant_text(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="assistant", content=context.payload.get("text", ""))


def _tool_end(context: EventContext) -> ToolEndEvent:
    error = context.payload.get("error")
    error_dict = error if isinstance(error, dict) else {}
    result, exit_code = _tool_result(context.value("tool_result"))
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=result,
        is_error=bool(context.payload.get("is_error")) or exit_code not in (None, 0),
        error_message=str(error_dict.get("message", error)) if error else None,
    )


def _tool_result(tool_result: Any) -> tuple[Any, Any]:
    """Return a V2 tool result's model-facing text and shell exit code, if any.

    V2 results are ``{output, content: [{type: "text", text}], metadata}``;
    ``content`` is what the model saw, ``output`` is tool-specific.
    """
    if not isinstance(tool_result, dict):
        return tool_result, None
    output = tool_result.get("output")
    exit_code = output.get("exit") if isinstance(output, dict) else None
    parts = tool_result.get("content")
    if isinstance(parts, list):
        texts = [part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
        if texts:
            return "\n".join(texts), exit_code
    return tool_result, exit_code


def _execution_failed(context: EventContext) -> ErrorOccurredEvent:
    error = context.payload.get("error")
    error_dict = error if isinstance(error, dict) else {}
    details = {"status": error_dict["status"]} if error_dict.get("status") is not None else {}
    return ErrorOccurredEvent(
        **context.base(),
        error_type=str(error_dict.get("type") or "opencode_error"),
        error_message=str(error_dict.get("message") or error or "OpenCode session execution failed"),
        error_details=details,
        recoverable=True,
    )


def _permission(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="system",
        content=context.payload.get("permission") or "permission_requested",
    )


SPEC = RuntimeSpec(
    name="opencode",
    source_sdk="opencode",
    event_key="hook_event_name",
    # OpenCode has no JSON command hooks: the installed V2 plugin subscribes to
    # its hooks and bus events and pipes each one, normalized, to
    # ``hook run opencode`` (see _opencode_plugin.js for the field mapping).
    hooks=(),
    rules={
        "session.created": session_start,
        "session.deleted": _session_end,
        "session.prompt": _prompt,
        # Emitted once per completed assistant text part, unlike the streamed
        # session.text.delta, so each assistant message is recorded once.
        "session.text.ended": _assistant_text,
        "tool.execute.before": tool_start,
        "tool.execute.after": _tool_end,
        "session.execution.failed": _execution_failed,
        "permission.asked": _permission,
    },
    metadata_keys=("cwd", "model", "metadata"),
    config=HookConfig(layout="flat"),
    probe_payload={"hook_event_name": "session.deleted", "session_id": "doctor"},
)


class OpenCodeHooksAdapter(SpecAdapter):
    """Convert normalized OpenCode V2 plugin callbacks into events."""

    SPEC = SPEC


_PLUGIN_PATH = ".opencode/plugins/agent-context-graph/index.js"


def init(
    project_dir: Path,
    connectors: list[str],
    *,
    hook_command: str | None = None,
    timeout: int = 30,
    force: bool = False,
) -> None:
    """Install the dependency-free OpenCode V2 capture plugin.

    The hook command is embedded as an argv array and spawned directly, so no
    shell (and no login-shell profile) sits between OpenCode and the hook.
    *timeout* is accepted for CLI symmetry; OpenCode plugins have no hook timeout.

    Raises:
        FileExistsError: if the plugin file exists and *force* is false.
    """
    target = project_dir / _PLUGIN_PATH
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing OpenCode plugin: {target} (pass force=True)")

    argv = shlex.split(hook_command) if hook_command else hook_command_argv(SPEC.name, connectors)
    template = (
        resources.files("agent_context_graph.adapters").joinpath("_opencode_plugin.js").read_text(encoding="utf-8")
    )
    plugin = template.replace("__AGENT_CONTEXT_GRAPH_COMMAND__", json.dumps(argv))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plugin, encoding="utf-8")
    print(f"Wrote {target}")


PLUGIN = SpecPlugin(SPEC, OpenCodeHooksAdapter, init=init)
