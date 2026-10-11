"""OpenAI Codex command-hook runtime adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    SpecPlugin,
    pre_tool_use_context,
    session_start,
    tool_end,
    tool_start,
    turn_end,
    write_hook_config,
)
from agent_context_graph.adapters.codex_memory import CODEX_MEMORY
from agent_context_graph.events import MessageEvent

if TYPE_CHECKING:
    from pathlib import Path

    from agent_context_graph.events import Event


def _user_prompt(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="user",
        content=context.value("prompt", ""),
        model=context.optional_text("model"),
    )


def _permission(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="system", content=context.text("tool_name", "permission_request"))


def _turn_end(context: EventContext) -> list[Event]:
    # Codex has no session-end hook, and Stop fires after every turn, so it is
    # only ever a turn end.
    return turn_end(context, reply=context.payload.get("last_assistant_message"))


_HOOKS_PATH = ".codex/hooks.json"
SPEC = RuntimeSpec(
    name="codex",
    source_sdk="codex",
    event_key="hook_event_name",
    hooks=("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PermissionRequest", "Stop"),
    rules={
        "SessionStart": session_start,
        "UserPromptSubmit": _user_prompt,
        "PreToolUse": tool_start,
        "PostToolUse": tool_end,
        "PermissionRequest": _permission,
        "Stop": _turn_end,
    },
    metadata_keys=(
        "cwd",
        "source",
        "transcript_path",
        "turn_id",
        "permission_mode",
        "tool_name",
        "tool_input",
        "tool_use_id",
        "reason",
        "decision",
        "stop_hook_active",
    ),
    config=HookConfig(
        path=_HOOKS_PATH,
        layout="nested",
        matchers={"SessionStart": "startup|resume|clear|compact", "PreToolUse": "*", "PostToolUse": "*"},
    ),
    responses={"Stop": {"continue": True}},
    context_before_tool=pre_tool_use_context,
    probe_payload={"hook_event_name": "Stop", "session_id": "doctor"},
)


class CodexHooksAdapter(SpecAdapter):
    """Convert Codex command-hook payloads into Event Protocol events."""

    SPEC = SPEC


def init(
    project_dir: Path,
    connectors: list[str],
    *,
    hook_command: str | None = None,
    timeout: int = 30,
    force: bool = False,
) -> None:
    """Generate private Codex ``config.toml`` and ``hooks.json`` files.

    Raises:
        FileExistsError: if either file exists and *force* is false.
    """
    config_path = project_dir / ".codex" / "config.toml"
    hooks_path = project_dir / _HOOKS_PATH
    existing = [path for path in (config_path, hooks_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing Codex config: {names} (pass force=True to replace)")

    write_hook_config(SPEC, project_dir, connectors, hook_command=hook_command, timeout=timeout, force=True)
    config_path.write_text("[features]\nhooks = true\n", encoding="utf-8")
    print(f"Wrote {config_path}")
    print(f"Wrote {hooks_path}")


PLUGIN = SpecPlugin(SPEC, CodexHooksAdapter, init=init, native_memory=CODEX_MEMORY)
