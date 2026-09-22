"""OpenAI Codex command-hook runtime adapter."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    event_user_id,
    extract_tool_result,
)
from agent_context_graph.adapters._spec import (
    build_hooks_config as build_spec_hooks_config,
)
from agent_context_graph.adapters._spec import (
    response_for_payload as spec_response_for_payload,
)
from agent_context_graph.events import MessageEvent, SessionEndEvent, SessionStartEvent, ToolEndEvent, ToolStartEvent
from agent_context_graph.hooks.runner import create_link, load_payload  # noqa: F401 -- public compatibility exports
from memgraph_toolbox.api.memgraph import MEMGRAPH_ENV_KEYS, memgraph_env

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from agent_context_graph.protocols import RuntimeAdapter


def _session_start(context: EventContext) -> SessionStartEvent:
    return SessionStartEvent(
        **context.base(),
        model=context.optional_text("model"),
        working_directory=context.optional_text("cwd"),
        user_id=event_user_id(context),
    )


def _user_prompt(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="user",
        content=context.value("prompt", ""),
        model=context.optional_text("model"),
    )


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


def _permission(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="system", content=context.text("tool_name", "permission_request"))


def _session_end(context: EventContext) -> SessionEndEvent:
    return SessionEndEvent(**context.base(), status="completed")


SPEC = RuntimeSpec(
    name="codex",
    source_sdk="codex",
    event_key="hook_event_name",
    hooks=("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PermissionRequest", "Stop"),
    rules={
        "SessionStart": _session_start,
        "UserPromptSubmit": _user_prompt,
        "PreToolUse": _tool_start,
        "PostToolUse": _tool_end,
        "PermissionRequest": _permission,
        "Stop": _session_end,
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
        path=".codex/hooks.json",
        layout="nested",
        matchers={"SessionStart": "startup|resume|clear", "PreToolUse": "*", "PostToolUse": "*"},
    ),
    response_events=frozenset({"Stop"}),
    probe_payload={"hook_event_name": "Stop", "session_id": "doctor"},
)


class CodexHooksAdapter(SpecAdapter):
    """Convert Codex command-hook payloads into Event Protocol events."""

    SPEC = SPEC


CodexAdapter = CodexHooksAdapter


def build_hooks_config(command: str, *, timeout: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Build a Codex hooks configuration using *command*."""
    return build_spec_hooks_config(SPEC, command, timeout=timeout)


def response_for_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return hook JSON when Codex requires a response."""
    return spec_response_for_payload(SPEC, payload)


def init(project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
    """Generate private Codex ``config.toml`` and ``hooks.json`` files."""
    hook_command = kwargs.get("hook_command")
    timeout = kwargs.get("timeout", 30)
    force = kwargs.get("force", False)
    setup_schema = kwargs.get("setup_schema", False)

    codex_dir = project_dir / ".codex"
    config_path = codex_dir / "config.toml"
    hooks_path = codex_dir / "hooks.json"
    existing = [path for path in (config_path, hooks_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing Codex config: {names} (pass force=True to replace)")

    resolved_memgraph_env = memgraph_env(
        url=kwargs.get("memgraph_url"),
        username=kwargs.get("memgraph_user"),
        password=kwargs.get("memgraph_password"),
        database=kwargs.get("memgraph_database"),
    )
    if hook_command is None:
        executable = shutil.which("agent-context-graph")
        base = [executable] if executable else [sys.executable, "-m", "agent_context_graph.cli"]
        command_parts = [*base, "hook", "run", "codex"]
        for connector in connectors:
            command_parts.extend(["--connector", connector])
        hook_command = shlex.join(command_parts)

    codex_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("[features]\nhooks = true\n", encoding="utf-8")
    hooks_path.write_text(
        json.dumps({"hooks": build_hooks_config(hook_command, timeout=timeout)}, indent=2) + "\n",
        encoding="utf-8",
    )

    if setup_schema:
        previous = {key: os.environ.get(key) for key in MEMGRAPH_ENV_KEYS}
        os.environ.update(resolved_memgraph_env)
        try:
            for connector in connectors:
                normalized = connector.strip().replace("-", "_")
                if normalized == "skills_graph":
                    from skills_graph import SkillGraph

                    SkillGraph().setup()
                elif normalized == "actions_graph":
                    from actions_graph import ActionsGraph

                    ActionsGraph().setup()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    print(f"Wrote {config_path}")
    print(f"Wrote {hooks_path}")
    print(f"Memgraph URL: {resolved_memgraph_env['MEMGRAPH_URL']}")
    print(f"Memgraph database: {resolved_memgraph_env['MEMGRAPH_DATABASE']}")
    secret = resolved_memgraph_env["MEMGRAPH_PASSWORD"]
    masked = hook_command.replace(shlex.quote(secret), "'****'").replace(secret, "****") if secret else hook_command
    print(f"Hook command: {masked}")


@dataclass(frozen=True)
class _CodexPlugin:
    """Codex runtime registration."""

    name: str = "codex"
    adapter_class: type[RuntimeAdapter] = CodexHooksAdapter
    probe_payload: Mapping[str, Any] = field(default_factory=lambda: SPEC.probe_payload)

    def response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        return response_for_payload(payload)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        return build_hooks_config(command, timeout=timeout)

    def init(self, project_dir: Path, connectors: list[str], **kwargs: Any) -> None:
        init(project_dir, connectors, **kwargs)


PLUGIN = _CodexPlugin()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Codex hook CLI."""
    from agent_context_graph.hooks.runner import run_hook

    return run_hook(PLUGIN, argv)


if __name__ == "__main__":
    raise SystemExit(main())
