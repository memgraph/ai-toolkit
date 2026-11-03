"""OpenAI Codex command-hook runtime adapter."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any

from memgraph_context_graph.adapters._spec import (
    EventContext,
    HookConfig,
    RuntimeSpec,
    SpecAdapter,
    SpecPlugin,
    hook_specific_session_context,
    pre_tool_use_context,
    session_end,
    session_start,
    tool_end,
    tool_start,
    turn_end,
    write_hook_config,
)
from memgraph_context_graph.adapters.codex_memory import CODEX_MEMORY
from memgraph_context_graph.events import MessageEvent, ToolEndEvent

if TYPE_CHECKING:
    from pathlib import Path

    from memgraph_context_graph.events import Event

#: Enough of the rollout to hold the turn being handled, without reading a long session's whole file.
_ROLLOUT_TAIL_BYTES = 4 * 1024 * 1024
#: How long PostToolUse waits for Codex's rollout writer to record the call it is about.
_ITEM_WAIT_SECONDS = 0.5
_ITEM_POLL_SECONDS = 0.05
_ITEM_TOOL_NAMES = {"CommandExecution": "Bash", "FileChange": "apply_patch"}


def _user_prompt(context: EventContext) -> MessageEvent:
    return MessageEvent(
        **context.base(),
        role="user",
        content=context.value("prompt", ""),
        model=context.optional_text("model"),
    )


def _permission(context: EventContext) -> MessageEvent:
    return MessageEvent(**context.base(), role="system", content=context.text("tool_name", "permission_request"))


def _completed_items(context: EventContext) -> dict[str, dict[str, Any]]:
    """This turn's ``item_completed`` records in Codex's rollout (``transcript_path``), by item id.

    An item's id is the ``tool_use_id`` the tool hooks carry. Only the rollout's tail is read: the turn
    being handled is the last one written. An unreadable rollout yields nothing.
    """
    path, turn_id = context.optional_text("transcript_path"), context.optional_text("turn_id")
    if not path or not turn_id:
        return {}
    try:
        with open(path, "rb") as file:
            start = max(0, file.seek(0, os.SEEK_END) - _ROLLOUT_TAIL_BYTES)
            file.seek(start)
            lines = file.read().split(b"\n")[1 if start else 0 :]
    except OSError:
        return {}
    items = {}
    for line in lines:
        if b'"item_completed"' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        payload = record.get("payload") if isinstance(record, dict) else None
        if not isinstance(payload, dict) or payload.get("type") != "item_completed":
            continue
        item = payload.get("item")
        if payload.get("turn_id") == turn_id and isinstance(item, dict) and isinstance(item.get("id"), str):
            items[item["id"]] = item
    return items


def _item_result(item: dict[str, Any]) -> Any:
    """What the tool behind a rollout item returned, shaped as its hook result would be."""
    result = item.get("result")
    if isinstance(result, dict):
        return result.get("content", result)
    return result if result is not None else item.get("aggregated_output", item.get("stdout"))


def _failure(item: dict[str, Any]) -> str | None:
    """The error message of a failed rollout item, or None if it succeeded."""
    exit_code = item.get("exit_code")
    if item.get("status") != "failed" and exit_code in (None, 0):
        return None
    blocks = _item_result(item)
    texts = (
        [b["text"] for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str)]
        if isinstance(blocks, list)
        else []
    )
    detail = "\n".join(texts) or item.get("error") or item.get("aggregated_output") or item.get("stderr")
    if detail:
        return str(detail)
    return f"exit code {exit_code}" if exit_code not in (None, 0) else f"{item.get('type', 'tool')} failed"


def _hook_tool_name(item: dict[str, Any]) -> str:
    """The ``tool_name`` Codex's hooks give the tool behind a rollout item."""
    if item.get("type") == "McpToolCall":
        server = re.sub(r"[^A-Za-z0-9_]", "_", str(item.get("server", "")))
        return f"mcp__{server}__{item.get('tool', '')}"
    return _ITEM_TOOL_NAMES.get(str(item.get("type")), str(item.get("type", "tool")))


def _tool_end(context: EventContext) -> ToolEndEvent:
    """Codex's tool end, failed when the rollout says so.

    A ``Bash`` call that exits non-zero still reaches PostToolUse, but the hook gets only its output;
    the exit code is in the rollout, whose record of the call Codex normally writes before running the
    hook (hence the short wait). A result stored as completed here stays so: :func:`_turn_end` can't
    correct it. Other tools reach PostToolUse only when they succeeded.
    """
    event = tool_end(context)
    if event.tool_name != "Bash" or not event.tool_use_id:
        return event
    deadline = time.monotonic() + _ITEM_WAIT_SECONDS
    while (item := _completed_items(context).get(event.tool_use_id)) is None and time.monotonic() < deadline:
        time.sleep(_ITEM_POLL_SECONDS)
    message = _failure(item) if item else None
    return dataclasses.replace(event, is_error=True, error_message=message) if message else event


def _turn_end(context: EventContext) -> list[Event]:
    """End the turn, first recording the tool calls that failed in it.

    Codex runs no PostToolUse hook for a failed tool (an MCP error, a rejected patch), so its failures
    are read from the turn's rollout. A failure PostToolUse already recorded is the same event again,
    which connectors store once. A patch Codex rejects before applying leaves no rollout item and
    stays a call without a result.
    """
    failures: list[Event] = [
        ToolEndEvent(
            **context.base(),
            tool_name=_hook_tool_name(item),
            tool_use_id=item_id,
            result=_item_result(item),
            is_error=True,
            error_message=message,
        )
        for item_id, item in _completed_items(context).items()
        if (message := _failure(item))
    ]
    # Stop fires after every turn; SessionEnd, registered separately, ends the session.
    return [*failures, *turn_end(context, reply=context.value("reply"))]


_HOOKS_PATH = ".codex/hooks.json"
SPEC = RuntimeSpec(
    name="codex",
    source_sdk="codex",
    event_key="hook_event_name",
    hooks=("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PermissionRequest", "Stop", "SessionEnd"),
    rules={
        "SessionStart": session_start,
        "UserPromptSubmit": _user_prompt,
        "PreToolUse": tool_start,
        "PostToolUse": _tool_end,
        "PermissionRequest": _permission,
        "Stop": _turn_end,
        "SessionEnd": session_end,
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
    session_context=hook_specific_session_context,
    # Codex cuts hook context at about 2,500 tokens (docs/research/2026-10-codex-memory.md);
    # 9,000 characters stays under that even with multi-byte text.
    session_context_limit=9_000,
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
