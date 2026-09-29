"""Declarative support for command-hook runtime adapters.

Runtime modules describe field aliases, event rules, hook responses, and hook
configuration in a :class:`RuntimeSpec`. This module owns the shared payload
dispatch, command-config rendering, and runtime registration so a new runtime
only declares what differs about it.
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agent_context_graph.adapters._identity import resolve_user_id
from agent_context_graph.events import SessionStartEvent, ToolEndEvent, ToolStartEvent
from agent_context_graph.protocols import RuntimeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from agent_context_graph.events import Event
    from agent_context_graph.link import AgentLink

EventRule = Callable[["EventContext"], "Event | list[Event] | None"]
Installer = Callable[..., None]


@dataclass(frozen=True)
class FieldMap:
    """Aliases for common fields in command-hook payloads."""

    session_id: tuple[str, ...] = ("session_id", "sessionId")
    cwd: tuple[str, ...] = ("cwd",)
    model: tuple[str, ...] = ("model",)
    tool_name: tuple[str, ...] = ("tool_name", "toolName")
    tool_input: tuple[str, ...] = ("tool_input", "toolArgs", "tool_args")
    tool_result: tuple[str, ...] = (
        "tool_response",
        "tool_result",
        "toolResult",
        "tool_output",
        "toolOutput",
    )
    tool_use_id: tuple[str, ...] = ("tool_use_id", "toolUseId", "toolCallId")
    prompt: tuple[str, ...] = ("prompt",)
    agent_id: tuple[str, ...] = ("agent_id", "agentId", "subagent_id", "subagentId")
    agent_type: tuple[str, ...] = (
        "agent_type",
        "agentType",
        "agent_name",
        "agentName",
        "subagent_type",
        "subagentType",
    )


@dataclass(frozen=True)
class HookConfig:
    """Describe how a runtime represents and stores command hooks.

    ``path`` is ``None`` for a runtime whose hooks are installed by something
    other than a project-local JSON file (e.g. Claude Code's plugin).
    ``flat_hooks`` names events written with the flat layout even when
    ``layout`` is nested, for runtimes that only accept matcher groups on some
    events.
    """

    layout: Literal["nested", "flat"]
    path: str | None = None
    root_key: str = "hooks"
    version: int | None = None
    merge: bool = False
    timeout_key: str | None = "timeout"
    timeout_multiplier: int = 1
    command_key: str = "command"
    include_type: bool = True
    inject_event_name: bool = False
    matchers: Mapping[str, str] = field(default_factory=dict)
    flat_hooks: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RuntimeSpec:
    """Declarative contract for one command-hook runtime.

    ``responses`` maps an event name to the JSON the runtime expects on stdout
    for it; events not listed get no output. ``event_aliases`` maps alternate
    spellings of an event name onto the canonical name used by ``rules`` and
    ``responses``.
    """

    name: str
    source_sdk: str
    event_key: str
    hooks: tuple[str, ...]
    rules: Mapping[str, EventRule]
    metadata_keys: tuple[str, ...]
    config: HookConfig
    fields: FieldMap = FieldMap()
    responses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    event_aliases: Mapping[str, str] = field(default_factory=dict)
    probe_payload: Mapping[str, Any] = field(default_factory=dict)

    def event_name(self, payload: Mapping[str, Any]) -> str:
        """Return *payload*'s canonical event name."""
        name = str(payload.get(self.event_key))
        return self.event_aliases.get(name, name)


@dataclass
class EventContext:
    """Resolved payload values supplied to a runtime's event rule."""

    spec: RuntimeSpec
    payload: dict[str, Any]
    session_id: str
    metadata: dict[str, Any]

    def value(self, field_name: str, default: Any = None) -> Any:
        """Return the first present alias for a common field."""
        aliases = getattr(self.spec.fields, field_name, (field_name,))
        for alias in aliases:
            if alias in self.payload:
                return self.payload[alias]
        return default

    def text(self, field_name: str, default: str = "") -> str:
        """Return a common field coerced to text, or *default* when missing or empty."""
        value = self.value(field_name)
        return default if value is None or value == "" else str(value)

    def optional_text(self, field_name: str) -> str | None:
        """Return a common field as text when present."""
        value = self.value(field_name)
        return None if value is None else str(value)

    def base(self) -> dict[str, Any]:
        """Return keyword arguments shared by every Event Protocol event."""
        return {
            "session_id": self.session_id,
            "source_sdk": self.spec.source_sdk,
            "metadata": self.metadata,
        }


class SpecAdapter(RuntimeAdapter):
    """Translate command-hook payloads using a :class:`RuntimeSpec`."""

    SPEC: RuntimeSpec

    def __init__(self, link: AgentLink, session_id: str | None = None) -> None:
        self._link = link
        self._session_id = session_id

    def get_runtime_hooks(self) -> dict[str, list[dict[str, Any]]]:
        """Return the runtime's hook configuration with its default command."""
        command = f"agent-context-graph hook run {self.SPEC.name}"
        return build_hooks_config(self.SPEC, command)

    def handle_payload(self, payload: dict[str, Any]) -> list[Event]:
        """Translate and emit every event described by *payload*."""
        rule = self.SPEC.rules.get(self.SPEC.event_name(payload))
        if rule is None:
            return []

        session_id = self._session_id or self._resolve_session_id(payload)
        context = EventContext(
            spec=self.SPEC,
            payload=payload,
            session_id=session_id,
            metadata=_metadata_from_payload(payload, self.SPEC.metadata_keys),
        )
        result = rule(context)
        events = result if isinstance(result, list) else ([] if result is None else [result])
        for event in events:
            self._link.emit(event)
        return events

    def _resolve_session_id(self, payload: dict[str, Any]) -> str:
        for key in self.SPEC.fields.session_id:
            if payload.get(key) is not None:
                return str(payload[key])
        return ""


@dataclass(frozen=True)
class SpecPlugin:
    """Runtime registration (``RuntimeCLIPlugin``) derived from a :class:`RuntimeSpec`.

    ``init`` is ``None`` for a runtime with no project-local installer, which
    the CLI reports as "not implemented" (see ``RuntimeCLIPlugin``).
    """

    spec: RuntimeSpec
    adapter_class: type[SpecAdapter]
    init: Installer | None = None

    @property
    def name(self) -> str:
        """Return the runtime's registered name."""
        return self.spec.name

    @property
    def probe_payload(self) -> Mapping[str, Any]:
        """Return the payload ``doctor`` replays to smoke-test the runtime."""
        return self.spec.probe_payload

    def response_for_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return the JSON the runtime expects on stdout for *payload*, if any."""
        response = self.spec.responses.get(self.spec.event_name(payload))
        return None if response is None else dict(response)

    def build_hooks_config(self, command: str, *, timeout: int = 30) -> dict[str, Any]:
        """Render the runtime's hook map using *command*."""
        return build_hooks_config(self.spec, command, timeout=timeout)


def json_hook_installer(spec: RuntimeSpec) -> Installer:
    """Return an ``init`` that writes *spec*'s project-local JSON hook file."""

    def init(
        project_dir: Path,
        connectors: list[str],
        *,
        hook_command: str | None = None,
        timeout: int = 30,
        force: bool = False,
    ) -> None:
        path = write_hook_config(spec, project_dir, connectors, hook_command=hook_command, timeout=timeout, force=force)
        print(f"Wrote {path}")

    return init


def build_hooks_config(spec: RuntimeSpec, command: str, *, timeout: int = 30) -> dict[str, list[dict[str, Any]]]:
    """Render the runtime-specific command-hook map for *spec*."""
    config: dict[str, list[dict[str, Any]]] = {}
    for hook_name in spec.hooks:
        hook_command = command
        if spec.config.inject_event_name:
            hook_command = f"{command} --event-name {shlex.quote(hook_name)}"
        command_entry: dict[str, Any] = {spec.config.command_key: hook_command}
        if spec.config.include_type:
            command_entry["type"] = "command"
        if spec.config.timeout_key is not None:
            command_entry[spec.config.timeout_key] = timeout * spec.config.timeout_multiplier

        if spec.config.layout == "nested" and hook_name not in spec.config.flat_hooks:
            entry: dict[str, Any] = {"hooks": [command_entry]}
        else:
            entry = command_entry
        matcher = spec.config.matchers.get(hook_name)
        if matcher is not None:
            entry["matcher"] = matcher
        config[hook_name] = [entry]
    return config


def write_hook_config(
    spec: RuntimeSpec,
    project_dir: Path,
    connectors: list[str],
    *,
    hook_command: str | None = None,
    timeout: int = 30,
    force: bool = False,
) -> Path:
    """Write a runtime's project-local JSON hook configuration.

    Existing settings files are merged only when the runtime declares
    ``merge=True``. Otherwise callers must pass ``force=True`` to replace one.

    Raises:
        ValueError: if *spec* declares no config path, or an existing file to
            merge into is not a JSON object with an object-valued hook map.
        FileExistsError: if the file exists, is not mergeable, and *force* is false.
    """
    if spec.config.path is None:
        raise ValueError(f"{spec.name} has no project-local hook config file")
    config_path = project_dir / spec.config.path
    if config_path.exists() and not spec.config.merge and not force:
        raise FileExistsError(f"Refusing to overwrite existing {spec.name} config: {config_path} (pass force=True)")

    command = hook_command or hook_command_for(spec.name, connectors)
    hooks = build_hooks_config(spec, command, timeout=timeout)
    document: dict[str, Any]
    if config_path.exists() and spec.config.merge:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a JSON object in {config_path}")
        document = loaded
    else:
        document = {}
    if spec.config.version is not None:
        document["version"] = spec.config.version
    existing_hooks = document.get(spec.config.root_key, {})
    if not isinstance(existing_hooks, dict):
        raise ValueError(f"Expected {spec.config.root_key!r} to be an object in {config_path}")
    document[spec.config.root_key] = {**existing_hooks, **hooks}

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return config_path


def hook_command_argv(runtime: str, connectors: list[str]) -> list[str]:
    """Build the installed hook command for *runtime* and *connectors* as argv."""
    executable = shutil.which("agent-context-graph")
    base = [executable] if executable else [sys.executable, "-m", "agent_context_graph.cli"]
    argv = [*base, "hook", "run", runtime]
    for connector in connectors:
        argv.extend(["--connector", connector])
    return argv


def hook_command_for(runtime: str, connectors: list[str]) -> str:
    """Build the installed hook command for *runtime* and *connectors* as a shell string."""
    return shlex.join(hook_command_argv(runtime, connectors))


def _metadata_from_payload(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload and payload[key] is not None}


def event_user_id(context: EventContext) -> str | None:
    """Resolve the stable user identity for a session-start event."""
    return resolve_user_id(context.payload)


def string_or_none(value: Any) -> str | None:
    """Coerce *value* to text while preserving ``None``."""
    return None if value is None else str(value)


def extract_tool_result(tool_response: Any) -> tuple[Any, bool, str | None]:
    """Normalize common command-hook tool result shapes."""
    if not isinstance(tool_response, dict):
        return tool_response, False, None
    error = tool_response.get("error")
    exit_code = tool_response.get("exit_code", tool_response.get("exitCode"))
    is_error = bool(tool_response.get("is_error", False) or error or exit_code not in (None, 0))
    error_message = error or tool_response.get("stderr")
    result = tool_response.get(
        "content",
        tool_response.get(
            "llmContent", tool_response.get("text_result_for_llm", tool_response.get("textResultForLlm"))
        ),
    )
    if result is None:
        result = tool_response
    return result, is_error, string_or_none(error_message)


def session_start(context: EventContext) -> SessionStartEvent:
    """Default session-start rule: working directory, model, and user identity."""
    return SessionStartEvent(
        **context.base(),
        model=context.optional_text("model"),
        working_directory=context.optional_text("cwd"),
        user_id=event_user_id(context),
    )


def tool_start(context: EventContext) -> ToolStartEvent:
    """Default tool-start rule shared by every command-hook runtime."""
    return ToolStartEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_input=context.value("tool_input"),
        tool_use_id=context.optional_text("tool_use_id"),
        agent_name=context.optional_text("agent_id"),
    )


def tool_end(context: EventContext) -> ToolEndEvent:
    """Default tool-end rule: normalize the result via :func:`extract_tool_result`."""
    result, is_error, error_message = extract_tool_result(context.value("tool_result"))
    return ToolEndEvent(
        **context.base(),
        tool_name=context.text("tool_name"),
        tool_use_id=context.optional_text("tool_use_id"),
        result=result,
        is_error=is_error,
        error_message=error_message,
        agent_name=context.optional_text("agent_id"),
    )
