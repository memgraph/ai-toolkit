"""Hook Configuration — persistent config for hook subprocesses.

Agent runtimes (Claude Code, Codex) spawn hook commands as non-interactive
subprocesses that do not inherit shell profile environment variables.  This
module provides a config-file-only resolution path so hook subprocesses (and
subprocesses they in turn spawn, e.g. sessions-graph's detached embed and
reconcile runs) can reliably read every setting the file holds; its keys are
:data:`memgraph_context_graph.config_schema.CONFIG_KEYS`.

Resolution order (per ADR 0002):
  CLI flag > config file > hardcoded default

Environment variables are **not** consulted for configuration *values* at hook
runtime, and nothing writes them into the file: ``setup`` and ``config set``
take values from the user. A process a hook spawns is handed the file's path and
reads it itself (:func:`spawn_env`, :func:`fill_env_from_config`; ADR 0007).

Config file location: ``~/.config/context-graph/config.toml``, unless
``CONTEXT_GRAPH_CONFIG`` names another file.  That override selects *which file*
to read, never what is in it, and exists because the default is a single global:
pointing one session's hooks at a different Memgraph would otherwise redirect
every other Claude Code session on the machine.  See ADR 0003.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from memgraph_context_graph.config_schema import (
    CONFIG_KEYS,
    SECTIONS,
    SETTINGS_SECTIONS,
    TRUTHY_VALUES,
    ConfigKey,
    config_key,
)
from memgraph_context_graph.config_schema import MEMORY_BACKEND_GRAPH as MEMORY_BACKEND_GRAPH
from memgraph_context_graph.config_schema import MEMORY_BACKEND_NATIVE as MEMORY_BACKEND_NATIVE
from memgraph_context_graph.config_schema import MEMORY_BACKENDS as MEMORY_BACKENDS

#: Selects which config *file* hooks read, never a value in it (ADR 0003): a parent
#: process hands the path to the session it spawns, so nothing is ambient.
CONFIG_PATH_ENV = "CONTEXT_GRAPH_CONFIG"

_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "context-graph"


def config_file() -> Path:
    """The config file hooks read, honouring :data:`CONFIG_PATH_ENV`."""
    override = os.environ.get(CONFIG_PATH_ENV)
    return Path(override) if override else default_config_file()


def default_config_file() -> Path:
    """The config file used when :data:`CONFIG_PATH_ENV` is unset."""
    return _DEFAULT_CONFIG_DIR / "config.toml"


# Defaults matching memgraph-toolbox's MEMGRAPH_ENV_DEFAULTS.
_MEMGRAPH_DEFAULTS = {
    "url": "bolt://localhost:7687",
    "user": "",
    "password": "",
    "database": "memgraph",
}

_LLM_DEFAULTS = {
    "openai_api_key": "",
    "anthropic_api_key": "",
}

_cached_config: HookConfig | None = None


@dataclass(frozen=True)
class HookConfig:
    """Parsed hook configuration from the config file."""

    #: Top-level ``edition``: "community" (or None, the default) or "enterprise"; see memgraph_context_graph.spaces.
    edition: str | None = None
    user_id: str | None = None
    memgraph_url: str = _MEMGRAPH_DEFAULTS["url"]
    memgraph_user: str = _MEMGRAPH_DEFAULTS["user"]
    memgraph_password: str = _MEMGRAPH_DEFAULTS["password"]
    memgraph_database: str = _MEMGRAPH_DEFAULTS["database"]
    openai_api_key: str = _LLM_DEFAULTS["openai_api_key"]
    anthropic_api_key: str = _LLM_DEFAULTS["anthropic_api_key"]
    #: ``[llm] derive_model``: the model ontology derivation calls; None means the provider's default.
    derive_model: str | None = None
    #: ``[capture] connectors``: the graph connectors a hook enables when its command names none;
    #: None (never set) means memgraph_context_graph.connectors.DEFAULT_CONNECTORS.
    connectors: tuple[str, ...] | None = None
    #: Every other ``[capture]`` key (``redact``, ``max_chars``, ``exclude_tools``, ``exclude_projects``),
    #: as written: memgraph_context_graph.capture_policy validates them, this module only keeps them.
    capture_settings: dict[str, str] = field(default_factory=dict)
    #: ``[redact]``: custom capture redaction patterns, name to regular expression.
    redact_patterns: dict[str, str] = field(default_factory=dict)
    auto_reconcile: bool | None = None
    #: The model recall embeds with; None means the consumer's own default.
    embedding_model: str | None = None
    #: Every other ``[recall]`` key (lanes and widths), as written: the recall
    #: tool validates them, this module only keeps them across rewrites.
    recall_settings: dict[str, str] = field(default_factory=dict)
    #: ``[ontology] path``: a schema file to extract this user's sessions under.
    ontology_path: str | None = None
    #: ``[ontology] derive``: "extend" (the default) or "off"; see sessions-graph's ontology module.
    ontology_derive: str | None = None
    #: ``[extraction]`` keys (GLiNER2 checkpoint, thresholds, chunking), as written: sessions-graph
    #: validates them, this module only keeps them across rewrites.
    extraction_settings: dict[str, str] = field(default_factory=dict)
    #: ``[github] token``: overrides the ``gh`` login resources-graph's Sweep otherwise fetches with.
    github_token: str | None = None
    #: ``[memory] backend``: ``"context-graph"`` when the user has handed their
    #: harness memory to the graph (the ``memory`` tool and session-start index
    #: are on); ``"native"`` or None leaves the harness's own memory in charge.
    memory_backend: str | None = None
    #: ``[session_start] resume``: whether a session starts with a brief of the
    #: previous sessions in its project. None (never set) means on.
    session_resume: bool | None = None
    #: The ``section.key`` names the file actually sets (top-level keys as ``key``), so a check can tell
    #: "left at the default" from "written": Enterprise refuses an explicit ``identity.user_id`` or
    #: ``memgraph.database``, which would quietly override what the login decides.
    present: frozenset[str] = frozenset()
    #: Why the file could not be read as written: unreadable, or lines that are not ``key = value``.
    #: Values those lines meant to set fell back to defaults, which can point hooks at the wrong
    #: Memgraph, so hooks log these to the capture-failure log and ``doctor`` fails on them.
    problems: tuple[str, ...] = ()

    @property
    def graph_memory(self) -> bool:
        """Whether Context Graph is this user's memory backend."""
        return self.memory_backend == MEMORY_BACKEND_GRAPH


def load_config() -> HookConfig:
    """Load hook configuration from the config file, with defaults for missing values; cached per process."""
    global _cached_config
    if _cached_config is None:
        _cached_config = _read_config_file()
    return _cached_config


def reload_config() -> HookConfig:
    """Re-read the config file, for long-lived processes (the MCP server) that must see later edits."""
    global _cached_config
    _cached_config = None
    return load_config()


def resolve_user_id(payload: dict[str, Any]) -> str | None:
    """Resolve user identity for hook subprocesses, as the configured edition's space resolver decides.

    In Community: the hook payload's ``user_id`` (forward-compat), else ``[identity] user_id``.
    """
    from memgraph_context_graph.spaces import spaces_for

    return spaces_for(load_config()).user_id(payload)


def resolve_memgraph_env(
    *,
    url: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> dict[str, str]:
    """Resolve Memgraph connection settings for hook subprocesses.

    Resolution order per setting: explicit arg (CLI flag) > the user's personal
    space, as the configured edition's space resolver gives it > default.
    Returns a dict with keys matching memgraph-toolbox's env contract.
    """
    from memgraph_context_graph.spaces import spaces_for

    env = spaces_for(load_config()).personal().target.env()
    overrides = {
        "MEMGRAPH_URL": url,
        "MEMGRAPH_USER": user,
        "MEMGRAPH_PASSWORD": password,
        "MEMGRAPH_DATABASE": database,
    }
    return {key: value if value is not None else env[key] for key, value in overrides.items()}


def resolve_llm_env() -> dict[str, str]:
    """Resolve LLM API keys for hook subprocesses, from the config file only.

    An unconfigured key is an empty string. Hook-spawned children must overlay these
    empty values too, so ambient credentials cannot supply a key the config file lacks.
    """
    config = load_config()
    return {
        "OPENAI_API_KEY": config.openai_api_key,
        "ANTHROPIC_API_KEY": config.anthropic_api_key,
    }


#: The variables :func:`fill_env_from_config` sets: memgraph-toolbox's connection contract and the provider
#: keys LightRAG and the derive client read, plus LightRAG's aliases for the connection.
CONFIG_ENV_KEYS = (
    "MEMGRAPH_URL",
    "MEMGRAPH_USER",
    "MEMGRAPH_PASSWORD",
    "MEMGRAPH_DATABASE",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
_CONFIG_ENV_ALIASES = ("MEMGRAPH_URI", "MEMGRAPH_USERNAME")


def fill_env_from_config(*, memgraph: bool = True) -> None:
    """Fill unset connection and LLM key variables in ``os.environ`` from the config file.

    The one place config values become environment variables, for a process whose libraries
    read them there. A variable already set wins, so ``MEMGRAPH_URL=... sessions-graph reconcile``
    still points a hand-run at another instance; a process a hook spawns starts with none of them
    (:func:`spawn_env`), so there the file alone decides. ``memgraph=False`` fills only the LLM
    keys, for a caller that picks its own Memgraph.

    Empty values are set too: an Enterprise login's unnamed database is ``MEMGRAPH_DATABASE=""``,
    which memgraph-toolbox reads as "the login's main database", where unset would mean ``memgraph``.
    """
    values = {**(resolve_memgraph_env() if memgraph else {}), **resolve_llm_env()}
    for key, value in values.items():
        os.environ.setdefault(key, value)


def spawn_env() -> dict[str, str]:
    """The environment for a process a hook spawns: this one's, minus every config-backed variable.

    The child resolves those itself with :func:`fill_env_from_config`, from the file this names in
    :data:`CONFIG_PATH_ENV`: the parent hands down which file, never what is in it (ADR 0003), and
    an ambient credential can't stand in for a key the file leaves empty (ADR 0002).
    """
    env = {key: value for key, value in os.environ.items() if key not in {*CONFIG_ENV_KEYS, *_CONFIG_ENV_ALIASES}}
    env[CONFIG_PATH_ENV] = str(config_file())
    return env


def resolve_embedding_model() -> str | None:
    """The configured ``[recall] embedding_model``, or None to leave sessions-graph's default."""
    return load_config().embedding_model


def resolve_extraction() -> dict[str, str]:
    """The configured ``[extraction]`` keys, as written; sessions-graph owns their meaning."""
    return dict(load_config().extraction_settings)


def resolve_derive_model() -> str | None:
    """The configured ``[llm] derive_model``, or None to leave the provider's default."""
    return load_config().derive_model


def resolve_ontology() -> tuple[str | None, str | None]:
    """The configured ``[ontology]`` (path, derive), each None when unset; sessions-graph owns their meaning."""
    config = load_config()
    return config.ontology_path, config.ontology_derive


def parse_bool_flag(value: str | bool) -> bool:
    """Parse a TOML/CLI boolean flag: :data:`TRUTHY_VALUES` (case-insensitive) is true, a ``bool`` passes through."""
    if isinstance(value, bool):
        return value
    return value.strip().lower() in TRUTHY_VALUES


def write_config(
    *,
    edition: str | None = None,
    user_id: str | None = None,
    memgraph_url: str | None = None,
    memgraph_user: str | None = None,
    memgraph_password: str | None = None,
    memgraph_database: str | None = None,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    derive_model: str | None = None,
    connectors: tuple[str, ...] | None = None,
    capture_settings: dict[str, str] | None = None,
    auto_reconcile: bool | None = None,
    embedding_model: str | None = None,
    ontology_path: str | None = None,
    ontology_derive: str | None = None,
    github_token: str | None = None,
    memory_backend: str | None = None,
    session_resume: bool | None = None,
) -> Path:
    """Write or update the config file. Returns the path written to.

    Only updates supplied values; preserves existing values for unspecified keys.
    ``capture_settings`` replaces the ``[capture]`` keys other than ``connectors`` as a whole.
    Creates the file with 0600 permissions if it does not exist.
    """
    updates = {
        "edition": edition,
        "user_id": user_id,
        "memgraph_url": memgraph_url,
        "memgraph_user": memgraph_user,
        "memgraph_password": memgraph_password,
        "memgraph_database": memgraph_database,
        "openai_api_key": openai_api_key,
        "anthropic_api_key": anthropic_api_key,
        "derive_model": derive_model,
        "connectors": connectors,
        "capture_settings": capture_settings,
        "auto_reconcile": auto_reconcile,
        "embedding_model": embedding_model,
        "ontology_path": ontology_path,
        "ontology_derive": ontology_derive,
        "github_token": github_token,
        "memory_backend": memory_backend,
        "session_resume": session_resume,
    }
    config = replace(_read_config_file(), **{key: value for key, value in updates.items() if value is not None})
    return _write_config_file(_render_config(config))


def set_config_value(name: str, raw: str) -> str:
    """Validate and write one key, as ``config set`` does; an empty value unsets an optional key.

    Returns:
        The value as written, normalized (``"OFF"`` becomes ``"false"``, a list is re-joined).

    Raises:
        KeyError: *name* is not a key of :data:`CONFIG_KEYS`.
        ValueError: *raw* is not a valid value for it, by its type or by the installed component
            that reads its section; the message says what was expected.
    """
    from memgraph_context_graph.connectors import check_settings

    key = config_key(name)
    if key is None:
        raise KeyError(name)
    value = key.normalize(raw)
    config = _read_config_file()
    if key.field is not None:
        config = replace(config, **{key.field: key.parse(value)})
    else:
        attr = SETTINGS_SECTIONS[key.section]
        settings = {**getattr(config, attr), key.key: value}
        if not value:
            del settings[key.key]
        check_settings(key.section, settings)
        config = replace(config, **{attr: settings})
    _write_config_file(_render_config(config))
    return value


def _write_config_file(content: str) -> Path:
    global _cached_config
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: the file holds credentials.
    _cached_config = None
    return path


def _read_config_file() -> HookConfig:
    """Parse the config file, every key as :data:`CONFIG_KEYS` describes it.

    A missing file gives defaults. An unreadable or partly unparsable one also
    gives defaults for whatever it failed to set, never an exception (a hook
    must not crash on its config), and says so in :attr:`HookConfig.problems`.
    """
    problems: list[str] = []
    try:
        if not config_file().is_file():
            return HookConfig()
        sections = _parse_toml(config_file(), problems)
    except Exception as exc:
        return HookConfig(problems=(f"unreadable: {type(exc).__name__}: {exc}",))

    fields: dict[str, Any] = {}
    for key in CONFIG_KEYS:
        raw = sections.get(key.section, {}).get(key.key)
        # An unset or empty key keeps HookConfig's default, except a list, where empty means "none".
        if key.field is not None and raw is not None and (raw or key.kind == "list"):
            fields[key.field] = key.parse(raw)
    field_keys = {(key.section, key.key) for key in CONFIG_KEYS if key.field is not None}
    for section, attr in SETTINGS_SECTIONS.items():
        fields[attr] = {k: v for k, v in sections.get(section, {}).items() if (section, k) not in field_keys}

    return HookConfig(
        **fields,
        redact_patterns=dict(sections.get("redact", {})),
        present=frozenset(
            f"{section}.{key}" if section else key
            for section, keys in sections.items()
            for key, value in keys.items()
            if value
        ),
        problems=tuple(problems),
    )


def _parse_toml(path: Path, problems: list[str] | None = None) -> dict[str, dict[str, str]]:
    """Minimal TOML parser — handles [section] and key = "value" pairs.

    Only supports string values (quoted or bare). Sufficient for our config shape.
    Keys before the first section header land in the ``""`` section. A line that
    is none of these is skipped and, when *problems* is given, reported there.
    """
    sections: dict[str, dict[str, str]] = {"": {}}
    current_section = ""

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            sections.setdefault(current_section, {})
            continue
        if "=" not in stripped or not stripped.partition("=")[0].strip():
            if problems is not None:
                problems.append(f"line {number} is not `key = value` and was ignored")
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if value[:1] not in {'"', "'"} and "#" in value:
            # Bare (unquoted) value with a trailing comment, e.g. `auto_reconcile = true  # off for now`.
            value = value.partition("#")[0].strip()
        value = value.strip('"').strip("'")
        sections[current_section][key] = value

    return sections


def stored_value(config: HookConfig, key: ConfigKey) -> str | None:
    """*key*'s value in *config* as the file stores it, or None while it is unset."""
    if key.field is None:
        return getattr(config, SETTINGS_SECTIONS[key.section]).get(key.key)
    value = getattr(config, key.field)
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return ", ".join(value)
    return str(value)


def _render_config(config: HookConfig) -> str:
    """Render the full config file content: every key the edition uses, with a line saying what it does.

    An unset key is written commented out, with its default where it has one, so the file lists
    everything a user can set while a re-read still tells "never configured" from an explicit value
    (``auto_reconcile = false`` is not the same as leaving it unset). A key the table doesn't know,
    in a section whose keys a component validates, is kept as written.
    """
    lines = [
        "# Context Graph configuration",
        f"# Location: {config_file()}",
        "# A commented-out key is unset; the line above it says what unset means. Uncomment a key to set it.",
        "# setup and `memgraph-context-graph config set` rewrite this file: values are kept, added comments are not.",
    ]
    for section in ("", *SECTIONS):
        keys = [key for key in CONFIG_KEYS if key.section == section and key.applies_to(config.edition)]
        if not keys and section != "redact":
            continue
        lines.append("")
        if section:
            lines.append(f"[{section}]")
        for key in keys:
            value = stored_value(config, key)
            lines.append(f"# {key.doc} Unset: {key.unset}." if key.unset else f"# {key.doc}")
            if value is not None or key.required:
                lines.append(f"{key.key} = {key.format(value or '')}")
            else:
                lines.append(f"# {key.key} = {key.format(key.example)}")
        if section in SETTINGS_SECTIONS:
            known = {key.key for key in CONFIG_KEYS if key.section == section}
            settings = getattr(config, SETTINGS_SECTIONS[section])
            lines += [f'{name} = "{value}"' for name, value in settings.items() if name not in known]
        if section == "redact":
            lines.append("# Extra capture redaction patterns, name = 'regular expression'.")
            lines += [f"{name} = '{pattern}'" for name, pattern in config.redact_patterns.items()]
            if not config.redact_patterns:
                lines.append("# ticket = 'ACME-[0-9]+'")
    lines.append("")
    return "\n".join(lines)


def _reset_cache() -> None:
    """Reset the module cache (for testing)."""
    global _cached_config
    _cached_config = None
