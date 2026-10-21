"""Hook Configuration — persistent config for hook subprocesses.

Agent runtimes (Claude Code, Codex) spawn hook commands as non-interactive
subprocesses that do not inherit shell profile environment variables.  This
module provides a config-file-only resolution path so hook subprocesses (and
subprocesses they in turn spawn, e.g. sessions-graph's detached
reconciliation process) can reliably access identity, Memgraph connection,
LLM API key, and reconciliation opt-in settings.

Resolution order (per ADR 0002):
  CLI flag > config file > hardcoded default

Environment variables are **not** consulted for configuration *values* at hook
runtime.  They are only used as a write-time input during ``bootstrap`` or
``config set``.

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

#: ``[memory] backend`` values: the graph, or the harness's built-in memory.
MEMORY_BACKEND_GRAPH = "context-graph"
MEMORY_BACKEND_NATIVE = "native"
MEMORY_BACKENDS = frozenset({MEMORY_BACKEND_GRAPH, MEMORY_BACKEND_NATIVE})

#: The boolean spellings parse_bool_flag accepts, shared with ``config set`` validation.
TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
FALSY_VALUES = frozenset({"0", "false", "no", "off"})

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


def write_full_config(
    *,
    user_id: str = "",
    memgraph_url: str = _MEMGRAPH_DEFAULTS["url"],
    memgraph_user: str = _MEMGRAPH_DEFAULTS["user"],
    memgraph_password: str = _MEMGRAPH_DEFAULTS["password"],
    memgraph_database: str = _MEMGRAPH_DEFAULTS["database"],
    openai_api_key: str = _LLM_DEFAULTS["openai_api_key"],
    anthropic_api_key: str = _LLM_DEFAULTS["anthropic_api_key"],
    auto_reconcile: bool | None = None,
) -> Path:
    """Write a complete config file, resetting identity, Memgraph and LLM settings to what is given.

    Every other section (``[capture]``, ``[redact]``, ``[reconcile]``, ``[recall]``, ``[ontology]``, ``[extraction]``,
    ``[github]``, ``[memory]``, ``[session_start]``), ``[llm] derive_model`` and the edition are kept from the existing file, since only ``config set`` or a hand edit ever sets them.
    ``auto_reconcile`` is kept likewise unless given here. :func:`write_config` is the partial update.
    """
    existing = _read_config_file()
    config = replace(
        existing,
        user_id=user_id,
        memgraph_url=memgraph_url,
        memgraph_user=memgraph_user,
        memgraph_password=memgraph_password,
        memgraph_database=memgraph_database,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        auto_reconcile=auto_reconcile if auto_reconcile is not None else existing.auto_reconcile,
    )
    return _write_config_file(_render_config(config))


def _write_config_file(content: str) -> Path:
    global _cached_config
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: the file holds credentials.
    _cached_config = None
    return path


def _read_config_file() -> HookConfig:
    """Parse the config file.

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

    identity = sections.get("identity", {})
    memgraph = sections.get("memgraph", {})
    llm = sections.get("llm", {})
    capture = sections.get("capture", {})
    reconcile = sections.get("reconcile", {})
    recall = sections.get("recall", {})
    ontology = sections.get("ontology", {})
    extraction = sections.get("extraction", {})
    github = sections.get("github", {})
    memory = sections.get("memory", {})
    session_start = sections.get("session_start", {})
    auto_reconcile_raw = reconcile.get("auto_reconcile")

    return HookConfig(
        edition=sections.get("", {}).get("edition") or None,
        user_id=identity.get("user_id") or None,
        memgraph_url=memgraph.get("url") or _MEMGRAPH_DEFAULTS["url"],
        memgraph_user=memgraph.get("user", _MEMGRAPH_DEFAULTS["user"]),
        memgraph_password=memgraph.get("password", _MEMGRAPH_DEFAULTS["password"]),
        memgraph_database=memgraph.get("database") or _MEMGRAPH_DEFAULTS["database"],
        openai_api_key=llm.get("openai_api_key", _LLM_DEFAULTS["openai_api_key"]),
        anthropic_api_key=llm.get("anthropic_api_key", _LLM_DEFAULTS["anthropic_api_key"]),
        derive_model=llm.get("derive_model") or None,
        connectors=_parse_list(capture["connectors"]) if "connectors" in capture else None,
        capture_settings={key: value for key, value in capture.items() if key != "connectors"},
        redact_patterns=dict(sections.get("redact", {})),
        auto_reconcile=parse_bool_flag(auto_reconcile_raw) if auto_reconcile_raw is not None else None,
        embedding_model=recall.get("embedding_model") or None,
        recall_settings={key: value for key, value in recall.items() if key != "embedding_model"},
        ontology_path=ontology.get("path") or None,
        ontology_derive=ontology.get("derive") or None,
        extraction_settings=dict(extraction),
        github_token=github.get("token") or None,
        memory_backend=memory.get("backend") or None,
        session_resume=parse_bool_flag(session_start["resume"]) if session_start.get("resume") else None,
        present=frozenset(
            f"{section}.{key}" if section else key
            for section, keys in sections.items()
            for key, value in keys.items()
            if value
        ),
        problems=tuple(problems),
    )


def _parse_list(value: str) -> tuple[str, ...]:
    """A comma-separated value (``"a, b"``) as its non-blank items; the parser has no arrays."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


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


def _render_config(config: HookConfig) -> str:
    """Render the full config file content.

    ``[reconcile]`` is omitted while ``auto_reconcile`` is ``None`` so a re-read keeps "never configured"
    distinct from an explicit ``false``, and ``[session_start]`` likewise while ``session_resume`` is;
    ``[capture]`` likewise while ``connectors`` is and no other capture key is set; ``[redact]``, ``[recall]``,
    ``[ontology]``, ``[extraction]``, ``[github]`` and ``[memory]`` are omitted while they hold nothing.
    """
    user_id = config.user_id or ""
    lines = [
        "# Context Graph hook configuration",
        "# Generated by: memgraph-context-graph config set / bootstrap",
        f"# Location: {config_file()}",
        "",
        *([f'edition = "{config.edition}"', ""] if config.edition else []),
        # Enterprise takes the user and their database from the Memgraph login, so it writes neither.
        *(["[identity]", f'user_id = "{user_id}"', ""] if config.edition != "enterprise" else []),
        "[memgraph]",
        f'url = "{config.memgraph_url}"',
        f'user = "{config.memgraph_user}"',
        f'password = "{config.memgraph_password}"',
        *([f'database = "{config.memgraph_database}"'] if config.edition != "enterprise" else []),
        "",
        "[llm]",
        f'openai_api_key = "{config.openai_api_key}"',
        f'anthropic_api_key = "{config.anthropic_api_key}"',
        *([f'derive_model = "{config.derive_model}"'] if config.derive_model else []),
    ]
    capture = {
        **({"connectors": ", ".join(config.connectors)} if config.connectors is not None else {}),
        **config.capture_settings,
    }
    if capture:
        lines += ["", "[capture]", *(f'{key} = "{value}"' for key, value in capture.items())]
    if config.redact_patterns:
        # Single quotes: a pattern may hold a double quote, and the parser does no escape processing.
        lines += ["", "[redact]", *(f"{key} = '{value}'" for key, value in config.redact_patterns.items())]
    if config.auto_reconcile is not None:
        lines += ["", "[reconcile]", f"auto_reconcile = {'true' if config.auto_reconcile else 'false'}"]
    recall = (
        {"embedding_model": config.embedding_model, **config.recall_settings}
        if config.embedding_model
        else config.recall_settings
    )
    if recall:
        lines += ["", "[recall]", *(f'{key} = "{value}"' for key, value in recall.items())]
    ontology = {
        key: value for key, value in (("path", config.ontology_path), ("derive", config.ontology_derive)) if value
    }
    if ontology:
        lines += ["", "[ontology]", *(f'{key} = "{value}"' for key, value in ontology.items())]
    if config.extraction_settings:
        lines += ["", "[extraction]", *(f'{key} = "{value}"' for key, value in config.extraction_settings.items())]
    if config.github_token:
        lines += ["", "[github]", f'token = "{config.github_token}"']
    if config.memory_backend:
        lines += ["", "[memory]", f'backend = "{config.memory_backend}"']
    if config.session_resume is not None:
        lines += ["", "[session_start]", f"resume = {'true' if config.session_resume else 'false'}"]
    lines.append("")
    return "\n".join(lines)


def _reset_cache() -> None:
    """Reset the module cache (for testing)."""
    global _cached_config
    _cached_config = None
