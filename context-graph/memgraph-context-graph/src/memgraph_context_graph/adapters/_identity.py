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

Inside that one file, a profile (``[profiles.<name>]``) is another destination:
its own ``edition``, ``[memgraph]`` and ``[memory]``, used in place of the top
level's by the agents ``[agents]`` maps to it, so one agent's memory can go to
a team's Memgraph while the others stay where they are.  Which profile applies
is chosen per process, never written into a value (ADR 0008): the agent a hook
or MCP server runs for (:func:`select_agent`), a command's ``--profile``
(:func:`select_profile`), or :data:`PROFILE_ENV`, which a parent hands to the
processes it spawns alongside the file's path.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from memgraph_context_graph.adapters._text_file import write_text_keeping_newlines
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
from memgraph_context_graph.spaces import EDITION_ENTERPRISE

#: Selects which config *file* hooks read, never a value in it (ADR 0003): a parent
#: process hands the path to the session it spawns, so nothing is ambient.
CONFIG_PATH_ENV = "CONTEXT_GRAPH_CONFIG"

#: Selects which profile of that file applies, never a value in it (ADR 0008): handed down like
#: :data:`CONFIG_PATH_ENV`, so a process a hook spawns writes where the hook did.
PROFILE_ENV = "CONTEXT_GRAPH_PROFILE"

#: Names the agent an MCP server runs for, as ``mcp --agent`` does, so its ``[agents]`` mapping picks the
#: profile. A harness config that can't rely on the installed CLI knowing ``--agent`` sets this instead:
#: an older CLI ignores the variable, where it would refuse the flag and the server wouldn't start.
AGENT_ENV = "CONTEXT_GRAPH_AGENT"

#: The reserved profile name for the top level, for choosing it over an agent's mapping or :data:`PROFILE_ENV`.
DEFAULT_PROFILE = "default"

#: What a profile sets in place of the top level: where memory goes. How capture behaves (``[capture]``,
#: ``[redact]``, exclusions) stays shared, so pointing an agent elsewhere can't switch redaction off.
PROFILE_KEYS = (
    "edition",
    "memgraph.url",
    "memgraph.user",
    "memgraph.password",
    "memgraph.database",
    "memory.backend",
)
_PROFILE_SECTIONS = ("", "memgraph", "memory")
_PROFILE_FIELDS = {key.field: key.name for key in CONFIG_KEYS if key.name in PROFILE_KEYS and key.field}
_PROFILE_NAME = re.compile(r"[A-Za-z0-9_-]+")

#: The profile a command chose with ``--profile``; wins over :data:`PROFILE_ENV` and the agent's mapping.
_selected_profile: str | None = None
#: The agent this process runs for, whose ``[agents]`` mapping picks the profile.
_selected_agent: str | None = None

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
    #: The profile the values above were resolved through; None for the top level.
    profile: str | None = None
    #: Every ``[profiles.<name>]``, as written: name to ``{key name: value}`` over :data:`PROFILE_KEYS`.
    profiles: dict[str, dict[str, str]] = field(default_factory=dict)
    #: ``[agents]``: runtime to the profile it uses; an agent not listed uses the top level.
    agents: dict[str, str] = field(default_factory=dict)

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


def select_agent(runtime: str | None) -> None:
    """Resolve the config for *runtime*: through the profile ``[agents]`` maps it to, if any.

    A ``--profile`` (:func:`select_profile`) or :data:`PROFILE_ENV` still wins: both are an
    explicit choice made by whoever started this process.
    """
    global _selected_agent, _cached_config
    _selected_agent = runtime
    _cached_config = None


def select_profile(name: str | None) -> None:
    """Resolve the config through profile *name*, ahead of :data:`PROFILE_ENV` and any agent's mapping.

    :data:`DEFAULT_PROFILE` chooses the top level; None drops the choice. Writes go to the same place.

    Raises:
        ValueError: *name* is not a valid profile name.
    """
    global _selected_profile, _cached_config
    if name is not None and name != DEFAULT_PROFILE:
        check_profile_name(name)
    _selected_profile = name
    _cached_config = None


def check_profile_name(name: str) -> str:
    """*name*, if it can name a profile: letters, digits, ``-`` and ``_``, and not :data:`DEFAULT_PROFILE`.

    Raises:
        ValueError: it can't; the message says why.
    """
    if name == DEFAULT_PROFILE:
        raise ValueError(f"{DEFAULT_PROFILE!r} is reserved for the top level")
    if not _PROFILE_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a profile name: use letters, digits, - and _")
    return name


def map_agents(runtimes: list[str] | tuple[str, ...], profile: str) -> Path:
    """Point *runtimes* at *profile* in ``[agents]``; :data:`DEFAULT_PROFILE` maps them back to the top level.

    Returns the path written to.

    Raises:
        ValueError: *profile* is not a valid profile name.
    """
    if profile != DEFAULT_PROFILE:
        check_profile_name(profile)
    config = _read_config_file(overlay=False)
    agents = dict(config.agents)
    for runtime in runtimes:
        if profile == DEFAULT_PROFILE:
            agents.pop(runtime, None)
        else:
            agents[runtime] = profile
    return _write_config_file(_render_config(replace(config, agents=agents)))


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
    :data:`CONFIG_PATH_ENV` and the profile it names in :data:`PROFILE_ENV`: the parent hands down
    which file and which profile, never what is in them (ADR 0003, ADR 0008), and
    an ambient credential can't stand in for a key the file leaves empty (ADR 0002).
    """
    env = {key: value for key, value in os.environ.items() if key not in {*CONFIG_ENV_KEYS, *_CONFIG_ENV_ALIASES}}
    env[CONFIG_PATH_ENV] = str(config_file())
    # Always named, the top level included, so an ambient profile can't redirect the child.
    env[PROFILE_ENV] = load_config().profile or DEFAULT_PROFILE
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
    While a profile applies (:func:`select_agent`, :func:`select_profile`), the destination keys
    (:data:`PROFILE_KEYS`) go to that profile, creating it, and the rest to the top level.
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
    supplied = {key: value for key, value in updates.items() if value is not None}
    config = _read_config_file(overlay=False)
    if (profile := _profile_name(config, [], writing=True)) is not None:
        destination = {_PROFILE_FIELDS[key]: str(supplied.pop(key)) for key in list(supplied) if key in _PROFILE_FIELDS}
        config = _with_profile_values(config, profile, destination)
    return _write_config_file(_render_config(replace(config, **supplied)))


def set_config_value(name: str, raw: str) -> str:
    """Validate and write one key, as ``config set`` does; an empty value unsets an optional key.

    A destination key (:data:`PROFILE_KEYS`) goes to the profile that applies, if any, as in :func:`write_config`.

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
    config = _read_config_file(overlay=False)
    if key.name in PROFILE_KEYS and (profile := _profile_name(config, [], writing=True)) is not None:
        config = _with_profile_values(config, profile, {key.name: value})
    elif key.field is not None:
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
    write_text_keeping_newlines(path, content)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: the file holds credentials.
    _cached_config = None
    return path


def _read_config_file(*, overlay: bool = True) -> HookConfig:
    """Parse the config file, every key as :data:`CONFIG_KEYS` describes it.

    A missing file gives defaults. An unreadable or partly unparsable one also
    gives defaults for whatever it failed to set, never an exception (a hook
    must not crash on its config), and says so in :attr:`HookConfig.problems`.

    With *overlay*, the profile that applies replaces the top level's destination
    (:func:`_apply_profile`); without it, the values are the top level's as written,
    which is what a rewrite must start from.
    """
    problems: list[str] = []
    try:
        if not config_file().is_file():
            return HookConfig()
        sections = _parse_toml(config_file(), problems)
    except Exception as exc:
        return HookConfig(problems=(f"unreadable: {type(exc).__name__}: {exc}",))

    profiles = _pop_profiles(sections, problems)
    agents = sections.pop("agents", {})
    profile = None
    if overlay:
        profile = _profile_name(HookConfig(profiles=profiles, agents=agents), problems)
        if profile is not None:
            _apply_profile(sections, profiles.get(profile, {}))

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
        profile=profile,
        profiles=profiles,
        agents=agents,
    )


def _profile_name(config: HookConfig, problems: list[str], *, writing: bool = False) -> str | None:
    """The profile that applies in this process, by :func:`select_profile`, :data:`PROFILE_ENV`, then ``[agents]``.

    None means the top level. A profile named by the environment or ``[agents]`` that the file
    doesn't define is a problem: its values fall back to the top level's, which may be the wrong
    Memgraph. One chosen with ``--profile`` is not, since the command is about to create it, and
    neither is any when *writing*, since the write creates it.
    """
    if _selected_profile is not None:
        name, origin = _selected_profile, ""
    elif env := os.environ.get(PROFILE_ENV):
        name, origin = env, PROFILE_ENV
    elif _selected_agent is not None and _selected_agent in config.agents:
        name, origin = config.agents[_selected_agent], f"[agents] {_selected_agent}"
    else:
        return None
    if name == DEFAULT_PROFILE:
        return None
    if origin and not writing and name not in config.profiles:
        problems.append(f"{origin} names profile {name!r}, which the file doesn't define; using the top level")
        return None
    return name


def _pop_profiles(sections: dict[str, dict[str, str]], problems: list[str]) -> dict[str, dict[str, str]]:
    """Take every ``[profiles.<name>]`` and ``[profiles.<name>.<section>]`` out of *sections*, as profiles.

    A key outside :data:`PROFILE_KEYS`, or a section that doesn't name a profile, is a problem and is dropped.
    """
    profiles: dict[str, dict[str, str]] = {}
    for section in [name for name in sections if name == "profiles" or name.startswith("profiles.")]:
        values = sections.pop(section)
        name, _, subsection = section.partition(".")[2].partition(".")
        if not _PROFILE_NAME.fullmatch(name) or name == DEFAULT_PROFILE:
            problems.append(f"[{section}] doesn't name a profile (letters, digits, - and _) and was ignored")
            continue
        profile = profiles.setdefault(name, {})
        for key, value in values.items():
            full = f"{subsection}.{key}" if subsection else key
            if full in PROFILE_KEYS:
                profile[full] = value
            else:
                problems.append(f"[{section}] {key} can't differ per profile and was ignored")
    return profiles


def _apply_profile(sections: dict[str, dict[str, str]], profile: dict[str, str]) -> None:
    """Replace *sections*' destination with *profile*'s, in place.

    A profile is a whole destination: what it leaves out is the default, never the top level's,
    so a top-level ``memgraph.database`` can't leak into an Enterprise profile, whose login decides it.
    For the same reason an Enterprise profile drops ``identity.user_id``.
    """
    sections[""] = {key: value for key, value in sections.get("", {}).items() if key != "edition"}
    sections["memgraph"] = {}
    sections["memory"] = {}
    for name, value in profile.items():
        section, _, key = name.rpartition(".")
        sections[section][key] = value
    if sections[""].get("edition") == EDITION_ENTERPRISE:
        sections.get("identity", {}).pop("user_id", None)


def _with_profile_values(config: HookConfig, profile: str, values: dict[str, str]) -> HookConfig:
    """*config* with *values* (key name to value) written into *profile*, creating it; an empty value unsets a key."""
    merged = {**config.profiles.get(profile, {}), **values}
    return replace(
        config, profiles={**config.profiles, profile: {key: value for key, value in merged.items() if value}}
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
    lines += _render_profiles(config)
    lines.append("")
    return "\n".join(lines)


def _render_profiles(config: HookConfig) -> list[str]:
    """The ``[profiles.<name>]`` sections and ``[agents]``, as set; nothing when the file uses neither."""
    if not config.profiles and not config.agents:
        return []
    lines = [
        "",
        "# Profiles: other places an agent's memory can go. Each sets its own edition, [memgraph] and [memory]",
        "# in place of the ones above; everything else (capture, redaction, recall) is shared by every agent.",
    ]
    for name, values in config.profiles.items():
        for section in _PROFILE_SECTIONS:
            keys = [key for key in CONFIG_KEYS if key.name in values and key.section == section]
            if section and not keys:
                continue
            lines += ["", f"[profiles.{name}.{section}]" if section else f"[profiles.{name}]"]
            lines += [f"{key.key} = {key.format(values[key.name])}" for key in keys]
    lines += ["", "# The profile each agent uses; an agent not listed uses the top level.", "[agents]"]
    lines += [f'{runtime} = "{profile}"' for runtime, profile in config.agents.items()]
    return lines


def _reset_cache() -> None:
    """Reset the module cache and the profile selection (for testing)."""
    global _cached_config, _selected_agent, _selected_profile
    _cached_config = None
    _selected_agent = None
    _selected_profile = None
