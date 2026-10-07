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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Points hooks at a different config *file*. Not a way to supply configuration
#: values -- ADR 0002 removed those from the environment deliberately, because
#: hook subprocesses do not source shell profiles and relying on ambient env
#: caused `doctor` and hooks to disagree about what was configured.
#:
#: This is the opposite situation: a parent process that spawns a session hands
#: the path down explicitly to its own child, so nothing is ambient and nothing
#: can drift. It exists because the config file is otherwise a single global,
#: which means pointing ONE session at a different Memgraph silently redirects
#: every other Claude Code session running on the machine -- observed while
#: building the eval gold slice, where an unrelated session's activity landed
#: in the graph under test.
CONFIG_PATH_ENV = "CONTEXT_GRAPH_CONFIG"

_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "context-graph"


def config_file() -> Path:
    """The config file hooks read, honouring :data:`CONFIG_PATH_ENV`."""
    override = os.environ.get(CONFIG_PATH_ENV)
    return Path(override) if override else _DEFAULT_CONFIG_DIR / "config.toml"


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

_RECONCILE_DEFAULTS = {
    "auto_reconcile": None,
}

#: Shared truthy/falsy vocabulary for TOML/CLI boolean flags — kept here so
#: cli.py can validate `config set` input against the same set parse_bool_flag
#: accepts, instead of redefining it inline.
TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
FALSY_VALUES = frozenset({"0", "false", "no", "off"})

_cached_config: HookConfig | None = None


@dataclass(frozen=True)
class HookConfig:
    """Parsed hook configuration from the config file."""

    user_id: str | None = None
    memgraph_url: str = _MEMGRAPH_DEFAULTS["url"]
    memgraph_user: str = _MEMGRAPH_DEFAULTS["user"]
    memgraph_password: str = _MEMGRAPH_DEFAULTS["password"]
    memgraph_database: str = _MEMGRAPH_DEFAULTS["database"]
    openai_api_key: str = _LLM_DEFAULTS["openai_api_key"]
    anthropic_api_key: str = _LLM_DEFAULTS["anthropic_api_key"]
    auto_reconcile: bool | None = _RECONCILE_DEFAULTS["auto_reconcile"]
    #: The model recall embeds with; None means the consumer's own default.
    embedding_model: str | None = None
    #: Every other ``[recall]`` key (lanes and widths), as written: the recall
    #: tool validates them, this module only keeps them across rewrites.
    recall_settings: dict[str, str] = field(default_factory=dict)
    #: ``[ontology] path``: a schema file to extract this user's sessions under.
    ontology_path: str | None = None
    #: ``[ontology] derive``: "extend" (the default) or "off"; see sessions-graph's ontology module.
    ontology_derive: str | None = None
    #: ``[github] token``: what resources-graph's Sweep fetches public GitHub content with.
    github_token: str | None = None


def load_config() -> HookConfig:
    """Load hook configuration from the config file.

    Returns a HookConfig with defaults for any missing values.
    Caches the result for the lifetime of the process.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    config = _read_config_file()
    _cached_config = config
    return config


def resolve_user_id(payload: dict[str, Any]) -> str | None:
    """Resolve user identity for hook subprocesses.

    Resolution order:
    1. ``user_id`` field in the hook payload (forward-compat).
    2. Config file ``[identity] user_id``.
    """
    if uid := _string_or_none(payload.get("user_id")):
        return uid
    return load_config().user_id


def resolve_memgraph_env(
    *,
    url: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> dict[str, str]:
    """Resolve Memgraph connection settings for hook subprocesses.

    Resolution order per setting: explicit arg (CLI flag) > config file > default.
    Returns a dict with keys matching memgraph-toolbox's env contract.
    """
    config = load_config()
    return {
        "MEMGRAPH_URL": url if url is not None else config.memgraph_url,
        "MEMGRAPH_USER": user if user is not None else config.memgraph_user,
        "MEMGRAPH_PASSWORD": password if password is not None else config.memgraph_password,
        "MEMGRAPH_DATABASE": database if database is not None else config.memgraph_database,
    }


def resolve_llm_env() -> dict[str, str]:
    """Resolve LLM API key settings for hook subprocesses.

    Config file only — mirrors :func:`resolve_memgraph_env`, but there is no
    CLI-flag override path for these since nothing resolves them from argparse.
    Values are empty strings when not configured; callers should treat an
    empty value as "not configured" rather than overlaying it onto a child
    process's environment.
    """
    config = load_config()
    return {
        "OPENAI_API_KEY": config.openai_api_key,
        "ANTHROPIC_API_KEY": config.anthropic_api_key,
    }


def resolve_auto_reconcile() -> bool:
    """Resolve whether sessions-graph should auto-trigger reconciliation on SESSION_END.

    Config file only — mirrors :func:`resolve_llm_env`. The config-file value
    is ``None`` when never configured; per ADR 0002 (config-file-only-hook-
    resolution) there is no ambient env var fallback to preserve here, so
    "never configured" collapses to ``False``, matching
    ``SessionsGraphConnector``'s own default. Off by default given LightRAG
    entity extraction's LLM cost; opt in with
    ``agent-context-graph config set reconcile.auto_reconcile true``.
    """
    return bool(load_config().auto_reconcile)


def resolve_embedding_model() -> str | None:
    """The configured ``[recall] embedding_model``, or None when unset.

    Config file only, like :func:`resolve_auto_reconcile`. None leaves the
    choice to sessions-graph's own default rather than duplicating it here.
    """
    return load_config().embedding_model


def resolve_ontology() -> tuple[str | None, str | None]:
    """The configured ``[ontology]`` (path, derive), each None when unset.

    Config file only, like :func:`resolve_auto_reconcile`. sessions-graph owns
    what the values mean and their defaults.
    """
    config = load_config()
    return config.ontology_path, config.ontology_derive


def parse_bool_flag(value: str | bool) -> bool:
    """Parse a TOML/CLI boolean flag string. Truthy: "1"/"true"/"yes"/"on" (case-insensitive).

    Passing an actual ``bool`` through unchanged guards against a future
    swap to a real TOML parser (which would hand back native booleans).
    """
    if isinstance(value, bool):
        return value
    return value.strip().lower() in TRUTHY_VALUES


def write_config(
    *,
    user_id: str | None = None,
    memgraph_url: str | None = None,
    memgraph_user: str | None = None,
    memgraph_password: str | None = None,
    memgraph_database: str | None = None,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    auto_reconcile: bool | None = None,
    embedding_model: str | None = None,
    ontology_path: str | None = None,
    ontology_derive: str | None = None,
    github_token: str | None = None,
) -> Path:
    """Write or update the config file. Returns the path written to.

    Only updates supplied values; preserves existing values for unspecified keys.
    Creates the file with 0600 permissions if it does not exist.
    """
    global _cached_config

    # Read existing config as base.
    existing = _read_config_file()

    final_user_id = user_id if user_id is not None else existing.user_id
    final_url = memgraph_url if memgraph_url is not None else existing.memgraph_url
    final_user = memgraph_user if memgraph_user is not None else existing.memgraph_user
    final_password = memgraph_password if memgraph_password is not None else existing.memgraph_password
    final_database = memgraph_database if memgraph_database is not None else existing.memgraph_database
    final_openai_api_key = openai_api_key if openai_api_key is not None else existing.openai_api_key
    final_anthropic_api_key = anthropic_api_key if anthropic_api_key is not None else existing.anthropic_api_key
    final_auto_reconcile = auto_reconcile if auto_reconcile is not None else existing.auto_reconcile
    final_embedding_model = embedding_model if embedding_model is not None else existing.embedding_model
    final_ontology_path = ontology_path if ontology_path is not None else existing.ontology_path
    final_ontology_derive = ontology_derive if ontology_derive is not None else existing.ontology_derive
    final_github_token = github_token if github_token is not None else existing.github_token

    content = _render_config(
        user_id=final_user_id or "",
        url=final_url,
        user=final_user,
        password=final_password,
        database=final_database,
        openai_api_key=final_openai_api_key,
        anthropic_api_key=final_anthropic_api_key,
        auto_reconcile=final_auto_reconcile,
        embedding_model=final_embedding_model,
        recall_settings=existing.recall_settings,
        ontology_path=final_ontology_path,
        ontology_derive=final_ontology_derive,
        github_token=final_github_token,
    )

    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    # Invalidate cache.
    _cached_config = None
    return config_file()


def write_full_config(
    *,
    user_id: str = "",
    memgraph_url: str = _MEMGRAPH_DEFAULTS["url"],
    memgraph_user: str = _MEMGRAPH_DEFAULTS["user"],
    memgraph_password: str = _MEMGRAPH_DEFAULTS["password"],
    memgraph_database: str = _MEMGRAPH_DEFAULTS["database"],
    openai_api_key: str = _LLM_DEFAULTS["openai_api_key"],
    anthropic_api_key: str = _LLM_DEFAULTS["anthropic_api_key"],
    auto_reconcile: bool | None = _RECONCILE_DEFAULTS["auto_reconcile"],
) -> Path:
    """Write a complete config file with all sections (used by bootstrap).

    Overwrites every section except ``[reconcile]``, ``[recall]``, ``[ontology]`` and ``[github]``: unlike identity/Memgraph/LLM
    settings, ``auto_reconcile`` has no legitimate ambient-env source for
    ``bootstrap`` to capture (nobody has ``SESSIONS_GRAPH_AUTO_RECONCILE``
    exported for an unrelated reason the way they might already have
    ``OPENAI_API_KEY``/`MEMGRAPH_PASSWORD` set) — it is only ever set via
    ``config set reconcile.auto_reconcile``. Re-running bootstrap must not
    silently revert it to off, so ``auto_reconcile`` is preserved from the
    existing file unless explicitly given here. ``[recall]``, ``[ontology]`` and
    ``[github]`` are preserved the same way: they are only ever set via ``config set`` or by
    editing the file.
    """
    global _cached_config

    existing = _read_config_file()
    final_auto_reconcile = auto_reconcile if auto_reconcile is not None else existing.auto_reconcile

    content = _render_config(
        user_id=user_id,
        url=memgraph_url,
        user=memgraph_user,
        password=memgraph_password,
        database=memgraph_database,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        auto_reconcile=final_auto_reconcile,
        embedding_model=existing.embedding_model,
        recall_settings=existing.recall_settings,
        ontology_path=existing.ontology_path,
        ontology_derive=existing.ontology_derive,
        github_token=existing.github_token,
    )

    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    _cached_config = None
    return config_file()


def config_file_path() -> Path:
    """Return the config file path (for display purposes)."""
    return config_file()


def config_dir_path() -> Path:
    """Return the directory holding the config file actually in use."""
    return config_file().parent


# --- Internal helpers ---


def _read_config_file() -> HookConfig:
    """Parse the config file. Returns defaults if file is missing or malformed."""
    if not config_file().is_file():
        return HookConfig()

    try:
        sections = _parse_toml(config_file())
    except Exception:
        return HookConfig()

    identity = sections.get("identity", {})
    memgraph = sections.get("memgraph", {})
    llm = sections.get("llm", {})
    reconcile = sections.get("reconcile", {})
    recall = sections.get("recall", {})
    ontology = sections.get("ontology", {})
    github = sections.get("github", {})
    auto_reconcile_raw = reconcile.get("auto_reconcile")

    return HookConfig(
        user_id=identity.get("user_id") or None,
        memgraph_url=memgraph.get("url") or _MEMGRAPH_DEFAULTS["url"],
        memgraph_user=memgraph.get("user", _MEMGRAPH_DEFAULTS["user"]),
        memgraph_password=memgraph.get("password", _MEMGRAPH_DEFAULTS["password"]),
        memgraph_database=memgraph.get("database") or _MEMGRAPH_DEFAULTS["database"],
        openai_api_key=llm.get("openai_api_key", _LLM_DEFAULTS["openai_api_key"]),
        anthropic_api_key=llm.get("anthropic_api_key", _LLM_DEFAULTS["anthropic_api_key"]),
        auto_reconcile=parse_bool_flag(auto_reconcile_raw) if auto_reconcile_raw is not None else None,
        embedding_model=recall.get("embedding_model") or None,
        recall_settings={key: value for key, value in recall.items() if key != "embedding_model"},
        ontology_path=ontology.get("path") or None,
        ontology_derive=ontology.get("derive") or None,
        github_token=github.get("token") or None,
    )


def _parse_toml(path: Path) -> dict[str, dict[str, str]]:
    """Minimal TOML parser — handles [section] and key = "value" pairs.

    Only supports string values (quoted or bare). Sufficient for our config shape.
    """
    sections: dict[str, dict[str, str]] = {}
    current_section: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            sections.setdefault(current_section, {})
            continue
        if current_section is not None and "=" in stripped:
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip()
            if value[:1] not in {'"', "'"} and "#" in value:
                # Bare (unquoted) value with a trailing comment, e.g. `auto_reconcile = true  # off for now`.
                value = value.partition("#")[0].strip()
            value = value.strip('"').strip("'")
            sections[current_section][key] = value

    return sections


def _render_config(
    *,
    user_id: str,
    url: str,
    user: str,
    password: str,
    database: str,
    openai_api_key: str,
    anthropic_api_key: str,
    auto_reconcile: bool | None,
    embedding_model: str | None = None,
    recall_settings: dict[str, str] | None = None,
    ontology_path: str | None = None,
    ontology_derive: str | None = None,
    github_token: str | None = None,
) -> str:
    """Render the full config file content.

    The ``[reconcile]`` section is omitted entirely when ``auto_reconcile`` is
    ``None`` (never configured), so a fresh read of the file resolves it back
    to ``None`` rather than a concrete ``false`` — see
    :func:`resolve_auto_reconcile` for why that distinction matters.
    ``[recall]``, ``[ontology]`` and ``[github]`` are likewise omitted while they hold nothing.
    """
    lines = [
        "# Context Graph hook configuration",
        "# Generated by: agent-context-graph config set / bootstrap",
        f"# Location: {config_file()}",
        "",
        "[identity]",
        f'user_id = "{user_id}"',
        "",
        "[memgraph]",
        f'url = "{url}"',
        f'user = "{user}"',
        f'password = "{password}"',
        f'database = "{database}"',
        "",
        "[llm]",
        f'openai_api_key = "{openai_api_key}"',
        f'anthropic_api_key = "{anthropic_api_key}"',
    ]
    if auto_reconcile is not None:
        lines += ["", "[reconcile]", f"auto_reconcile = {'true' if auto_reconcile else 'false'}"]
    recall = {"embedding_model": embedding_model, **(recall_settings or {})} if embedding_model else recall_settings
    if recall:
        lines += ["", "[recall]", *(f'{key} = "{value}"' for key, value in recall.items())]
    ontology = {key: value for key, value in (("path", ontology_path), ("derive", ontology_derive)) if value}
    if ontology:
        lines += ["", "[ontology]", *(f'{key} = "{value}"' for key, value in ontology.items())]
    if github_token:
        lines += ["", "[github]", f'token = "{github_token}"']
    lines.append("")
    return "\n".join(lines)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value)
    return s if s else None


def _reset_cache() -> None:
    """Reset the module cache (for testing)."""
    global _cached_config
    _cached_config = None
