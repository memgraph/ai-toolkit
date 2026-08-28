"""Hook Configuration — persistent config for hook subprocesses.

Agent runtimes (Claude Code, Codex) spawn hook commands as non-interactive
subprocesses that do not inherit shell profile environment variables.  This
module provides a config-file-only resolution path so hook subprocesses (and
subprocesses they in turn spawn, e.g. sessions-graph's detached
reconciliation process) can reliably access identity, Memgraph connection,
LLM API key, and reconciliation opt-in settings.

Resolution order (per ADR 0002):
  CLI flag > config file > hardcoded default

Environment variables are **not** consulted at hook runtime.  They are only
used as a write-time input during ``bootstrap`` or ``config set``.

Config file location: ``~/.config/context-graph/config.toml``
"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path.home() / ".config" / "context-graph"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"

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


def resolve_auto_reconcile() -> bool | None:
    """Resolve whether sessions-graph should auto-trigger reconciliation on SESSION_END.

    Config file only — mirrors :func:`resolve_llm_env`. Returns ``None`` when
    never configured (the config-file key is absent), distinct from an
    explicit ``False`` — callers (``agent_context_graph.hooks.runner``) pass
    this straight through to ``SessionsGraphConnector(auto_reconcile=...)``,
    whose own resolution order is explicit arg > ``SESSIONS_GRAPH_AUTO_RECONCILE``
    env var > default off. Collapsing "never configured" to ``False`` here
    would short-circuit that env var fallback for anyone still relying on it.
    Off by default given LightRAG entity extraction's LLM cost; opt in with
    ``agent-context-graph config set reconcile.auto_reconcile true``.
    """
    return load_config().auto_reconcile


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

    content = _render_config(
        user_id=final_user_id or "",
        url=final_url,
        user=final_user,
        password=final_password,
        database=final_database,
        openai_api_key=final_openai_api_key,
        anthropic_api_key=final_anthropic_api_key,
        auto_reconcile=final_auto_reconcile,
    )

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(content, encoding="utf-8")
    _CONFIG_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    # Invalidate cache.
    _cached_config = None
    return _CONFIG_FILE


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

    Overwrites every section except ``[reconcile]``: unlike identity/Memgraph/LLM
    settings, ``auto_reconcile`` has no legitimate ambient-env source for
    ``bootstrap`` to capture (nobody has ``SESSIONS_GRAPH_AUTO_RECONCILE``
    exported for an unrelated reason the way they might already have
    ``OPENAI_API_KEY``/`MEMGRAPH_PASSWORD` set) — it is only ever set via
    ``config set reconcile.auto_reconcile``. Re-running bootstrap must not
    silently revert it to off, so ``auto_reconcile`` is preserved from the
    existing file unless explicitly given here.
    """
    global _cached_config

    final_auto_reconcile = auto_reconcile if auto_reconcile is not None else _read_config_file().auto_reconcile

    content = _render_config(
        user_id=user_id,
        url=memgraph_url,
        user=memgraph_user,
        password=memgraph_password,
        database=memgraph_database,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        auto_reconcile=final_auto_reconcile,
    )

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(content, encoding="utf-8")
    _CONFIG_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    _cached_config = None
    return _CONFIG_FILE


def config_file_path() -> Path:
    """Return the config file path (for display purposes)."""
    return _CONFIG_FILE


def config_dir_path() -> Path:
    """Return the config directory path."""
    return _CONFIG_DIR


# --- Internal helpers ---


def _read_config_file() -> HookConfig:
    """Parse the config file. Returns defaults if file is missing or malformed."""
    if not _CONFIG_FILE.is_file():
        return HookConfig()

    try:
        sections = _parse_toml(_CONFIG_FILE)
    except Exception:
        return HookConfig()

    identity = sections.get("identity", {})
    memgraph = sections.get("memgraph", {})
    llm = sections.get("llm", {})
    reconcile = sections.get("reconcile", {})
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
) -> str:
    """Render the full config file content.

    The ``[reconcile]`` section is omitted entirely when ``auto_reconcile`` is
    ``None`` (never configured), so a fresh read of the file resolves it back
    to ``None`` rather than a concrete ``false`` — see
    :func:`resolve_auto_reconcile` for why that distinction matters.
    """
    lines = [
        "# Context Graph hook configuration",
        "# Generated by: agent-context-graph config set / bootstrap",
        f"# Location: {_CONFIG_FILE}",
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
