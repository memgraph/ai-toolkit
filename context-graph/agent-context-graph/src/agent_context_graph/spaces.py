"""Spaces — where a user's context graph lives, and whose it is.

Every component asks a :class:`SpaceResolver` instead of reading the config
file's ``[identity]`` and ``[memgraph]`` sections itself. It answers three
questions: who the user is, which space their activity is written to (their
personal space), and which spaces recall may read.

There are two editions, chosen explicitly by the config file's top-level
``edition`` key and never detected from the server (map #501):

- **Community** (the default): one user, one Memgraph database, one space,
  exactly as the config file says. :class:`CommunitySpaces` is the whole of it;
  it makes no queries.
- **Enterprise**: many users on a shared Memgraph, each with a personal
  database and membership in shared ones, enforced by Memgraph's own access
  control. Implemented by the separate ``context-graph-spaces`` package, so
  Community never imports it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_context_graph.adapters._identity import HookConfig

EDITION_COMMUNITY = "community"
EDITION_ENTERPRISE = "enterprise"
EDITIONS = frozenset({EDITION_COMMUNITY, EDITION_ENTERPRISE})

#: The name of every user's own space, in both editions.
PERSONAL = "personal"


class SpaceError(Exception):
    """A space can't be resolved: unknown edition, missing package, or a space the user can't use."""


@dataclass(frozen=True)
class MemgraphTarget:
    """One Memgraph connection: where, as whom, and which database.

    ``database`` None means "the login's main database", which is how an
    Enterprise user reaches their personal space without naming it.
    """

    url: str
    user: str
    password: str
    database: str | None

    def env(self) -> dict[str, str]:
        """The target in memgraph-toolbox's ``MEMGRAPH_*`` env contract; an unnamed database is an empty string."""
        return {
            "MEMGRAPH_URL": self.url,
            "MEMGRAPH_USER": self.user,
            "MEMGRAPH_PASSWORD": self.password,
            "MEMGRAPH_DATABASE": self.database or "",
        }

    def client_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for a component's graph class (``SessionsGraph(**target.client_kwargs())``)."""
        return {"url": self.url, "username": self.user, "password": self.password, "database": self.database}


@dataclass(frozen=True)
class Space:
    """A place memory lives: the user's personal space or a shared one."""

    name: str
    target: MemgraphTarget

    @property
    def shared(self) -> bool:
        """Whether other users can read this space."""
        return self.name != PERSONAL


@runtime_checkable
class SpaceResolver(Protocol):
    """Who the user is and which spaces they write to and read from."""

    @property
    def edition(self) -> str: ...

    def user_id(self, payload: dict[str, Any] | None = None) -> str | None:
        """The user, or None when none is configured. *payload* is a hook payload some editions may honour."""
        ...

    def personal(self) -> Space:
        """The user's own space: hooks write all activity here."""
        ...

    def readable(self) -> list[Space]:
        """Every space recall reads, personal first."""
        ...

    def shared_space(self, name: str) -> Space:
        """The shared space *name*.

        Raises:
            SpaceError: the edition has no shared spaces, or the user isn't a member of *name*.
        """
        ...


class CommunitySpaces:
    """One user and one database, straight from the config file; makes no queries."""

    edition = EDITION_COMMUNITY

    def __init__(self, config: HookConfig) -> None:
        self._config = config
        self._personal = Space(
            PERSONAL,
            MemgraphTarget(
                url=config.memgraph_url,
                user=config.memgraph_user,
                password=config.memgraph_password,
                database=config.memgraph_database,
            ),
        )

    def user_id(self, payload: dict[str, Any] | None = None) -> str | None:
        """The payload's ``user_id`` if it has one, else ``[identity] user_id``."""
        if payload and (uid := payload.get("user_id")):
            return str(uid)
        return self._config.user_id

    def personal(self) -> Space:
        """The configured database."""
        return self._personal

    def readable(self) -> list[Space]:
        """Only the configured database."""
        return [self._personal]

    def shared_space(self, name: str) -> Space:
        """Community has no shared spaces.

        Raises:
            SpaceError: always.
        """
        raise SpaceError(f"Shared space {name!r} needs the Enterprise edition; Community has only a personal space.")


_cached: tuple[HookConfig, SpaceResolver] | None = None


def spaces_for(config: HookConfig) -> SpaceResolver:
    """The resolver for *config*'s edition, kept while the same config object is in use.

    Raises:
        SpaceError: the edition is unknown, or Enterprise without ``context-graph-spaces`` installed.
    """
    global _cached
    if _cached is not None and _cached[0] is config:
        return _cached[1]
    resolver = _build(config)
    _cached = (config, resolver)
    return resolver


def _build(config: HookConfig) -> SpaceResolver:
    edition = config.edition or EDITION_COMMUNITY
    if edition == EDITION_COMMUNITY:
        return CommunitySpaces(config)
    if edition == EDITION_ENTERPRISE:
        try:
            enterprise = importlib.import_module("context_graph_spaces")
        except ImportError as exc:
            raise SpaceError(
                "edition = enterprise needs the context-graph-spaces package: "
                "uv tool install 'agent-context-graph[mcp]' --with context-graph-spaces"
            ) from exc
        return enterprise.EnterpriseSpaces(config)
    raise SpaceError(f"Unknown edition {edition!r} in the config file; use one of: {', '.join(sorted(EDITIONS))}.")


def _reset_cache() -> None:
    """Forget the cached resolver (for tests)."""
    global _cached
    _cached = None
