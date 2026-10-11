"""The Enterprise edition's space resolver: the Memgraph login is the user.

agent-context-graph finds :class:`EnterpriseSpaces` through the
``agent_context_graph.editions`` entry point when the config file says
``edition = "enterprise"``. Map #501's rules, as code:

- Identity is the login (``[memgraph] user``). Memgraph authenticates it on
  every connection, so it can't be claimed. A configured
  ``identity.user_id`` or ``memgraph.database`` is refused rather than ignored,
  because either would silently disagree with the login. A hook payload's
  ``user_id`` is ignored.
- The personal space is the login's main database, reached by naming no
  database. Hooks therefore resolve their target without a single query.
- Shared spaces are read from the ``(:SpaceMembership)`` catalog that
  ``context-graph-spaces apply`` writes into the personal database. That is the
  one query, made only by long-lived readers (the MCP server's recall and
  memory tools), and cached until :meth:`EnterpriseSpaces.refresh`.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from agent_context_graph.spaces import EDITION_ENTERPRISE, PERSONAL, MemgraphTarget, Space, SpaceError

from .provision import MEMBERSHIP_LABEL

if TYPE_CHECKING:
    from agent_context_graph.adapters._identity import HookConfig

#: Config keys the login decides in Enterprise; writing them is an error.
_REFUSED_KEYS = {
    "identity.user_id": "the user is your Memgraph login",
    "memgraph.database": "your personal space is your login's main database",
}


class EnterpriseSpaces:
    """One login's personal space and the shared spaces it belongs to."""

    edition = EDITION_ENTERPRISE

    def __init__(self, config: HookConfig) -> None:
        """Check *config* is a valid Enterprise config.

        Raises:
            SpaceError: no ``[memgraph] user``, or a key the login decides is set.
        """
        for key, reason in _REFUSED_KEYS.items():
            if key in config.present:
                raise SpaceError(f"Remove {key} from the config file: in the Enterprise edition {reason}.")
        if not config.memgraph_user:
            raise SpaceError("The Enterprise edition needs [memgraph] user: your Memgraph login.")
        self._login = config.memgraph_user
        self._personal = Space(
            PERSONAL,
            MemgraphTarget(
                url=config.memgraph_url, user=config.memgraph_user, password=config.memgraph_password, database=None
            ),
        )
        self._lock = threading.Lock()
        self._shared: dict[str, Space] | None = None

    def user_id(self, payload: dict[str, Any] | None = None) -> str | None:
        """The login; a payload can't name anyone else."""
        return self._login

    def personal(self) -> Space:
        """The login's main database."""
        return self._personal

    def readable(self) -> list[Space]:
        """The personal space, then the shared spaces the catalog lists, in name order."""
        return [self._personal, *sorted(self._shared_spaces().values(), key=lambda space: space.name)]

    def shared_space(self, name: str) -> Space:
        """The shared space *name*, if the login belongs to it.

        Raises:
            SpaceError: the catalog doesn't list *name*.
        """
        spaces = self._shared_spaces()
        if name not in spaces:
            known = ", ".join(sorted(spaces)) or "none"
            raise SpaceError(f"You aren't a member of a shared space named {name!r} (yours: {known}).")
        return spaces[name]

    def refresh(self) -> None:
        """Forget the cached catalog, e.g. after Memgraph refused a space it listed."""
        with self._lock:
            self._shared = None

    def _shared_spaces(self) -> dict[str, Space]:
        with self._lock:
            if self._shared is None:
                self._shared = self._read_catalog()
            return self._shared

    def _read_catalog(self) -> dict[str, Space]:
        from memgraph_toolbox.api.memgraph import Memgraph

        target = self._personal.target
        client = Memgraph(**target.client_kwargs(), user_agent="context-graph-spaces")
        try:
            rows = client.query(f"MATCH (m:{MEMBERSHIP_LABEL}) RETURN m.space AS space, m.database AS database")
        finally:
            client.close()
        return {
            row["space"]: Space(
                row["space"],
                MemgraphTarget(url=target.url, user=target.user, password=target.password, database=row["database"]),
            )
            for row in rows
        }
