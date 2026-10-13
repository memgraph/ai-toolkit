"""Where the viewer reads from: the user's spaces, resolved exactly as the hooks resolve them."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from memgraph_context_graph.adapters._identity import load_config
from memgraph_context_graph.spaces import Space, spaces_for
from memgraph_toolbox.api.memgraph import Memgraph

#: Seconds to wait for Memgraph before a poll gives up; short, so a dead server shows up as an error, not a hang.
CONNECTION_TIMEOUT = 5.0


@dataclass(frozen=True)
class Identity:
    """Who the viewer is looking as, and which spaces it may show."""

    edition: str
    user_id: str | None
    spaces: tuple[Space, ...]


def resolve_identity() -> Identity:
    """Resolve the configured edition's user and readable spaces from the config file.

    Community gives ``[identity] user_id`` and one personal space; Enterprise makes the Memgraph login the
    user and adds every shared space the login is a member of.

    Raises:
        memgraph_context_graph.spaces.SpaceError: the edition is unknown or its package isn't installed.
    """
    resolver = spaces_for(load_config())
    return Identity(edition=resolver.edition, user_id=resolver.user_id(), spaces=tuple(resolver.readable()))


def describe(space: Space) -> str:
    """``host:port/database`` for the status line; never the login or password."""
    parts = urlsplit(space.target.url)
    host = parts.hostname or space.target.url
    where = f"{host}:{parts.port}" if parts.port else host
    return f"{where}/{space.target.database or 'main'}"


class Reader:
    """A lazily connected, read-only query runner for one space.

    Connecting happens on the first query, inside the polling worker, so a Memgraph that is down when the
    viewer starts is reported as an error on the view and retried on the next poll.
    """

    def __init__(self, space: Space) -> None:
        self.space = space
        self._client: Memgraph | None = None
        self._lock = threading.Lock()

    def rows(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Run *query* with *params* and return its rows as plain dicts (temporals as ISO strings)."""
        return self._connected().query(query, params)

    def close(self) -> None:
        """Close the connection, if one was opened."""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _connected(self) -> Memgraph:
        with self._lock:
            if self._client is None:
                kwargs = self.space.target.client_kwargs()
                driver_config = {**kwargs.pop("driver_config", {}), "connection_timeout": CONNECTION_TIMEOUT}
                self._client = Memgraph(**kwargs, driver_config=driver_config, user_agent="context-graph-viewer")
            return self._client
