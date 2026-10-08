"""The Sweep: turns pending Touches into Resources, out of band.

Never runs inside a hook. A FETCHED Touch means the agent really fetched the
Address, so the Sweep fetches it again even if it is stored — that is how
memory gets refreshed, only when the model judged it stale enough to go to
GitHub. A PROMPTED Touch of a stored Address just resolves to it.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from .address import Address
from .github import UnresolvedError
from .models import FETCHED, SweepReport

if TYPE_CHECKING:
    from collections.abc import Callable

    from .core import ResourcesGraph
    from .github import GitHubSource


def sweep(
    graph: ResourcesGraph,
    source: GitHubSource,
    *,
    limit: int | None = None,
    log: Callable[[str], None] = lambda _line: None,
) -> SweepReport:
    """Resolve up to ``limit`` pending Touches, oldest first.

    Each Address is fetched at most once per Sweep, however many Touches name it.

    Raises:
        GitHubAuthError: the token was rejected; Touches stay pending.
    """
    # Touched Address key -> (stored Address key, None) or (None, UnresolvedError reason).
    outcomes: dict[str, tuple[str | None, str | None]] = {}
    resolved, fetched = 0, 0
    unresolved: Counter[str] = Counter()
    for touch in graph.pending_touches(limit):
        key = touch["address"]
        if key not in outcomes:
            address = Address.from_key(key)
            if touch["provenance"] != FETCHED and graph.has_resource(address):
                outcomes[key] = (key, None)
            else:
                try:
                    stored, fetches = _fetch(graph, source, address)
                    fetched += fetches
                    outcomes[key] = (stored, None)
                except UnresolvedError as exc:
                    outcomes[key] = (None, exc.reason)
        stored, reason = outcomes[key]
        if stored is not None:
            graph.resolve_touch(touch["touch_id"], stored)
            resolved += 1
            log(f"resolved   {key}" + ("" if stored == key else f" -> {stored}"))
        else:
            graph.unresolve_touch(touch["touch_id"], reason or "not_found")
            unresolved[reason or "not_found"] += 1
            log(f"unresolved {key}: {reason}")
    return SweepReport(resolved=resolved, unresolved=dict(unresolved), fetched=fetched)


def _fetch(graph: ResourcesGraph, source: GitHubSource, address: Address) -> tuple[str, int]:
    """Fetch and store ``address`` (and its Repository, when missing).

    Returns the stored Address key — which differs from ``address`` after a
    rename or transfer — and how many fetches it took.
    """
    if address.kind == "repo":
        return graph.store_repository(source.fetch_repository(address)), 1
    item = source.fetch_item(address)
    fetches = 1
    owner, _, name = item["repository"]["nameWithOwner"].partition("/")
    repository = Address("repo", owner, name)
    if not graph.has_resource(repository):
        graph.store_repository(source.fetch_repository(repository))
        fetches += 1
    return graph.store_item(item), fetches
