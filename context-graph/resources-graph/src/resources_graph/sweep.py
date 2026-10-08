"""The Sweep: turns pending Touches into Resources, out of band.

Never runs inside a hook. A FETCHED Touch means the agent really fetched the
Address — the model judged memory stale enough to go to GitHub — so a stored
Address is revalidated: one cheap check of ``updatedAt`` (a Listing: a light
index of its members), and only what changed is fetched again at full depth.
A PROMPTED Touch of a stored Address just resolves to it, and a Cache Read
never comes here at all. There is no TTL and no background crawl.

When GitHub rate-limits a fetch, that Touch is kept as Unresolved
(``rate_limited``) and the Sweep stops: the rest stay pending, and the next
Sweep retries them all.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .address import Address
from .core import RATE_LIMITED
from .github import UnresolvedError
from .models import FETCHED, SweepReport

if TYPE_CHECKING:
    from collections.abc import Callable

    from .core import ResourcesGraph
    from .github import Source


@dataclass
class _Counts:
    fetched: int = 0  # full-depth fetches: an item, a repository, a new Listing's pages
    checked: int = 0  # cheap revalidations: an item's updatedAt, a Listing's light index
    unchanged: int = 0  # stored Resources confirmed current without refetching


def sweep(
    graph: ResourcesGraph,
    source: Source,
    *,
    limit: int | None = None,
    log: Callable[[str], None] = lambda _line: None,
) -> SweepReport:
    """Resolve up to ``limit`` pending Touches, oldest first.

    Each Address (a Listing: each Address and limit) is fetched at most once per
    Sweep, however many Touches name it.

    Raises:
        GitHubAuthError: the token was rejected; Touches stay pending.
    """
    # (Touched Address key, limit) -> (stored key, None) or (None, Unresolved reason).
    outcomes: dict[tuple[str, int | None], tuple[str | None, str | None]] = {}
    counts = _Counts()
    resolved = 0
    unresolved: Counter[str] = Counter()
    for touch in graph.pending_touches(limit):
        key = touch["address"]
        once = (key, touch.get("limit"))
        if once not in outcomes:
            address = Address.from_key(key, limit=touch.get("limit"))
            if touch["provenance"] != FETCHED and _stored(graph, address):
                outcomes[once] = (key, None)
            else:
                try:
                    outcomes[once] = (_fetch(graph, source, address, counts), None)
                except UnresolvedError as exc:
                    outcomes[once] = (None, exc.reason)
        stored, reason = outcomes[once]
        if stored is not None:
            graph.resolve_touch(touch["touch_id"], stored)
            resolved += 1
            log(f"resolved   {key}" + ("" if stored == key else f" -> {stored}"))
            continue
        graph.unresolve_touch(touch["touch_id"], reason or "not_found")
        unresolved[reason or "not_found"] += 1
        log(f"unresolved {key}: {reason}")
        if reason == RATE_LIMITED:
            log("rate-limited by GitHub: stopping; the rest stay pending for the next Sweep")
            break
    return SweepReport(
        resolved=resolved,
        unresolved=dict(unresolved),
        fetched=counts.fetched,
        linked=graph.link_touches(),
        checked=counts.checked,
        unchanged=counts.unchanged,
    )


def _stored(graph: ResourcesGraph, address: Address) -> bool:
    return graph.has_listing(address) if address.kind == "listing" else graph.has_resource(address)


def _fetch(graph: ResourcesGraph, source: Source, address: Address, counts: _Counts) -> str:
    """Fetch (or revalidate) and store ``address``; its Repository too, when missing.

    Returns the stored key, which differs from ``address`` after a rename or transfer.
    """
    if address.kind == "repo":
        counts.fetched += 1
        return graph.store_repository(source.fetch_repository(address))
    if address.kind == "listing":
        return _fetch_listing(graph, source, address, counts)
    node_id = graph.stored_node_id(address)
    if node_id is not None:
        counts.checked += 1
        fresh = source.item_freshness(address)
        stored = graph.stored_freshness([node_id]).get(node_id)
        if fresh["id"] == node_id and stored and stored["updated_at"] == fresh["updatedAt"]:
            graph.confirm_fresh([node_id])
            counts.unchanged += 1
            return address.key
    item = source.fetch_item(address)
    counts.fetched += 1
    _ensure_repository(graph, source, item["repository"]["nameWithOwner"], counts)
    stored_key = graph.store_item(item)
    if stored_key != address.key:
        graph.mark_moved(address.key, stored_key)
    return stored_key


def _fetch_listing(graph: ResourcesGraph, source: Source, address: Address, counts: _Counts) -> str:
    _ensure_repository(graph, source, f"{address.owner}/{address.repo}", counts)
    if not graph.has_listing_key(address):
        total, members = source.fetch_listing(address)
        counts.fetched += 1
        return graph.store_listing(address, total, [graph.store_item(item) for item in members])
    total, index = source.listing_index(address)
    counts.checked += 1
    stored = graph.stored_freshness([member["id"] for member in index])
    keys, unchanged = [], []
    for member in index:
        known = stored.get(member["id"])
        if known and known["updated_at"] == member["updatedAt"] and known["address"]:
            keys.append(known["address"])
            unchanged.append(member["id"])
            continue
        owner, _, name = member["repository"]["nameWithOwner"].partition("/")
        keys.append(graph.store_item(source.fetch_item(Address("item", owner, name, member["number"]))))
        counts.fetched += 1
    graph.confirm_fresh(unchanged)
    counts.unchanged += len(unchanged)
    return graph.store_listing(address, total, keys)


def _ensure_repository(graph: ResourcesGraph, source: Source, name_with_owner: str, counts: _Counts) -> None:
    """Store the repository unless it already is, so no content-less Repository is ever created."""
    owner, _, name = name_with_owner.partition("/")
    repository = Address("repo", owner, name)
    if not graph.has_resource(repository):
        graph.store_repository(source.fetch_repository(repository))
        counts.fetched += 1
