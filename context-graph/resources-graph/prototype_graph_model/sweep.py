"""PROTOTYPE — throwaway. The out-of-band Sweep: pending Touches -> Resources/Listings."""

from __future__ import annotations

import time

import github
from store import Store, address_of


def sweep(store: Store, log=print) -> None:
    for row in store.pending():
        t = row["t"]
        addr = address_of(t)
        started = time.monotonic()
        try:
            if addr.kind == "repo":
                key = store.upsert_repo(github.fetch_repo(addr.owner, addr.repo))
                log(f"repo     {key}")
            elif addr.kind == "item":
                key, prev = store.upsert_item(github.fetch_item(addr.owner, addr.repo, addr.number))
                change = "first contact" if prev is None else ("unchanged" if prev == _updated(store, key) else "CHANGED")
                log(f"item     {key}  ({change})")
            else:
                store.upsert_repo(github.fetch_repo(addr.owner, addr.repo))
                members, total = [], 0
                for total, page in github.fetch_listing(
                    addr.owner, addr.repo, addr.item_kind, addr.filter_dict(), addr.query, addr.limit
                ):
                    members += [store.upsert_item(n)[0] for n in page]
                    log(f"listing  {addr.key}  {len(members)}/{total if addr.limit is None else min(total, addr.limit)}")
                key = store.upsert_listing(addr, total, members)
            store.resolve_touch(t["touch_id"], key)
        except github.Unresolved as e:
            store.unresolve_touch(t["touch_id"], e.reason)
            log(f"UNRESOLVED {addr.key}: {e.reason}")
        log(f"         {time.monotonic() - started:.1f}s")


def _updated(store: Store, key: str) -> str | None:
    rows = store.run("MATCH (r:Resource {address: $k}) RETURN r.updated_at AS u", k=key)
    return rows[0]["u"] if rows else None
