"""Listing matching: when a stored Listing can answer an asked one. Pure, no I/O.

Two ways a Listing answers from memory:

- **exact** — the same key, and the stored Listing either holds every member
  or at least as many as are asked for (members keep GitHub's order, newest
  created first, so a shorter ask is a prefix of a longer one);
- **subsumed** — a *fully expanded* Listing of the same repository and kind
  whose every filter the asked one applies at least as narrowly; the asked
  filters are then applied to the stored members locally.

Free-text searches only ever match exactly: whether a broader search contains
a narrower one's results is GitHub's ranking, not something to guess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .address import Address

#: Which member states each listing state admits. gh's ``--state closed`` on
#: pull requests includes merged ones.
_STATES = {
    ("issues", "open"): {"OPEN"},
    ("issues", "closed"): {"CLOSED"},
    ("pulls", "open"): {"OPEN"},
    ("pulls", "closed"): {"CLOSED", "MERGED"},
    ("pulls", "merged"): {"MERGED"},
}


def exact_serves(*, fully_expanded: bool, member_count: int, wanted: Address) -> bool:
    """Whether a stored Listing with ``wanted``'s key can answer it."""
    return fully_expanded or (wanted.limit is not None and wanted.limit <= member_count)


def subsumes(stored: Address, wanted: Address) -> bool:
    """Whether a fully expanded ``stored`` Listing contains every member ``wanted`` asks for."""
    if stored.kind != "listing" or wanted.kind != "listing":
        return False
    if (stored.owner, stored.repo, stored.item_kind) != (wanted.owner, wanted.repo, wanted.item_kind):
        return False
    if stored.query or wanted.query:
        return False
    have, want = stored.filter_map, wanted.filter_map
    if not _states(stored.item_kind, want["state"]) <= _states(stored.item_kind, have["state"]):
        return False
    if not _labels(have) <= _labels(want):  # GitHub label filters are AND: more labels, fewer members
        return False
    return all(not have.get(name) or have.get(name) == want.get(name) for name in ("author", "assignee", "milestone"))


def member_matches(member: dict[str, Any], wanted: Address) -> bool:
    """Whether a stored member belongs in ``wanted`` (state, labels, author, assignee, milestone)."""
    want = wanted.filter_map
    if member["state"] not in _states(wanted.item_kind, want["state"]):
        return False
    if not _labels(want) <= {label.lower() for label in member.get("labels") or []}:
        return False
    if want.get("author") and (member.get("author") or "").lower() != want["author"]:
        return False
    if want.get("assignee") and want["assignee"] not in {login.lower() for login in member.get("assignees") or []}:
        return False
    return not want.get("milestone") or (member.get("milestone") or "").lower() == want["milestone"]


def _states(item_kind: str | None, state: str) -> set[str]:
    if state == "all":
        return {"OPEN", "CLOSED", "MERGED"}
    return _STATES[(item_kind or "issues", state)]


def _labels(filters: dict[str, str]) -> set[str]:
    return set(filters["labels"].split(",")) if filters.get("labels") else set()
