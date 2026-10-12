"""What the store hands back: a Cache Read's outcome and the stored content it served."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

HIT = "hit"
SUBSUMED = "subsumed"
MISS = "miss"

FETCHED = "FETCHED"
PROMPTED = "PROMPTED"


@dataclass(frozen=True)
class Served:
    """The answer to one Cache Read.

    ``resource`` holds the stored properties (``fetched_at``, ``updated_at`` and
    the content) and ``comments`` the item's Comments, oldest first; both are
    empty on a miss. ``kind`` is the Resource Kind (Repository, Issue,
    PullRequest), or ``Listing``: then ``resource`` holds the Listing's own
    facts (``fetched_at``, ``member_count``, ``total_count``, ``fully_expanded``),
    ``index`` one row per member, newest created first, and ``served_from`` the
    stored Listing's key — another Listing's when the outcome is ``subsumed``.
    """

    address: str
    outcome: str
    kind: str | None = None
    resource: dict[str, Any] = field(default_factory=dict)
    comments: list[dict[str, Any]] = field(default_factory=list)
    index: list[dict[str, Any]] = field(default_factory=list)
    served_from: str | None = None


@dataclass(frozen=True)
class SweepReport:
    """What one Sweep did.

    Touches ``resolved`` and left ``unresolved`` (by reason); full-depth fetches
    (``fetched``); cheap revalidations (``checked``) and the stored Resources they
    confirmed current without a refetch (``unchanged``); Touch links drawn (``linked``).
    """

    resolved: int = 0
    unresolved: dict[str, int] = field(default_factory=dict)
    fetched: int = 0
    linked: int = 0
    checked: int = 0
    unchanged: int = 0
