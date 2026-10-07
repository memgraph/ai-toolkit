"""What the store hands back: a Cache Read's outcome and the stored content it served."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

HIT = "hit"
MISS = "miss"

FETCHED = "FETCHED"
PROMPTED = "PROMPTED"


@dataclass(frozen=True)
class Served:
    """The answer to one Cache Read.

    ``resource`` holds the stored properties (``fetched_at``, ``updated_at`` and
    the content) and ``comments`` the item's Comments, oldest first; both are
    empty on a miss. ``kind`` is the Resource Kind (Repository, Issue, PullRequest).
    """

    address: str
    outcome: str
    kind: str | None = None
    resource: dict[str, Any] = field(default_factory=dict)
    comments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SweepReport:
    """What one Sweep did: Touches resolved, left unresolved by reason, and fetches made."""

    resolved: int = 0
    unresolved: dict[str, int] = field(default_factory=dict)
    fetched: int = 0
