"""Capture health's local half: what the hooks' capture-failure log says went wrong (#378).

Failures are never written to the graph, so this is the only place a hook that failed to record anything shows up.
The log is read through ``memgraph_context_graph.capture_log``, which resolves the same path the hooks write to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from rich.text import Text

from . import fmt, theme

if TYPE_CHECKING:
    from memgraph_context_graph.capture_log import Failure, FailureGroup

WINDOW = timedelta(hours=24)


@dataclass
class Failures:
    """Capture failures in the last day, grouped by what broke."""

    #: Why the log can't be read (an older memgraph-context-graph without it); None when it can.
    unavailable: str | None = None
    path: str | None = None
    groups: list[FailureGroup] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Failures in the window."""
        return sum(group.count for group in self.groups)

    def describe(self, now: datetime) -> list[Text]:
        """Lines for the health pane. Never claims "no failures" when the log can't be read."""
        if self.unavailable:
            return [Text("capture-failure log: ", style=theme.MUTED).append(self.unavailable, style=theme.WARN)]
        if not self.groups:
            return [Text("no capture failures in the last 24h", style=theme.OK)]
        last = max(group.last_at for group in self.groups)
        lines = [
            Text(f"{self.total} capture failure(s) in the last 24h", style=theme.BAD_STRONG).append(
                f", last {fmt.ago(last, now)}"
            )
        ]
        for group in self.groups[:3]:
            where = "/".join(part for part in (group.stage, group.connector) if part)
            line = Text(f"  {where} {group.error_type} ×{group.count}  ", style=theme.BAD)
            lines.append(line.append(fmt.clip(group.last_error, 70), style=theme.MUTED))
        lines.append(Text(f"  log: {self.path}", style=theme.MUTED))
        return lines


def _read(since: datetime) -> list[Failure] | None:
    try:
        from memgraph_context_graph import capture_log
    except ImportError:
        return None
    return capture_log.read_recent(since=since)


def recent_failures(now: datetime, *, project: str | None = None) -> Failures:
    """The last day of the capture-failure log, grouped by (stage, connector, error_type), newest first.

    Args:
        project: Keep only failures whose ``cwd`` is this working directory.
    """
    failures = _read(now - WINDOW)
    if failures is None:
        return Failures(unavailable="not available in this memgraph-context-graph; upgrade it to see failures")
    from memgraph_context_graph import capture_log

    if project is not None:
        failures = [failure for failure in failures if failure.cwd == project]
    return Failures(path=str(capture_log.path()), groups=capture_log.summarize(failures))


def session_failures(session_id: str, now: datetime) -> list[Failure]:
    """Capture failures in the last day for one session, newest first; empty when the log can't be read."""
    return [failure for failure in _read(now - WINDOW) or [] if failure.session_id == session_id]
