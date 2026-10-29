"""Events the graph refused a connector, kept on this machine until ``update`` can replay them.

When a newer Context Graph has reshaped a component's part of the graph
(:mod:`memgraph_context_graph.graph_version`), this install stops writing it.
The capture-failure log keeps identifiers only, so without this file what the
agent did meanwhile would be lost. Instead, every event of that run is
appended here, as the capture policy left it: redacted, clipped, with excluded
tools already dropped, and with its original timestamp. ``update`` then
replays it through the connectors the graph allows again (:func:`replay`).

One JSON object per line:

    {"v": 1, "spooled_at": "...", "connectors": ["skills-graph"], "attempts": 0,
     "event": {"type": "ToolStartEvent", "fields": {...}}}

Later releases read files an older one wrote, so a change to that shape raises
:data:`FORMAT_VERSION` and keeps a reader for every earlier one. The file holds
the user's own captured content: it is created ``0600``, never leaves the
machine, and stops growing at :data:`MAX_BYTES`, after which new events are
dropped and logged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memgraph_context_graph import capture_log, events
from memgraph_context_graph.protocols import GraphConnector

if TYPE_CHECKING:
    from pathlib import Path

    from memgraph_context_graph.events import Event

#: Raised whenever a line's shape changes; readers keep handling every earlier value.
FORMAT_VERSION = 1
#: Size past which new events are dropped rather than kept.
MAX_BYTES = 64 * 1024 * 1024
#: Replays an entry may fail before it is dropped, so one bad event can't block the rest forever.
MAX_ATTEMPTS = 3


def path() -> Path:
    """The spool: beside the capture-failure log, so a config chosen with ``CONTEXT_GRAPH_CONFIG`` gets its own."""
    log = capture_log.path()
    return log.with_name(log.name.removesuffix("capture-failures.jsonl") + "refused-events.jsonl")


class Spool(GraphConnector):
    """Stands in for the connectors the graph refused, keeping every event for them in the spool."""

    def __init__(self, connectors: list[str]) -> None:
        self.connectors = list(connectors)

    def supports(self, event: Event) -> bool:
        # Which events the refused connectors want is theirs to decide at replay, so all are kept.
        return True

    def on_event(self, event: Event) -> None:
        if not append(event, self.connectors):
            capture_log.record_failure(
                "emit",
                level="error",
                connector=",".join(self.connectors),
                event_type=event.event_type.value,
                session_id=event.session_id,
                error_type="RefusedEventSpoolFull",
                error=f"{path()} is full; run memgraph-context-graph update to replay and empty it",
            )


def append(event: Event, connectors: list[str], *, attempts: int = 0) -> bool:
    """Keep *event* for *connectors*; whether it was written (False when the spool is full or unwritable)."""
    entry = {
        "v": FORMAT_VERSION,
        "spooled_at": datetime.now(timezone.utc).isoformat(),
        "connectors": connectors,
        "attempts": attempts,
        "event": encode(event),
    }
    return _append_line(json.dumps(entry, default=str) + "\n")


def encode(event: Event) -> dict[str, Any]:
    """*event* as JSON: its class name and the fields its constructor takes."""
    fields = {f.name: getattr(event, f.name) for f in dataclasses.fields(event) if f.init}
    return {"type": type(event).__name__, "fields": fields}


def decode(raw: Any) -> Event | None:
    """The event :func:`encode` wrote; None for a type this release doesn't know.

    Fields this release doesn't know are dropped and missing ones take their
    defaults, so an event spooled by another release still replays.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("fields"), dict):
        return None
    cls = getattr(events, str(raw.get("type")), None)
    if not (isinstance(cls, type) and issubclass(cls, events.Event)):
        return None
    known = {f.name for f in dataclasses.fields(cls) if f.init}
    try:
        return cls(**{name: value for name, value in raw["fields"].items() if name in known})
    except TypeError:
        return None


def pending() -> bool:
    """Whether any events are kept: in the spool, or left by a replay that died."""
    spool = path()
    return spool.exists() or any(spool.parent.glob(f"{spool.name}.*.replaying"))


@dataclass(frozen=True)
class ReplayResult:
    """What one replay did with the spooled events."""

    replayed: int = 0
    waiting: int = 0
    dropped: int = 0


def replay(memgraph_env: dict[str, str]) -> ReplayResult:
    """Send every spooled event to the connectors the graph now allows, keeping the rest for later.

    The spool is renamed before it is read, so hooks appending meanwhile start
    a new file rather than racing this one; a run that died mid-replay left its
    renamed file behind, which the next replay takes too. An entry whose
    connectors the graph still refuses goes back, as does one a connector
    failed, until :data:`MAX_ATTEMPTS`.

    Raises:
        Exception: when Memgraph can't be reached; the spool is then left as it was.
    """
    from memgraph_context_graph import graph_version
    from memgraph_context_graph.hooks.runner import create_link

    claimed = _claim()
    if not claimed:
        return ReplayResult()
    # Fresh, so create_link below agrees; strict, because allowing everything on a failed check would replay too early.
    rules = graph_version.check_at(memgraph_env, fresh=True, strict=True)
    links: dict[tuple[str, ...], Any] = {}
    failed: list[str] = []
    replayed = waiting = dropped = 0
    for entry in _entries(claimed):
        event = decode(entry.get("event"))
        connectors = [c for c in entry.get("connectors", []) if isinstance(c, str)]
        attempts = entry["attempts"] if isinstance(entry.get("attempts"), int) else 0
        if event is None or not connectors:
            dropped += 1
            continue
        allowed = tuple(c for c in connectors if rules.may_write(c))
        refused = [c for c in connectors if c not in allowed]
        if refused:
            waiting += 1
            append(event, refused, attempts=attempts)
        if not allowed:
            continue
        if allowed not in links:
            links[allowed] = create_link(
                allowed, memgraph_env=memgraph_env, on_error=lambda name, _event, _exc: failed.append(name)
            )
        failed.clear()
        links[allowed].emit(event)
        if not failed:
            replayed += 1
        elif attempts + 1 < MAX_ATTEMPTS:
            waiting += 1
            append(event, list(dict.fromkeys(failed)), attempts=attempts + 1)
        else:
            dropped += 1
            capture_log.record_failure(
                "emit",
                connector=",".join(dict.fromkeys(failed)),
                event_type=event.event_type.value,
                session_id=event.session_id,
                error_type="RefusedEventDropped",
                error=f"replay failed {MAX_ATTEMPTS} times; the event was dropped",
            )
    for file in claimed:
        with contextlib.suppress(OSError):
            file.unlink()
    return ReplayResult(replayed, waiting, dropped)


def _claim() -> list[Path]:
    """Rename the spool aside for this replay, and pick up any a died replay left; the files to read, oldest first."""
    spool = path()
    with contextlib.suppress(OSError):
        os.replace(spool, spool.with_name(f"{spool.name}.{os.getpid()}.replaying"))
    return sorted(spool.parent.glob(f"{spool.name}.*.replaying"), key=lambda file: file.stat().st_mtime)


def _entries(files: list[Path]) -> list[dict[str, Any]]:
    entries = []
    for file in files:
        with contextlib.suppress(OSError):
            for line in file.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(ValueError):
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        entries.append(entry)
    return entries


def _append_line(line: str) -> bool:
    """One ``O_APPEND`` write of a whole line, so concurrent hooks never interleave inside one."""
    data = line.encode("utf-8")
    target = path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    except OSError:
        return False
    try:
        if os.fstat(fd).st_size + len(data) > MAX_BYTES:
            return False
        os.write(fd, data)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)
