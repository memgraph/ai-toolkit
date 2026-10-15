"""Turning graph values into short terminal text: times, durations, bars and action labels."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_FRACTION = re.compile(r"(\.\d{6})\d+")


def to_datetime(value: Any) -> datetime | None:
    """Read a timestamp as the graph stores it: ISO string (any precision), epoch seconds or ms, or datetime.

    Naive values are taken as UTC. Returns None for anything unreadable.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = value / 1000 if value > 10**11 else value
        moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
    else:
        text = str(value).strip().replace("Z", "+00:00")
        # Memgraph writes nanoseconds and appends a zone name; fromisoformat takes neither.
        text = _FRACTION.sub(r"\1", text.split("[", 1)[0])
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def ago(moment: datetime | None, now: datetime) -> str:
    """``3m ago``-style age of *moment* relative to *now*; ``—`` when unknown."""
    if moment is None:
        return "—"
    seconds = max((now - moment).total_seconds(), 0)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return f"{int(seconds)}s ago"


def duration(start: datetime | None, end: datetime | None) -> str:
    """``4.2s`` or ``3m07s`` between two moments; empty when either is unknown."""
    if start is None or end is None:
        return ""
    seconds = max((end - start).total_seconds(), 0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def bar(start: datetime | None, end: datetime | None, t0: datetime | None, t1: datetime | None, width: int) -> str:
    """A time bar placing ``start``..``end`` within the ``t0``..``t1`` window, *width* cells wide."""
    if start is None or t0 is None or t1 is None or t1 <= t0:
        return ""
    span = (t1 - t0).total_seconds()
    left = int((start - t0).total_seconds() / span * width)
    right = int(((end or start) - t0).total_seconds() / span * width)
    return " " * left + ("█" * max(right - left, 0) or "▌")


def id_width(ids: Sequence[str], least: int = 8) -> int:
    """How many leading characters of each id tell them apart, never fewer than *least*.

    Codex session ids start with a timestamp, so two sessions started in the same minute share eight characters.
    """
    width = least
    while width < max((len(i) for i in ids), default=0) and len({i[:width] for i in ids}) < len(set(ids)):
        width += 1
    return width


def clip_start(value: Any, width: int) -> str:
    """*value* as one line, cut from the left to *width* characters with an ellipsis, for paths."""
    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= width else "…" + text[-(width - 1) :]


def project_name(working_directory: str | None) -> str:
    """The last path segment of a working directory, or ``—``."""
    if not working_directory:
        return "—"
    return PurePath(str(working_directory).rstrip("/\\")).name or str(working_directory)


def tool_input(props: str | None) -> dict[str, Any]:
    """The ``tool_input`` an actions-graph ToolCall stores in its JSON ``properties``; empty when unreadable."""
    try:
        value = json.loads(props or "{}").get("tool_input")
    except (ValueError, AttributeError):
        return {}
    return value if isinstance(value, dict) else {}


def action_label(action: dict[str, Any], width: int = 48) -> str:
    """One line describing an action: the tool and its self-description, or the message text."""
    if action.get("type") == "tool_call":
        given = tool_input(action.get("props"))
        detail = next(
            (
                given[key]
                for key in ("description", "command", "file_path", "path", "pattern", "url", "query", "prompt")
                if given.get(key)
            ),
            "",
        )
        text = f"{action.get('tool') or 'tool'} {' '.join(str(detail).split())}"
    else:
        snippet = " ".join(str(action.get("text") or "").split())
        text = f"{action.get('type') or '?'} {snippet}"
    text = text.strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def clip(value: Any, width: int) -> str:
    """*value* as one line, cut to *width* characters with an ellipsis."""
    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= width else text[: width - 1] + "…"
