"""The local capture-failure log: every capture failure, kept on this machine, never synced to the graph.

A hook must never break the session it observes, so the runner swallows its
failures; this log is where they go instead of nowhere. The graph records only
what came in, and reconciliation state stays on the Session (#381), so the log
holds the failures that left nothing in the graph to attach to.

Writers (the hook runner, the connector link, sessions-graph's detached
children) call :func:`record_failure`. Readers (``doctor``, the viewer) call
:func:`read_recent` and :func:`summarize`. Both resolve the file through
:func:`path`, so they always agree on which log belongs to which config.

The write path is shaped by many concurrent hook processes appending to one
file with no lock and no daemon (#378): one ``O_APPEND`` write of a whole line
of at most :data:`MAX_LINE_BYTES`, rotated to ``<log>.1`` past
:data:`ROTATE_BYTES`. Racing writers can lose old lines at a rotation, never
corrupt new ones, and readers skip any line that doesn't parse.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

#: Bumped whenever a line's shape changes, so readers never guess.
SCHEMA_VERSION = 1
#: One line, newline included. Keeps a single ``write()`` whole on local filesystems.
MAX_LINE_BYTES = 4096
#: Size at which the log rotates to ``<log>.1``; at most about twice this is kept.
ROTATE_BYTES = 1024 * 1024
_MAX_ERROR_CHARS = 512
#: How far from the end of each file a read looks; the whole bounded log fits.
_TAIL_BYTES = 256 * 1024
#: Values shorter than this are not redacted: replacing a two-letter user name everywhere would garble the message.
_MIN_REDACT_LENGTH = 3

#: Where in a hook's life a failure happened; the viewer groups on it.
STAGES = frozenset(
    {"args", "config", "payload", "setup", "connect", "translate", "emit", "respond", "spawn", "embed", "reconcile"}
)

_DEFAULT_LOG = Path.home() / ".local" / "state" / "context-graph" / "capture-failures.jsonl"

#: Set by tests only, so a suite never appends to the user's real log.
_path_override: Path | None = None


def path() -> Path:
    """The capture-failure log for the config in use.

    The default config logs to ``~/.local/state/context-graph/capture-failures.jsonl``.
    A config chosen with ``CONTEXT_GRAPH_CONFIG`` logs beside that file as
    ``<config-stem>.capture-failures.jsonl``, so a temporary config (tests,
    eval) never writes to the user's log. ``XDG_STATE_HOME`` is deliberately
    not read: the hook and the viewer must resolve the same file whatever
    environment each was started from (ADR 0002).
    """
    if _path_override is not None:
        return _path_override
    from memgraph_context_graph.adapters._identity import CONFIG_PATH_ENV, default_config_file

    override = os.environ.get(CONFIG_PATH_ENV)
    if not override:
        return _DEFAULT_LOG
    config = Path(override).expanduser()
    if config == default_config_file():
        return _DEFAULT_LOG
    return config.parent / f"{config.stem}.capture-failures.jsonl"


def record_failure(
    stage: str,
    exc: BaseException | None = None,
    *,
    level: str = "error",
    runtime: str | None = None,
    hook_event: str | None = None,
    event_type: str | None = None,
    connector: str | None = None,
    session_id: str | None = None,
    tool_name: str | None = None,
    cwd: str | None = None,
    error_type: str | None = None,
    error: str | None = None,
    started: float | None = None,
) -> bool:
    """Append one failure to the log. Never raises and never blocks.

    Args:
        stage: Where it failed, one of :data:`STAGES`.
        exc: The exception, when there is one; its type, cause type, message
            and innermost frame are recorded, never its traceback or locals.
        level: ``"error"`` for a lost capture, ``"warning"`` for one that went
            through degraded (a malformed config, a missing session id).
        runtime, hook_event, event_type, connector, session_id, tool_name, cwd:
            What was being captured. Identifiers only: payloads, tool input and
            output, and prompts are never written.
        error_type, error: Used when there is no exception to describe.
        started: ``time.monotonic()`` at hook start, for ``elapsed_ms``.

    Returns:
        Whether the line was written.
    """
    try:
        entry = _entry(
            stage,
            exc,
            level=level,
            runtime=runtime,
            hook_event=hook_event,
            event_type=event_type,
            connector=connector,
            session_id=session_id,
            tool_name=tool_name,
            cwd=cwd,
            error_type=error_type,
            error=error,
            started=started,
        )
        _append(path(), _encode(entry))
        return True
    except Exception:  # The log must never be the reason a hook fails.
        return False


@dataclass(frozen=True)
class Failure:
    """One parsed line of the log."""

    at: datetime
    level: str
    stage: str
    error_type: str
    error: str
    connector: str | None = None
    session_id: str | None = None
    runtime: str | None = None
    hook_event: str | None = None
    tool_name: str | None = None
    cwd: str | None = None
    #: The whole line as written, for fields this class doesn't name.
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class FailureGroup:
    """Repeats of one failure, which is how the log is read: one row, not 1,200 identical lines."""

    stage: str
    connector: str | None
    error_type: str
    count: int
    first_at: datetime
    last_at: datetime
    #: The newest member's message and level.
    last_error: str
    level: str
    #: Distinct sessions that hit it, newest first.
    session_ids: tuple[str, ...] = ()


def read_recent(*, since: datetime | None = None, limit: int = 1000, log_path: Path | None = None) -> list[Failure]:
    """The newest failures in the log, newest first.

    Reads the tail of the current file and its rotated predecessor and skips
    any line that doesn't parse (a torn last line, a future schema).

    Args:
        since: Only failures at or after this (timezone-aware) time.
        limit: At most this many.
        log_path: The log to read; :func:`path` by default.
    """
    target = log_path or path()
    failures: list[Failure] = []
    for file in (target.with_name(target.name + ".1"), target):  # oldest first: append order
        failures.extend(_parse_lines(_tail(file)))
    if since is not None:
        failures = [failure for failure in failures if failure.at >= since]
    # Timestamps are millisecond-precise, so failures can tie; reversed append order breaks ties newest
    # first, and the sort is stable.
    failures.reverse()
    failures.sort(key=lambda failure: failure.at, reverse=True)
    return failures[:limit]


def summarize(failures: list[Failure]) -> list[FailureGroup]:
    """Group *failures* by (stage, connector, error_type), most recently seen first."""
    groups: dict[tuple[str, str | None, str], list[Failure]] = {}
    for failure in failures:
        groups.setdefault((failure.stage, failure.connector, failure.error_type), []).append(failure)
    summary = []
    for (stage, connector, error_type), members in groups.items():
        members.sort(key=lambda failure: failure.at, reverse=True)
        newest = members[0]
        sessions = tuple(dict.fromkeys(m.session_id for m in members if m.session_id))
        summary.append(
            FailureGroup(
                stage=stage,
                connector=connector,
                error_type=error_type,
                count=len(members),
                first_at=members[-1].at,
                last_at=newest.at,
                last_error=newest.error,
                level=newest.level,
                session_ids=sessions,
            )
        )
    summary.sort(key=lambda group: group.last_at, reverse=True)
    return summary


def recent_summary(*, hours: int = 24, log_path: Path | None = None) -> list[FailureGroup]:
    """:func:`summarize` over the last *hours* of the log, which is what ``doctor`` shows."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    return summarize(read_recent(since=since, log_path=log_path))


# --- writing ---


def _entry(
    stage: str,
    exc: BaseException | None,
    *,
    level: str,
    runtime: str | None,
    hook_event: str | None,
    event_type: str | None,
    connector: str | None,
    session_id: str | None,
    tool_name: str | None,
    cwd: str | None,
    error_type: str | None,
    error: str | None,
    started: float | None,
) -> dict[str, Any]:
    secrets, memgraph = _config_facts()
    message = error if error is not None else (str(exc) if exc is not None else "")
    cause = exc.__cause__ if exc is not None else None
    return {
        "v": SCHEMA_VERSION,
        "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": level,
        "runtime": runtime,
        "hook_event": _clip(hook_event, 64),
        "event_type": event_type,
        "stage": stage,
        "connector": connector,
        "session_id": _clip(session_id, 128),
        "tool_name": _clip(tool_name, 128),
        "cwd": _clip(cwd, 512),
        "error_type": error_type or (_qualified_name(type(exc)) if exc is not None else "Unknown"),
        "cause_type": _qualified_name(type(cause)) if cause is not None else None,
        "error": _clip(_redact(message, secrets), _MAX_ERROR_CHARS),
        "where": _where(exc),
        "memgraph": memgraph,
        "versions": _versions(connector),
        "pid": os.getpid(),
        "elapsed_ms": round((time.monotonic() - started) * 1000) if started is not None else None,
    }


def _encode(entry: dict[str, Any]) -> bytes:
    line = json.dumps(entry, ensure_ascii=True, separators=(",", ":"), default=str) + "\n"
    if len(line) > MAX_LINE_BYTES:
        # Only the free-text fields can be long; shrink them until the line fits.
        entry = {**entry, "error": _clip(entry.get("error"), 128), "cwd": _clip(entry.get("cwd"), 128)}
        line = json.dumps(entry, ensure_ascii=True, separators=(",", ":"), default=str) + "\n"
    if len(line) > MAX_LINE_BYTES:
        keep = ("v", "at", "level", "runtime", "stage", "connector", "session_id", "error_type")
        line = json.dumps({key: entry.get(key) for key in keep}, ensure_ascii=True, separators=(",", ":")) + "\n"
    return line.encode("ascii")


def forget_session(session_id: str, *, log_path: Path | None = None) -> int:
    """Rewrite the log (and its rotated ``.1``) without *session_id*'s lines; how many went.

    Each file is replaced atomically, so a reader never sees half of one; a
    failure a hook appends between the read and the replace is lost, which
    a diagnostics log can afford.
    """
    target = log_path or path()
    removed = 0
    for file in (target, target.with_name(target.name + ".1")):
        if not file.is_file():
            continue
        lines = file.read_bytes().splitlines(keepends=True)
        kept = [line for line in lines if not _is_from(line, session_id)]
        if len(kept) == len(lines):
            continue
        removed += len(lines) - len(kept)
        scratch = file.with_name(file.name + ".forget")
        fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
        try:
            os.write(fd, b"".join(kept))
        finally:
            os.close(fd)
        os.replace(scratch, file)
    return removed


def _is_from(line: bytes, session_id: str) -> bool:
    try:
        entry = json.loads(line)
    except ValueError:
        return False
    return isinstance(entry, dict) and entry.get("session_id") == session_id


def _append(target: Path, line: bytes) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    # O_NONBLOCK: a FIFO planted at the path fails the open instead of hanging the hook.
    flags |= getattr(os, "O_NONBLOCK", 0)
    fd = os.open(target, flags, 0o600)
    try:
        if os.fstat(fd).st_size >= ROTATE_BYTES:
            # A failed rename (e.g. a Windows sharing violation) just rotates on a later write.
            with contextlib.suppress(OSError):
                os.replace(target, target.with_name(target.name + ".1"))
        os.write(fd, line)
    finally:
        os.close(fd)


def _config_facts() -> tuple[list[str], str | None]:
    """The configured secrets to redact, and ``host:port/database`` without credentials."""
    try:
        from memgraph_context_graph.adapters._identity import load_config

        config = load_config()
    except Exception:
        return [], None
    secrets = [
        value
        for value in (
            config.memgraph_password,
            config.memgraph_user,
            config.openai_api_key,
            config.anthropic_api_key,
            config.github_token,
        )
        if value and len(value) >= _MIN_REDACT_LENGTH
    ]
    try:
        url = urlsplit(config.memgraph_url)
        memgraph = f"{url.hostname}:{url.port or 7687}/{config.memgraph_database}"
    except ValueError:
        memgraph = None
    return secrets, memgraph


def _redact(text: str, secrets: list[str]) -> str:
    # Longest first, so a secret containing another is replaced whole.
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "***")
    return text


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    value = str(value)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _qualified_name(cls: type) -> str:
    module = cls.__module__
    return cls.__qualname__ if module in {"builtins", "__main__"} else f"{module}.{cls.__qualname__}"


def _where(exc: BaseException | None) -> str | None:
    """The innermost frame as ``package/file.py:line in func``: enough to find an adapter bug, no locals."""
    if exc is None or exc.__traceback__ is None:
        return None
    frame = traceback.extract_tb(exc.__traceback__)[-1]
    filename = frame.filename.replace("\\", "/")
    for marker in ("/site-packages/", "/src/"):
        if marker in filename:
            filename = filename.rsplit(marker, 1)[1]
            break
    else:
        filename = filename.rsplit("/", 1)[-1]
    return f"{filename}:{frame.lineno} in {frame.name}"


def _versions(connector: str | None) -> dict[str, str]:
    from importlib import metadata

    versions = {}
    for package in ("memgraph-context-graph", connector):
        if not package:
            continue
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


# --- reading ---


def _tail(file: Path) -> list[bytes]:
    try:
        with file.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - _TAIL_BYTES)
            handle.seek(start)
            data = handle.read()
    except OSError:
        return []
    lines = data.split(b"\n")
    if start > 0:
        lines = lines[1:]  # The first line is cut by the seek.
    return [line for line in lines if line.strip()]


def _parse_lines(lines: list[bytes]) -> list[Failure]:
    failures = []
    for line in lines:
        try:
            raw = json.loads(line)
            at = datetime.fromisoformat(str(raw["at"]).replace("Z", "+00:00"))
            failures.append(
                Failure(
                    at=at,
                    level=str(raw.get("level") or "error"),
                    stage=str(raw["stage"]),
                    error_type=str(raw.get("error_type") or "Unknown"),
                    error=str(raw.get("error") or ""),
                    connector=raw.get("connector"),
                    session_id=raw.get("session_id"),
                    runtime=raw.get("runtime"),
                    hook_event=raw.get("hook_event"),
                    tool_name=raw.get("tool_name"),
                    cwd=raw.get("cwd"),
                    raw=raw,
                )
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return failures
