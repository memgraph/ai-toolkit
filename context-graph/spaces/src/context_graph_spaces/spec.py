"""The spaces spec: who the members are and which shared spaces they belong to.

An admin keeps one TOML file, reviewed like code, and ``context-graph-spaces
apply`` makes Memgraph match it::

    members = ["ante", "bob"]
    personal_memory_limit_mb = 2048        # optional, per personal space

    [spaces.payments]
    members = ["ante", "bob"]
    memory_limit_mb = 4096                 # optional
    labels = ["Memory", "MemoryVersion"]   # optional; what members may write
    edge_types = ["HAS_MEMORY"]            # optional

Every member gets a personal space automatically; it isn't listed. A shared
space's ``labels``/``edge_types`` default to the curated memory the memory tool
writes (map #501: v1 shares curated memory only). They are data, not code, so
widening what a space holds is an edit here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # tomli is tomllib's backport
    import tomli as tomllib

#: What members may create and change in a shared space by default: the
#: memory tool's files, their history, and what links them to the user and
#: project. Chunks and entities are left to the space's reconcile worker.
DEFAULT_SHARED_LABELS = ("Memory", "MemoryVersion", "User", "Project")
DEFAULT_SHARED_EDGE_TYPES = ("HAS_MEMORY", "HAS_MEMORY_VERSION", "PREVIOUS_VERSION", "ABOUT")

# Logins and space names become database, role and user names in DDL, which
# Memgraph can't parameterize; this pattern is the guard that keeps them inert.
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
#: Prefixes the provisioner owns; a member login can't take one.
RESERVED_PREFIXES = ("cg_", "space_", "u_")


class SpecError(ValueError):
    """The spec file is malformed or inconsistent."""


@dataclass(frozen=True)
class SharedSpace:
    """One shared space: its members and what they may write in it."""

    name: str
    members: tuple[str, ...]
    labels: tuple[str, ...] = DEFAULT_SHARED_LABELS
    edge_types: tuple[str, ...] = DEFAULT_SHARED_EDGE_TYPES
    memory_limit_mb: int | None = None


@dataclass(frozen=True)
class Spec:
    """The whole desired state: members and shared spaces."""

    members: tuple[str, ...]
    spaces: tuple[SharedSpace, ...] = ()
    personal_memory_limit_mb: int | None = None

    def spaces_of(self, login: str) -> list[SharedSpace]:
        """The shared spaces *login* belongs to, in spec order."""
        return [space for space in self.spaces if login in space.members]


def load_spec(path: str | Path) -> Spec:
    """Read and validate a spec file.

    Raises:
        SpecError: the file isn't valid TOML or breaks a rule above.
    """
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SpecError(f"Can't read spec {path}: {exc}") from exc
    return parse_spec(data)


def parse_spec(data: dict[str, Any]) -> Spec:
    """Validate a parsed spec mapping.

    Raises:
        SpecError: unknown keys, bad names, members of a space who aren't members, duplicate entries.
    """
    _no_unknown_keys(data, {"members", "spaces", "personal_memory_limit_mb"}, "the spec")
    members = _names(data.get("members", []), "members", login=True)
    spaces = []
    for name, body in (data.get("spaces") or {}).items():
        where = f"[spaces.{name}]"
        if not isinstance(body, dict):
            raise SpecError(f"{where} must be a table")
        _no_unknown_keys(body, {"members", "labels", "edge_types", "memory_limit_mb"}, where)
        _check_name(name, where)
        space_members = _names(body.get("members", []), f"{where} members", login=True)
        if outsiders := sorted(set(space_members) - set(members)):
            raise SpecError(f"{where} lists {', '.join(outsiders)}, who aren't in the top-level members")
        spaces.append(
            SharedSpace(
                name=name,
                members=space_members,
                labels=_labels(body.get("labels", DEFAULT_SHARED_LABELS), f"{where} labels"),
                edge_types=_labels(body.get("edge_types", DEFAULT_SHARED_EDGE_TYPES), f"{where} edge_types"),
                memory_limit_mb=_limit(body.get("memory_limit_mb"), f"{where} memory_limit_mb"),
            )
        )
    return Spec(
        members=members,
        spaces=tuple(spaces),
        personal_memory_limit_mb=_limit(data.get("personal_memory_limit_mb"), "personal_memory_limit_mb"),
    )


def _no_unknown_keys(table: dict[str, Any], known: set[str], where: str) -> None:
    if unknown := sorted(set(table) - known):
        raise SpecError(f"Unknown key(s) in {where}: {', '.join(unknown)}")


def _names(value: Any, where: str, *, login: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SpecError(f"{where} must be a list of names")
    for item in value:
        _check_name(item, where)
        if login and item.startswith(RESERVED_PREFIXES):
            raise SpecError(f"{where}: {item!r} starts with a reserved prefix ({', '.join(RESERVED_PREFIXES)})")
    if len(set(value)) != len(value):
        raise SpecError(f"{where} lists a name twice")
    return tuple(value)


def _check_name(name: str, where: str) -> None:
    if not _NAME.match(name):
        raise SpecError(f"{where}: {name!r} must be lowercase letters, digits and _, starting with a letter")


def _labels(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and _LABEL.match(item) for item in value):
        raise SpecError(f"{where} must be a list of label or edge-type names")
    return tuple(value)


def _limit(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SpecError(f"{where} must be a positive whole number of megabytes")
    return value
