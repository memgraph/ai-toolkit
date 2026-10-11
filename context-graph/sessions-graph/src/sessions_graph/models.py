"""Data models for Sessions Graph.

Defines the core data structure for a Memory — a cross-session-durable
free-form text assertion written explicitly by an agent, stored as one file
under the ``/memories`` tree the memory tool exposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_@.\-]{1,256}$")
_MEMORY_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

#: The root every Memory path lives under, as the memory tool names it.
MEMORY_ROOT = "/memories"
_MAX_PATH_CHARS = 1024
# Encoded separators and dots: a path is matched literally, so an encoded
# traversal can't do harm here, but a client that decodes before passing the
# path on would turn it into a real one.
_ENCODED_TRAVERSAL_RE = re.compile(r"%(2e|2f|5c)", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return str(uuid4())


class MemoryValidationError(ValueError):
    """Raised when a Memory field violates validation rules."""


def validate_user_id(user_id: str) -> str:
    if not user_id or not _USER_ID_RE.match(user_id):
        raise MemoryValidationError(f"user_id must match pattern {_USER_ID_RE.pattern!r}, got: {user_id!r}")
    return user_id


def validate_memory_id(memory_id: str) -> str:
    if not memory_id or not _MEMORY_ID_RE.match(memory_id):
        raise MemoryValidationError(f"memory_id must match pattern {_MEMORY_ID_RE.pattern!r}, got: {memory_id!r}")
    return memory_id


def normalize_memory_path(path: str) -> str:
    """*path* in canonical form: ``/memories`` or ``/memories/<segment>/...``.

    Collapses repeated and trailing slashes. Paths are matched as strings,
    never resolved against a filesystem, so rejecting traversal is what keeps
    a path inside its owner's tree.

    Raises:
        MemoryValidationError: the path is outside ``/memories``, contains a
            ``.``/``..`` segment, a backslash, a control character, or an
            encoded separator or dot.
    """
    if not isinstance(path, str) or not path:
        raise MemoryValidationError("path must be a non-empty string")
    if len(path) > _MAX_PATH_CHARS:
        raise MemoryValidationError(f"path must be at most {_MAX_PATH_CHARS} characters")
    if "\\" in path or any(ord(char) < 32 for char in path) or _ENCODED_TRAVERSAL_RE.search(path):
        raise MemoryValidationError(
            f"Invalid path {path!r}: backslashes, control characters and encoded dots are not allowed"
        )
    segments = [segment for segment in path.split("/") if segment]
    if not segments or segments[0] != MEMORY_ROOT.strip("/") or not path.startswith("/"):
        raise MemoryValidationError(f"Path must start with {MEMORY_ROOT}, got: {path}")
    if any(segment in {".", ".."} for segment in segments):
        raise MemoryValidationError(f"Path {path} would escape {MEMORY_ROOT}")
    return "/" + "/".join(segments)


def project_key(path: str) -> str | None:
    """The project a canonical memory *path* is about, or None outside ``/memories/projects/<key>/``.

    A key is one path segment, so the project a file belongs to never depends
    on how deep the file sits inside its folder.
    """
    segments = path.split("/")
    # ["", "memories", "projects", "<key>", "<file>", ...]
    if len(segments) >= 5 and segments[2] == "projects":
        return segments[3]
    return None


def validate_content(content: str) -> str:
    if not content or not content.strip():
        raise MemoryValidationError("content must be a non-empty string")
    return content


@dataclass
class Memory:
    """A cross-session-durable free-form text assertion owned by a user.

    Attributes:
        user_id:    The identity of the user this memory belongs to.
        content:    The free-form text assertion.
        memory_id:  Unique identifier (auto-generated UUID by default).
        created_at: ISO-format UTC timestamp of when the memory was written.
        session_id: The session that produced this memory (optional provenance).
        path:       Where the memory lives under ``/memories``; defaults to
                    ``/memories/notes/<memory_id>.md``.
        updated_at: ISO-format UTC timestamp of the last change; defaults to ``created_at``.
    """

    user_id: str
    content: str
    memory_id: str = field(default_factory=_generate_id)
    created_at: str = field(default_factory=_utc_now)
    session_id: str | None = None
    path: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        validate_user_id(self.user_id)
        validate_memory_id(self.memory_id)
        validate_content(self.content)
        self.path = normalize_memory_path(self.path or f"{MEMORY_ROOT}/notes/{self.memory_id}.md")
        if self.path == MEMORY_ROOT:
            raise MemoryValidationError(f"{MEMORY_ROOT} is a directory, not a memory file")
        self.updated_at = self.updated_at or self.created_at
