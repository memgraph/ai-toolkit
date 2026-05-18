"""Data models for Memory Graph.

Defines the core data structure for a Memory — a cross-session-durable
free-form text assertion written explicitly by an agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_@.\-]{1,256}$")
_MEMORY_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


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
    """

    user_id: str
    content: str
    memory_id: str = field(default_factory=_generate_id)
    created_at: str = field(default_factory=_utc_now)
    session_id: str | None = None

    def __post_init__(self) -> None:
        validate_user_id(self.user_id)
        validate_memory_id(self.memory_id)
        validate_content(self.content)
