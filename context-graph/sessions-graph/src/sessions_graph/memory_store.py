"""The memory tool's commands over one user's ``(:Memory)`` files.

Implements the command surface of the Claude API memory tool
(``memory_20250818``): ``view``, ``create``, ``str_replace``, ``insert``,
``delete`` and ``rename`` on paths under ``/memories``. Each file is one
``(:Memory {path, content})`` node owned by the store's user; directories are
not stored, they are the path prefixes their files share.

Every write that discards text (an overwriting ``create``, ``str_replace``,
``insert``, ``delete``) first keeps it as a ``(:MemoryVersion)``, so a stale
or mistaken write is recoverable; the newest :data:`MAX_VERSIONS` per file are
kept. Versions hang off the user as well as the file, so a deleted file's
history outlives it. Results and error
messages follow the memory tool documentation, because the model reads them.

The store is the single implementation behind every entry point, such as the
MCP ``memory`` tool, so path validation lives here, not in the wrappers. The user a store acts for
is fixed when it is built, by trusted code; a command only ever names paths,
so a model can't reach another user's memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .models import MEMORY_ROOT, MemoryValidationError, normalize_memory_path, project_key, validate_user_id

if TYPE_CHECKING:
    from collections.abc import Mapping

    from memgraph_toolbox.api.memgraph import Memgraph

#: Largest file a write may leave behind, in UTF-8 bytes.
MAX_FILE_BYTES = 100 * 1024
#: A file view longer than this is cut, and the model pages with ``view_range``.
VIEW_CHAR_LIMIT = 16_000
MAX_LINES = 999_999
#: Versions kept per file; older ones are deleted as new ones arrive.
MAX_VERSIONS = 20
_LINE_NUMBER_WIDTH = 6
_LISTING_DEPTH = 2


class MemoryCommandError(Exception):
    """A command that can't be carried out; the message is written for the model."""


@dataclass(frozen=True)
class MemoryVersion:
    """Text a write replaced or a delete removed, with when and by which session."""

    path: str
    content: str
    replaced_at: str
    session_id: str | None
    deleted: bool


@dataclass(frozen=True)
class MemoryFile:
    """One stored file: where it lives, what it says, and when it last changed."""

    path: str
    content: str
    updated_at: str


class MemoryStore:
    """The memory tool's commands for one user, against a live Memgraph.

    Args:
        memgraph:   Client for the graph the memories live in.
        user_id:    Whose memory this is. Every command reads and writes only this user's files.
        session_id: The session making the changes, recorded as ``PRODUCED_MEMORY`` provenance.
    """

    def __init__(self, memgraph: Memgraph, user_id: str, *, session_id: str | None = None) -> None:
        self._db = memgraph
        self.user_id = validate_user_id(user_id)
        self.session_id = session_id

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(self, arguments: Mapping[str, Any]) -> str:
        """Run one memory tool call, given its input as the model sent it.

        Raises:
            MemoryCommandError: unknown command, missing argument, or a command that fails.
        """
        command = arguments.get("command")
        if command == "view":
            return self.view(_required(arguments, "path"), _view_range(arguments.get("view_range")))
        if command == "create":
            return self.create(_required(arguments, "path"), _required(arguments, "file_text"))
        if command == "str_replace":
            return self.str_replace(
                _required(arguments, "path"), _required(arguments, "old_str"), str(arguments.get("new_str") or "")
            )
        if command == "insert":
            return self.insert(
                _required(arguments, "path"),
                _insert_line(arguments.get("insert_line")),
                _required(arguments, "insert_text"),
            )
        if command == "delete":
            return self.delete(_required(arguments, "path"))
        if command == "rename":
            return self.rename(_required(arguments, "old_path"), _required(arguments, "new_path"))
        raise MemoryCommandError(f"Error: unknown command {command}")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> str:
        """A directory listing two levels deep, or a file's numbered lines.

        Raises:
            MemoryCommandError: the path is invalid, doesn't exist, or the range is out of bounds.
        """
        path = _path(path)
        content = self._content(path)
        if content is None:
            if not self._is_directory(path):
                raise MemoryCommandError(f"The path {path} does not exist. Please provide a valid path.")
            return self._listing(path)

        lines = content.split("\n")
        if len(lines) > MAX_LINES:
            raise MemoryCommandError(f"File {path} exceeds maximum line limit of 999,999 lines.")
        start = 1
        if view_range is not None:
            first, last = view_range
            if first < 1 or first > len(lines) or (last != -1 and last < first):
                raise MemoryCommandError(
                    f"Invalid `view_range`: {[first, last]}. Lines run from 1 to {len(lines)}; "
                    "the end must be -1 or at least the start."
                )
            start = first
            lines = lines[first - 1 :] if last == -1 else lines[first - 1 : last]

        numbered: list[str] = []
        shown = 0
        for offset, line in enumerate(lines):
            rendered = f"{str(start + offset).rjust(_LINE_NUMBER_WIDTH)}\t{line}"
            if shown + len(rendered) > VIEW_CHAR_LIMIT and numbered:
                numbered.append(
                    f"[Truncated at line {start + offset - 1} of {start + len(lines) - 1}: "
                    f"view the rest with view_range [{start + offset}, -1].]"
                )
                break
            numbered.append(rendered)
            shown += len(rendered) + 1
        return f"Here's the content of {path} with line numbers:\n" + "\n".join(numbered)

    def create(self, path: str, file_text: str) -> str:
        """Create a file, or overwrite one that already exists.

        Raises:
            MemoryCommandError: the path is invalid, is the root or a directory,
                sits under a file, or the text is empty or too large.
        """
        path = _path(path)
        if path == MEMORY_ROOT or self._is_directory(path):
            raise MemoryCommandError(f"Error: {path} is a directory")
        if parent := self._file_ancestor(path):
            raise MemoryCommandError(f"Error: {parent} is a file, so {path} can't be created under it")
        self._write(path, file_text)
        return f"File created successfully at: {path}"

    def str_replace(self, path: str, old_str: str, new_str: str = "") -> str:
        """Replace the one occurrence of *old_str* in a file with *new_str*.

        Raises:
            MemoryCommandError: no such file, *old_str* is absent or not unique,
                or the result is empty or too large.
        """
        path = _path(path)
        content = self._existing_file(path)
        count = content.count(old_str) if old_str else 0
        if count == 0:
            raise MemoryCommandError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            )
        if count > 1:
            lines = _occurrence_lines(content, old_str)
            raise MemoryCommandError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines: "
                f"{', '.join(map(str, lines))}. Please ensure it is unique"
            )
        changed_line = content[: content.index(old_str)].count("\n")
        updated = content.replace(old_str, new_str, 1)
        self._write(path, updated, expected=content)

        lines = updated.split("\n")
        first, last = max(0, changed_line - 2), min(len(lines), changed_line + new_str.count("\n") + 3)
        snippet = "\n".join(f"{str(n + 1).rjust(_LINE_NUMBER_WIDTH)}\t{lines[n]}" for n in range(first, last))
        return (
            f"The memory file has been edited. Here is the snippet showing the change (with line numbers):\n{snippet}"
        )

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        """Insert *insert_text* after line *insert_line*; 0 inserts at the top.

        Raises:
            MemoryCommandError: no such file, the line is out of range, or the result is too large.
        """
        path = _path(path)
        content = self._existing_file(path)
        lines = content.split("\n")
        trailing_newline = lines[-1] == ""
        if trailing_newline:
            lines.pop()
        if insert_line < 0 or insert_line > len(lines):
            raise MemoryCommandError(
                f"Error: Invalid `insert_line` parameter: {insert_line}. "
                f"It should be within the range of lines of the file: [0, {len(lines)}]"
            )
        lines.insert(insert_line, insert_text.removesuffix("\n"))
        updated = "\n".join(lines) + ("\n" if trailing_newline else "")
        self._write(path, updated, expected=content)
        return f"The file {path} has been edited."

    def delete(self, path: str) -> str:
        """Delete a file, or a directory and everything under it.

        Raises:
            MemoryCommandError: the path is the root, invalid, or doesn't exist.
        """
        path = _path(path)
        if path == MEMORY_ROOT:
            raise MemoryCommandError(f"Error: Cannot delete the {MEMORY_ROOT} directory itself")
        rows = self._db.query(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            WHERE m.path = $path OR m.path STARTS WITH $prefix
            CREATE (u)-[:HAS_MEMORY_VERSION]->(:MemoryVersion {
                memory_id: m.memory_id, path: m.path, content: m.content,
                replaced_at: $now, session_id: $session_id, deleted: true
            })
            WITH m, m.memory_id AS memory_id
            DETACH DELETE m
            RETURN memory_id
            """,
            params={
                "user_id": self.user_id,
                "path": path,
                "prefix": path + "/",
                "now": _now(),
                "session_id": self.session_id,
            },
        )
        if not rows:
            raise MemoryCommandError(f"Error: The path {path} does not exist")
        for row in rows:
            self._prune_versions(row["memory_id"])
        return f"Successfully deleted {path}"

    def rename(self, old_path: str, new_path: str) -> str:
        """Move a file or a directory; the destination must not exist.

        Files keep their identity (``memory_id``) and links across the move.

        Raises:
            MemoryCommandError: either path is invalid, the source is the root or
                missing, the destination exists, or a directory would move into itself.
        """
        old_path, new_path = _path(old_path), _path(new_path)
        if old_path == MEMORY_ROOT:
            raise MemoryCommandError(f"Error: Cannot rename the {MEMORY_ROOT} directory itself")
        if new_path == MEMORY_ROOT or self._content(new_path) is not None or self._is_directory(new_path):
            raise MemoryCommandError(f"Error: The destination {new_path} already exists")
        if new_path.startswith(old_path + "/"):
            raise MemoryCommandError(f"Error: Cannot move {old_path} into itself")
        if parent := self._file_ancestor(new_path):
            raise MemoryCommandError(f"Error: {parent} is a file, so {new_path} can't be created under it")

        rows = self._db.query(
            """
            MATCH (m:Memory {user_id: $user_id})
            WHERE m.path = $old OR m.path STARTS WITH $old_prefix
            SET m.path = $new + substring(m.path, size($old)), m.updated_at = $now
            RETURN m.memory_id AS memory_id, m.path AS path
            """,
            params={
                "user_id": self.user_id,
                "old": old_path,
                "old_prefix": old_path + "/",
                "new": new_path,
                "now": _now(),
            },
        )
        if not rows:
            raise MemoryCommandError(f"Error: The path {old_path} does not exist")
        for row in rows:
            self._link_project(row["memory_id"], row["path"])
            self._record_provenance(row["memory_id"])
        return f"Successfully renamed {old_path} to {new_path}"

    # ------------------------------------------------------------------
    # Reads for other components
    # ------------------------------------------------------------------

    def versions(self, path: str) -> list[MemoryVersion]:
        """What writes to *path* replaced, newest first.

        For a live file, its whole history (also from before a rename); for a
        deleted one, the versions recorded under that path, ending with the
        text it had when deleted.

        Raises:
            MemoryValidationError: *path* is not a valid memory path.
        """
        path = normalize_memory_path(path)
        rows = self._db.query(
            """
            MATCH (u:User {user_id: $user_id})
            OPTIONAL MATCH (u)-[:HAS_MEMORY]->(live:Memory {path: $path})
            WITH u, live.memory_id AS live_id
            MATCH (u)-[:HAS_MEMORY_VERSION]->(v:MemoryVersion)
            WHERE (live_id IS NOT NULL AND v.memory_id = live_id) OR (live_id IS NULL AND v.path = $path)
            RETURN v.path AS path, v.content AS content, v.replaced_at AS replaced_at,
                   v.session_id AS session_id, coalesce(v.deleted, false) AS deleted
            ORDER BY replaced_at DESC
            """,
            params={"user_id": self.user_id, "path": path},
        )
        return [
            MemoryVersion(row["path"], row["content"], row["replaced_at"], row["session_id"], row["deleted"])
            for row in rows
        ]

    def files(self, directory: str = MEMORY_ROOT) -> list[MemoryFile]:
        """Every file under *directory*, at any depth, ordered by path.

        Raises:
            MemoryValidationError: *directory* is not a valid memory path.
        """
        directory = normalize_memory_path(directory)
        rows = self._db.query(
            """
            MATCH (m:Memory {user_id: $user_id})
            WHERE m.path STARTS WITH $prefix
            RETURN m.path AS path, m.content AS content, m.updated_at AS updated_at
            ORDER BY path
            """,
            params={"user_id": self.user_id, "prefix": directory + "/"},
        )
        return [MemoryFile(row["path"], row["content"], row["updated_at"] or "") for row in rows]

    # ------------------------------------------------------------------
    # Graph access
    # ------------------------------------------------------------------

    def _content(self, path: str) -> str | None:
        rows = self._db.query(
            "MATCH (m:Memory {user_id: $user_id, path: $path}) RETURN m.content AS content",
            params={"user_id": self.user_id, "path": path},
        )
        return rows[0]["content"] if rows else None

    def _existing_file(self, path: str) -> str:
        content = self._content(path)
        if content is None:
            raise MemoryCommandError(f"Error: The path {path} does not exist. Please provide a valid path.")
        return content

    def _is_directory(self, path: str) -> bool:
        if path == MEMORY_ROOT:
            return True
        rows = self._db.query(
            """
            MATCH (m:Memory {user_id: $user_id})
            WHERE m.path STARTS WITH $prefix
            RETURN count(m) > 0 AS found
            """,
            params={"user_id": self.user_id, "prefix": path + "/"},
        )
        return bool(rows and rows[0]["found"])

    def _file_ancestor(self, path: str) -> str | None:
        """The first ancestor of *path* that is a file, since a file can't hold other files."""
        segments = path.split("/")
        ancestors = ["/".join(segments[:depth]) for depth in range(3, len(segments))]
        if not ancestors:
            return None
        rows = self._db.query(
            """
            MATCH (m:Memory {user_id: $user_id})
            WHERE m.path IN $ancestors
            RETURN m.path AS path ORDER BY size(path) LIMIT 1
            """,
            params={"user_id": self.user_id, "ancestors": ancestors},
        )
        return rows[0]["path"] if rows else None

    def _listing(self, directory: str) -> str:
        sizes: dict[str, int] = {}
        total = 0
        for file in self.files(directory):
            relative = file.path[len(directory) + 1 :].split("/")
            if any(segment.startswith(".") or segment == "node_modules" for segment in relative):
                continue
            size = len(file.content.encode("utf-8"))
            total += size
            for depth in range(1, min(len(relative), _LISTING_DEPTH) + 1):
                entry = "/".join(relative[:depth]) + ("/" if depth < len(relative) else "")
                sizes[entry] = sizes.get(entry, 0) + size
        lines = [f"{_human_size(total)}\t{directory}"]
        lines += [f"{_human_size(sizes[entry])}\t{directory}/{entry}" for entry in sorted(sizes)]
        return (
            f"Here're the files and directories up to 2 levels deep in {directory}, "
            "excluding hidden items and node_modules:\n" + "\n".join(lines)
        )

    def _write(self, path: str, content: str, *, expected: str | None = None) -> None:
        """Store *content* at *path*, creating the file when *expected* is None.

        With *expected*, the write applies only if the file still holds that
        text: an edit computed from a stale read fails instead of silently
        discarding what another session wrote in between.
        """
        if not content.strip():
            raise MemoryCommandError(f"Error: {path} would be empty; delete it instead.")
        size = len(content.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            raise MemoryCommandError(
                f"Error: {path} would be {_human_size(size)}, over the {_human_size(MAX_FILE_BYTES)} limit. "
                "Split it into smaller files."
            )
        now = _now()
        if expected is None:
            rows = self._db.query(
                """
                MERGE (u:User {user_id: $user_id})
                MERGE (m:Memory {user_id: $user_id, path: $path})
                ON CREATE SET m.memory_id = $memory_id, m.created_at = $now
                WITH u, m, m.content AS previous
                SET m.content = $content, m.updated_at = $now
                MERGE (u)-[:HAS_MEMORY]->(m)
                """
                + _KEEP_PREVIOUS
                + """
                RETURN m.memory_id AS memory_id
                """,
                params={
                    "user_id": self.user_id,
                    "path": path,
                    "memory_id": str(uuid4()),
                    "content": content,
                    "now": now,
                    "session_id": self.session_id,
                },
            )
        else:
            rows = self._db.query(
                """
                MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory {path: $path})
                WHERE m.content = $expected
                WITH u, m, m.content AS previous
                SET m.content = $content, m.updated_at = $now
                """
                + _KEEP_PREVIOUS
                + """
                RETURN m.memory_id AS memory_id
                """,
                params={
                    "user_id": self.user_id,
                    "path": path,
                    "expected": expected,
                    "content": content,
                    "now": now,
                    "session_id": self.session_id,
                },
            )
            if not rows:
                raise MemoryCommandError(f"Error: {path} changed while it was being edited. View it again and retry.")
        memory_id = rows[0]["memory_id"]
        self._link_project(memory_id, path)
        self._record_provenance(memory_id)
        self._prune_versions(memory_id)

    def _prune_versions(self, memory_id: str) -> None:
        """Keep the newest :data:`MAX_VERSIONS` versions of one file."""
        self._db.query(
            """
            MATCH (:User {user_id: $user_id})-[:HAS_MEMORY_VERSION]->(v:MemoryVersion {memory_id: $memory_id})
            WITH v ORDER BY v.replaced_at DESC
            SKIP $keep
            DETACH DELETE v
            """,
            params={"user_id": self.user_id, "memory_id": memory_id, "keep": MAX_VERSIONS},
        )

    def _link_project(self, memory_id: str, path: str) -> None:
        """Point the memory at the project its path is under, and nowhere else."""
        self._db.query(
            "MATCH (m:Memory {memory_id: $memory_id})-[r:ABOUT]->(:Project) DELETE r",
            params={"memory_id": memory_id},
        )
        if key := project_key(path):
            self._db.query(
                """
                MATCH (m:Memory {memory_id: $memory_id})
                MERGE (p:Project {key: $key})
                MERGE (m)-[:ABOUT]->(p)
                """,
                params={"memory_id": memory_id, "key": key},
            )

    def _record_provenance(self, memory_id: str) -> None:
        if not self.session_id:
            return
        self._db.query(
            """
            MERGE (s:Session {session_id: $session_id})
            WITH s
            MATCH (m:Memory {memory_id: $memory_id})
            MERGE (s)-[:PRODUCED_MEMORY]->(m)
            """,
            params={"session_id": self.session_id, "memory_id": memory_id},
        )


# Appended to a write that has bound u (the owner), m (the file) and previous
# (its text before the write): keeps that text as a version when the write
# changed it. Part of the same query, so the version and the write commit together.
_KEEP_PREVIOUS = """
                WITH u, m, previous
                FOREACH (_ IN CASE WHEN previous IS NOT NULL AND previous <> $content THEN [1] ELSE [] END |
                    CREATE (u)-[:HAS_MEMORY_VERSION]->(v:MemoryVersion {
                        memory_id: m.memory_id, path: m.path, content: previous,
                        replaced_at: $now, session_id: $session_id, deleted: false
                    })
                    CREATE (m)-[:PREVIOUS_VERSION]->(v)
                )
                WITH m
"""


def _path(path: str) -> str:
    try:
        return normalize_memory_path(path)
    except MemoryValidationError as exc:
        raise MemoryCommandError(f"Error: {exc}") from exc


def _required(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise MemoryCommandError(f"Error: `{name}` is required and must be a string")
    return value


def _view_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(isinstance(v, int) for v in value):
        raise MemoryCommandError("Error: `view_range` must be two integers, [start_line, end_line]")
    return value[0], value[1]


def _insert_line(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MemoryCommandError("Error: `insert_line` is required and must be an integer")
    return value


def _occurrence_lines(content: str, needle: str) -> list[int]:
    lines, start = [], 0
    while (position := content.find(needle, start)) != -1:
        lines.append(content[:position].count("\n") + 1)
        start = position + 1
    return lines


def _human_size(size: int) -> str:
    """Sizes as the reference implementation prints them: ``512B``, ``1.5K``, ``2M``."""
    if size == 0:
        return "0B"
    units = ["B", "K", "M", "G"]
    index = min((size.bit_length() - 1) // 10, len(units) - 1)
    scaled = size / 1024**index
    return f"{int(scaled)}{units[index]}" if scaled == int(scaled) else f"{scaled:.1f}{units[index]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
