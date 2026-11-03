"""What a harness's model sees of its memory files at session start.

A harness's built-in memory loads an index file into context when a session
starts. With Context Graph as the memory backend, the session-start hook
does that instead, but generates the index from the files themselves, so
the model never has to maintain it and it can't go stale:

- one line per file in the user-wide root and the current project's folder:
  its path and a description (the ``description:`` frontmatter, or its first line);
- the full text of files marked ``pin: true``, for rules that must never
  depend on the model choosing to open a file;
- where the current project's folder is, and how to use the memory.

All of it fits in what the harness shows its model of session-start text:
Claude Code shows a hook's first 10,000 characters and otherwise only a
2,000-character preview the model is never told to read past. Pinned text
takes at most half of that; the index narrows its descriptions before it
drops any file, and says how many it dropped.

The current project is keyed by its git remote, so every clone and every
teammate's checkout of a repository lands in the same folder.
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .models import MEMORY_ROOT

if TYPE_CHECKING:
    from .memory_store import MemoryFile, MemoryStore

#: Index lines shown at most, matching the budget Claude Code gives its own memory index.
MAX_INDEX_LINES = 200
#: Pinned text shown at most, in characters, likewise matching that budget's 25KB.
MAX_PINNED_CHARS = 25_000
#: Description widths an index tries, widest first, before it drops files: a path with no description
#: still tells the model the file exists and can be opened.
_DESCRIPTION_WIDTHS = (150, 100, 60, 0)
_GIT_TIMEOUT_SECONDS = 2
_PROJECTS = f"{MEMORY_ROOT}/projects"
_SCP_REMOTE_RE = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>.+)$")
_UNSAFE_KEY_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def project_key_for(directory: str | Path) -> str | None:
    """The memory folder key for the repository *directory* is in, or None outside any.

    Uses the ``origin`` remote, normalized so SSH and HTTPS clones agree
    (``github.com_memgraph_ai-toolkit``); without one, the repository's
    directory name. Outside a git repository, the directory's own name.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    remote = _git(directory, "remote", "get-url", "origin")
    if remote:
        return _key_from_remote(remote)
    toplevel = _git(directory, "rev-parse", "--show-toplevel")
    return safe_key(Path(toplevel).name if toplevel else directory.name)


def render_index(store: MemoryStore, project: str | None, *, guidance: str, max_chars: int | None = None) -> str:
    """The session-start text: *guidance*, the project folder, pinned files and the index.

    *max_chars* is how much of it the harness shows; None leaves only the
    index's line cap and the pinned-text cap.
    """
    budget = math.inf if max_chars is None else max_chars
    project_folder = f"{_PROJECTS}/{project}" if project else None
    files = [file for file in store.files() if _in_scope(file.path, project_folder)]

    lines = [guidance]
    if project_folder:
        lines.append(f"Current project folder: {project_folder}/")
    if not files:
        lines.append(f"No memory files yet in {MEMORY_ROOT}" + (f" or {project_folder}/." if project_folder else "."))
        return "\n".join(lines)

    # Pinned text before the index: a harness that cuts hook context cuts from
    # the end, and rules that must always apply matter more than the tail of a
    # listing. Half the budget at most, so the index stays visible too.
    pinned = [file for file in files if _frontmatter(file.content).get("pin", "").lower() == "true"]
    if pinned:
        lines += _pinned(pinned, min(MAX_PINNED_CHARS, budget // 2))

    lines.append("Memory index (view a file for its full text):")
    lines += index_lines(files, budget - _length(lines), limit=MAX_INDEX_LINES, folder=MEMORY_ROOT)
    return "\n".join(lines)


def index_lines(files: list[MemoryFile], room: float, *, limit: int, folder: str) -> list[str]:
    """One line per file, at most *limit* of them, in at most *room* characters counting newlines.

    Descriptions narrow before any file is dropped; the files left out are
    counted in a last line that points at *folder*.
    """
    shown = files[:limit]
    entries: list[str] = []
    for width in _DESCRIPTION_WIDTHS:
        entries = [_entry(file, width) for file in shown]
        if _length(lines := entries + _more(len(files) - len(entries), folder)) <= room:
            return lines
    while entries:
        entries.pop()
        if _length(lines := entries + _more(len(files) - len(entries), folder)) <= room:
            return lines
    return _more(len(files), folder)


def _pinned(files: list[MemoryFile], budget: float) -> list[str]:
    """Pinned files in full while *budget* lasts; a file that doesn't fit, and those after it, are named instead."""
    lines = ["Pinned memories (always apply):"]
    for file in files:
        block = [f'<memory path="{file.path}">', _body(file.content).strip(), "</memory>"]
        if _length(block) > budget:
            lines.append(f"[{file.path} and further pinned files cut at the pinned-memory budget; view them.]")
            break
        lines += block
        budget -= _length(block)
    return lines


def _entry(file: MemoryFile, width: int) -> str:
    return f"- {file.path} — {describe(file.content, width)}" if width else f"- {file.path}"


def _more(count: int, folder: str) -> list[str]:
    return [f"- ...and {count} more; view {folder} to list them."] if count else []


def _length(lines: list[str]) -> int:
    """Characters *lines* take in the joined text, each with its newline."""
    return sum(len(line) + 1 for line in lines)


def _in_scope(path: str, project_folder: str | None) -> bool:
    """Root files always; project files only for the current project."""
    if not path.startswith(_PROJECTS + "/"):
        return True
    return project_folder is not None and path.startswith(project_folder + "/")


def _frontmatter(content: str) -> dict[str, str]:
    """Flat ``key: value`` pairs from a leading ``---`` block; nested keys are read as if top-level.

    Nested keys matter because Claude Code's own memory files nest ``type``
    under ``metadata:``, and imported files keep that shape.
    """
    if not content.startswith("---\n"):
        return {}
    end = content.find("\n---", 4)
    if end == -1:
        return {}
    fields: dict[str, str] = {}
    for line in content[4:end].splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and value.strip():
            fields.setdefault(key.strip(), value.strip().strip("\"'"))
    return fields


def _body(content: str) -> str:
    """*content* without its leading frontmatter block, if it has one."""
    if content.startswith("---\n") and (end := content.find("\n---", 4)) != -1:
        return content[end + 4 :].lstrip("\n")
    return content


def describe(content: str, width: int = _DESCRIPTION_WIDTHS[0]) -> str:
    """A file's one-line index entry, at most *width* characters: its frontmatter ``description``, else its first line."""
    description = _frontmatter(content).get("description")
    if not description:
        first = next((line.strip() for line in _body(content).splitlines() if line.strip()), "")
        description = first.lstrip("#").strip() or "(empty)"
    return description if len(description) <= width else description[: width - 1] + "…"


def _key_from_remote(remote: str) -> str:
    remote = remote.strip()
    if "://" in remote:
        remote = remote.split("://", 1)[1].split("@", 1)[-1]
    elif match := _SCP_REMOTE_RE.match(remote):
        remote = f"{match['host']}/{match['path']}"
    host, _, path = remote.partition("/")
    host = host.split(":", 1)[0].lower()
    path = path.strip("/").removesuffix(".git")
    return safe_key(f"{host}/{path}" if path else host)


def safe_key(name: str) -> str:
    """One path segment: separators become ``_``, and nothing a path check rejects survives."""
    key = _UNSAFE_KEY_CHARS_RE.sub("_", name.replace("/", "_")).strip("._") or "project"
    return key


def _git(directory: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout.strip()
    return output if completed.returncode == 0 and output else None
