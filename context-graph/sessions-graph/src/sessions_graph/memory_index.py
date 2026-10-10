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

The current project is keyed by its git remote, so every clone and every
teammate's checkout of a repository lands in the same folder.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .models import MEMORY_ROOT

if TYPE_CHECKING:
    from .memory_store import MemoryStore

#: Index lines shown at most, matching the budget Claude Code gives its own memory index.
MAX_INDEX_LINES = 200
#: Pinned text shown at most, in characters, likewise matching that budget's 25KB.
MAX_PINNED_CHARS = 25_000
_DESCRIPTION_CHARS = 150
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


def render_index(store: MemoryStore, project: str | None, *, guidance: str) -> str:
    """The session-start text: *guidance*, the project folder, the index, and pinned files."""
    project_folder = f"{_PROJECTS}/{project}" if project else None
    files = [file for file in store.files() if _in_scope(file.path, project_folder)]

    lines = [guidance]
    if project_folder:
        lines.append(f"Current project folder: {project_folder}/")
    if not files:
        lines.append(f"No memory files yet in {MEMORY_ROOT}" + (f" or {project_folder}/." if project_folder else "."))
        return "\n".join(lines)

    lines.append("Memory index (view a file for its full text):")
    shown = files[:MAX_INDEX_LINES]
    lines += [f"- {file.path} — {_description(file.content)}" for file in shown]
    if len(files) > len(shown):
        lines.append(f"- ...and {len(files) - len(shown)} more; view {MEMORY_ROOT} to list them.")

    pinned = [file for file in files if _frontmatter(file.content).get("pin", "").lower() == "true"]
    if pinned:
        lines.append("Pinned memories (always apply):")
        budget = MAX_PINNED_CHARS
        for file in pinned:
            body = _body(file.content).strip()
            if len(body) > budget:
                lines.append(f"[{file.path} and further pinned files cut at the pinned-memory budget; view them.]")
                break
            lines += [f'<memory path="{file.path}">', body, "</memory>"]
            budget -= len(body)
    return "\n".join(lines)


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


def _description(content: str) -> str:
    description = _frontmatter(content).get("description")
    if not description:
        first = next((line.strip() for line in _body(content).splitlines() if line.strip()), "")
        description = first.lstrip("#").strip() or "(empty)"
    return description if len(description) <= _DESCRIPTION_CHARS else description[: _DESCRIPTION_CHARS - 1] + "…"


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
